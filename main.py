import os
from typing import cast, Callable, List, Literal, Optional

import chainlit as cl
from chainlit.user_session import UserSession
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt


os.environ["GOOGLE_API_KEY"]="AIzaSyAjcU7MUcRpZFZJF1uEh44ApVJm1MKJEjQ"
gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("messages", [])

    mcp_client = MultiServerMCPClient({
        "kubectl": {
            "command": "npx",
            "args": ["mcp-server-kubernetes"],
            "transport": "stdio"
        },
    })
    tools = await mcp_client.get_tools()
    agent = await get_chat_agent(tools=tools)
    model = gemini.bind_tools(tools)

    cl.user_session.set("agent", agent)
    cl.user_session.set("model", model)


@cl.on_message
async def on_message(message: cl.Message):
    agent = cast(CompiledGraph, cl.user_session.get("agent"))
    model = cl.user_session.get("model")
    thread_id = cl.context.session.id
    config = RunnableConfig(configurable={"thread_id": thread_id, "model": model}, callbacks=[cl.LangchainCallbackHandler()])

    interrupt = None

    messages = cl.user_session.get("messages")
    messages.append(message)
    cl.user_session.set("messages", messages)

    response = cl.Message(content="")

    stream = agent.astream(
        {"messages": ChatUtils.get_langchain_messages(cl.user_session)},
        config=config,
        stream_mode=['messages', 'updates'],
    )

    while stream:
        async for stream_mode, pack in stream:
            if stream_mode == 'messages':
                msg, metadata = pack
                if (
                    msg.content
                    and not isinstance(msg, HumanMessage)
                    and metadata["langgraph_node"] == "final"
                ):
                    await response.stream_token(msg.content)
                stream = None

            else:
                if '__interrupt__' in pack:
                    interrupt = pack['__interrupt__'][0]
                    res = await cl.AskActionMessage(
                        content=interrupt.value,
                        actions=[
                            cl.Action(name="continue", payload={"value": "continue"}, label="✅ Continue"),
                            cl.Action(name="cancel", payload={"value": "cancel"}, label="❌ Cancel"),
                        ],
                    ).send()
                    
                    if res['payload']['value'] == 'continue':
                        cmd = Command(resume=True)
                    else:
                        cmd = Command(update={"messages": [HumanMessage("I don't want to call a tool")]}, resume=False)
                        
                    stream = agent.astream(
                        cmd,
                        config=config,
                        stream_mode=['messages', 'updates'],
                    )
                else:
                    stream = None

    messages.append(response)
    cl.user_session.set("messages", messages)

    await response.send()


class ChatUtils:
    @classmethod
    def covert_to_human_message(cls, message: cl.Message):
        return HumanMessage(content=message.content)

    @classmethod
    def covert_to_ai_message(cls, message: cl.Message):
        return AIMessage(content=message.content)
    
    @classmethod
    def convert_to_system_message(cls, message: cl.Message):
        return SystemMessage(content=message.content)
    
    @classmethod
    def get_langchain_messages(cls, session: UserSession) -> list[BaseMessage]:
        messages = session.get("messages")

        langchain_messages = []
        for message in messages:
            if message.type == "user_message":
                langchain_messages.append(cls.covert_to_human_message(message))
            elif message.type == "assistant_message":
                langchain_messages.append(cls.covert_to_ai_message(message))
            elif message.type == "system_message":
                langchain_messages.append(cls.convert_to_system_message(message))
            else:
                raise ValueError(f"Invalid message type: {message.type}")

        return langchain_messages

    @classmethod
    def save_message(cls, message: cl.Message, session: UserSession):
        messages = session.get("messages")
        if len(messages) > 40:
            messages.pop(0)

        messages.append(message)
        session.set("messages", messages)

    @classmethod
    def get_messages(cls, session: UserSession) -> list[cl.Message]:
        messages = session.get("messages")
        return messages


def call_model(state: MessagesState, config: dict):
    model = config['configurable']['model']
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def call_final_model(state: MessagesState, config: dict):
    model = config['configurable']['model']
    messages = state["messages"]
    last_ai_message = messages[-1]
    response = model.invoke(
        [
            SystemMessage("Rewrite this in the voice of Al Roker"),
            HumanMessage(last_ai_message.content),
        ]
    )
    response.id = last_ai_message.id
    return {"messages": [response]}


def should_continue(state: MessagesState) -> Literal["tools", "final"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        permit = interrupt("Are you sure you want to call a tool?")
        if permit:
            return "tools"
        else:
            return "final"
        
    return "final"


async def get_chat_agent(tools: Optional[List[Callable]] = None) -> CompiledGraph:
    tool_node = ToolNode(tools=tools)

    builder = StateGraph(MessagesState)

    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)
    builder.add_node("final", call_final_model)

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        should_continue,
    )

    builder.add_edge("tools", "call_model")
    builder.add_edge("final", END)

    graph = builder.compile(checkpointer=InMemorySaver())

    return graph