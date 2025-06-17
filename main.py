import os
from typing import cast, Callable, List, Literal, Optional

import chainlit as cl
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt


os.environ["GOOGLE_API_KEY"]="<YOUR_GOOGLE_API_KEY>"

cl.on_app_startup

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("messages", [])
    agent = await get_chat_agent()
    cl.user_session.set("agent", agent)


@cl.on_message
async def on_message(message: cl.Message):
    agent = cast(CompiledGraph, cl.user_session.get("agent"))
    config = RunnableConfig(
        configurable={"thread_id": cl.context.session.id}, 
        callbacks=[cl.LangchainCallbackHandler()],
    )

    messages = cl.user_session.get("messages")
    messages.append(HumanMessage(content=message.content))
    cl.user_session.set("messages", messages)

    interrupt = None
    response = cl.Message(content="")

    stream = agent.astream(
        {"messages": messages},
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
                            cl.Action(name="continue", payload={"value": "continue"}, label="Continue"),
                            cl.Action(name="cancel", payload={"value": "cancel"}, label="Cancel"),
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

    messages.append(AIMessage(content=response.content))
    cl.user_session.set("messages", messages)

    await response.send()


async def get_chat_agent(tools: Optional[List[Callable]] = None) -> CompiledGraph:
    mcp_client = MultiServerMCPClient({
        "kubectl": {
            "command": "npx",
            "args": ["mcp-server-kubernetes"],
            "transport": "stdio"
        },
    })
    tools = await mcp_client.get_tools()

    gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    model = gemini.bind_tools(tools)

    def call_model(state: MessagesState):
        nonlocal model
        messages = state["messages"]
        response = model.invoke(messages)
        return {"messages": [response]}


    def call_final_model(state: MessagesState):
        nonlocal model
        messages = state["messages"]
        last_ai_message = messages[-1]
        response = model.invoke(
            [
                SystemMessage("Rewrite this in the voice of Sid in Ice Age"),
                HumanMessage(last_ai_message.content),
            ]
        )
        response.id = last_ai_message.id
        return {"messages": [response]}


    def should_continue(state: MessagesState) -> Literal["tools", "final"]:
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            permit = interrupt(f"I need to call **{last_message.tool_calls[0]['name']}**. Are you sure you want to call a tool?")
            if permit:
                return "tools"
            else:
                return "final"
            
        return "final"
    
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

    graph.get_graph().draw_png("graph.png")

    return graph
