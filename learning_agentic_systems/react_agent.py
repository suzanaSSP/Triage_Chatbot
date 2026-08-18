from typing import Annotated, Sequence, TypedDict
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
# pyrefly: ignore [missing-import]
from langchain_core.tools import tool
# pyrefly: ignore [missing-import]
from langgraph.graph.message import add_messages
# pyrefly: ignore [missing-import]
from langgraph.graph import StateGraph, END
# pyrefly: ignore [missing-import]
from langgraph.prebuilt import ToolNode
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b:int):
    """This is an addition function that adds two numbers together"""
    return a+ b

@tool
def subtractions(a: int, b:int):
    """This is a subtraction function that subtracts two numbers together"""
    return a- b

@tool
def multiply(a: int, b:int):
    """This is a multiplication function that multiplies two numbers together"""
    return a* b


tools_list = [add, subtractions, multiply]

model = ChatGroq(model="llama-3.3-70b-versatile").bind_tools(tools_list)

def model_call(state: AgentState)-> AgentState:
    system_prompt = SystemMessage(content=
    "You are my AI assistant, please answer my query to the best of your ability."
    )
    response = model.invoke([system_prompt] + state['messages'])
    return {'messages': [response]}


def should_continue(state: AgentState)-> AgentState:
    messages = state['messages']
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools_list)
graph.add_node("tools_list", tool_node)
graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue":"tools_list",
        "end": END,
    },
)

graph.add_edge("tools_list", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s['messages'][-1]
        if isinstance(message, tuple):
            print(message)

        else:
            message.pretty_print()

inputs = {"messages": [("user", "Do this operation (2+4)*(8-1)")]}
print_stream(app.stream(inputs, stream_mode="values"))