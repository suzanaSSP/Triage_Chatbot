from typing import Dict, TypedDict, List, Union
# pyrefly: ignore [missing-import]
from langgraph.graph import StateGraph, START, END
import random
# pyrefly: ignore [missing-import]
from langchain_core.messages import HumanMessage, AIMessage
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()

# AGENT I
# class AgentState(TypedDict):
#     messages: List[HumanMessage]

llm = ChatGroq(model="llama-3.3-70b-versatile")

# def process(state: AgentState)-> AgentState:
#     response = llm.invoke(state["messages"])
#     print(f"\nAI: {response.content}")
#     return state

# graph = StateGraph(AgentState)
# graph.add_node("process", process)
# graph.add_edge(START, "process")
# graph.add_edge("process", END)
# app = graph.compile()

# user_input = input("Enter: ")
# while user_input != "exit":  
#     app.invoke({"messages": [HumanMessage(content=user_input)]})
#     user_input = input("Enter")


# AGENT II

class SecondAgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

def second_process(state: SecondAgentState)-> SecondAgentState:
     response = llm.invoke(state["messages"])
     state["messages"].append(AIMessage(content=response.content))
     print(f"\nAi: {response.content}")
     
     return state

graph = StateGraph(SecondAgentState)
graph.add_node("process", second_process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = []


user_input = input("Enter:")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({'messages': conversation_history})

    #print(result['messages'])
    conversation_history = result["messages"]

    user_input = input("Enter: ")

with open("logging.txt", "w") as file:
    file.write("Your conversation log: \n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")

print("Conversation saved to logging.txt")