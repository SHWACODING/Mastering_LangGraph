from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from pydantic import SecretStr
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    os.environ['GROQ_API_KEY'] = groq_api_key
else:
    print("No Keys")

class AgentState(TypedDict):
    messages: List[HumanMessage]

llm = ChatGroq(
    api_key=SecretStr(groq_api_key) if groq_api_key else None,
    model="llama-3.3-70b-versatile"
)

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    return state


graph = StateGraph(AgentState)

graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

user_input = input("Enter: ")

while user_input != 'exit':
    agent.invoke({
        "messages": [HumanMessage(content=user_input)]
    })
    user_input = input("Enter: ")
