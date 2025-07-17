from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from pydantic import SecretStr
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key
else:
    print("No Keys")


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=SecretStr(groq_api_key) if groq_api_key else None,
)


def process(state: AgentState) -> AgentState:
    """This Node Will Solve The Request To You Input"""
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    print("CURRENT STATE: ", state["messages"])
    return state


graph = StateGraph(AgentState)

graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")

while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages": conversation_history})

    conversation_history = result["messages"]

    user_input = input("Enter: ")

## There is a Big Problem With The Previous Way of Preserving The Conversation ->
## When Exiting the Chat, Everything is Gone
## So That Another Way is To Use Any Database, Files...
## Now For Simplicity I Will Use Simple File ??

with open("logging.txt", "w", encoding="utf-8") as file:
    file.write("Your Conversation Log:\n\n")

    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")

print("Conversation saved to logging.txt")
