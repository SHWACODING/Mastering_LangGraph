from typing import Annotated, Sequence, TypedDict # Annotated - Provides Additional Context Without Affecting the Type itself, Sequence - To automatically handle the state updates for sequences such as by adding new messages to chat history
from langchain_core.messages import BaseMessage   # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage   # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages  # Reducder Function - Add All Messages Without Overwirting
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
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
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def  add(a: int, b: int):
    """ This is an addition function that adds 2 numbers """
    return a + b

@tool
def subtract(a: int, b: int):
    """Subtraction function"""
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiplication function"""
    return a * b

tools = [add, subtract, multiply]

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=SecretStr(groq_api_key) if groq_api_key else None
).bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are my AI assistant, please answer my query to the best of your ability."
    )
    response = model.invoke([system_prompt] + list(state["messages"]))
    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 40 + 12, then subtract 10 and then multiply the result by 6. Also tell me a joke please.")]}

print_stream(app.stream(inputs, stream_mode="values"))

