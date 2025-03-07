from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from state import State
from agent import Assistant, internal_tools, assistant_runnable
builder = StateGraph(State)
from utils import create_tool_node_with_fallback
from langchain_core.runnables import Runnable, RunnableConfig


# Define nodes: these do the work
builder.add_node("assistant", Assistant(assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(internal_tools))
# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

memory = MemorySaver()
part_1_graph = builder.compile(checkpointer=memory)

def run_assistant(user_message):
    """
    Runs the assistant with a given user input.
    """
    initial_state = State(messages=[("user", user_message)])
    config = RunnableConfig(configurable={"passenger_id": "12345"})  # Example configurable data
    response = part_1_graph.invoke(initial_state, config)
    return response

if __name__ == "__main__":
    print("Swiss Airlines AI Assistant is now running...")
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting AI Assistant.")
            break
        response = run_assistant(user_input)
        print("\nAssistant:", response)
