import os
import json
import datetime
import uuid
from typing import Annotated

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()

# Define available tools
import internal_tool as tools

internal_tools = [
    tools.get_external_tools,
    tools.get_time,
    tools.calculate,
    tools.http_request
]

# Define State
class State(dict):
    messages: Annotated[list[AnyMessage], add_messages]
    execution_steps: list
    step_index: int
    retry_count: int

from langchain_core.messages import AIMessage, HumanMessage

class Planner:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        configuration = config.get("configurable", {})
        user_info = configuration.get("user_info", "Unknown User")
        state["user_info"] = user_info

        planning_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an AI planner. Break down the user's request into structured steps.\n"
                    "Each step must specify the tool to use, parameters, and expected output.\n"
                    "Ensure all steps are clear and executable.\n"
                    "\n\nUser Info: {user_info}\n"
                    "Current time: {time}.",
                ),
                ("placeholder", "{messages}"),
            ]
        ).partial(time=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # 🛠️ **Convert messages to correct format**
        formatted_messages = [
            HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
            for msg in state["messages"]
        ]

        plan = self.runnable.invoke(formatted_messages)  # ✅ Now passing correct format

        return {
            "execution_steps": plan,
            "step_index": 0,
            "retry_count": 0  # Initialize retry count
        }

# =============== EXECUTOR (Step-by-Step with Retry) ===============
class Executor:
    MAX_RETRIES = 3  # Set max retry attempts

    def __init__(self, tools: list):
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, state: State):
        execution_steps = state.get("execution_steps", [])
        step_index = state.get("step_index", 0)
        retry_count = state.get("retry_count", 0)

        if step_index >= len(execution_steps):
            return {"executed_results": state.get("executed_results", []), "validation_needed": True}

        step = execution_steps[step_index]
        tool_name = step.get("tool")
        params = step.get("parameters", {})

        if tool_name not in self.tools_by_name:
            return {"error": f"Tool '{tool_name}' not found", "retry_count": retry_count + 1}

        # Execute the tool
        tool_result = self.tools_by_name[tool_name].invoke(params)
        success = tool_result is not None and tool_result != "ERROR"

        if success:
            # Move to the next step
            return {
                "executed_results": state.get("executed_results", []) + [{"tool": tool_name, "result": tool_result}],
                "step_index": step_index + 1,
                "retry_count": 0  # Reset retry count
            }
        elif retry_count < self.MAX_RETRIES:
            # Retry the same step
            return {"retry_count": retry_count + 1}
        else:
            # Move to validation even if retries fail
            return {"error": f"Step {step_index} failed after {self.MAX_RETRIES} retries", "validation_needed": True}

# =============== VALIDATOR ===============
class Validator:
    def __init__(self, validation_model):
        self.validation_model = validation_model  # Use a cheaper LLM for validation

    def __call__(self, state: State):
        executed_results = state.get("executed_results", [])
        validated_results = []

        for result in executed_results:
            validation_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an AI validator. Your task is to verify if the tool execution was correct.\n"
                        "Compare the actual result with the expected output and flag any errors.\n\n"
                        "Tool: {tool}\n"
                        "Expected Output: {expected_output}\n"
                        "Actual Result: {result}\n\n"
                        "Return 'VALID' if correct, otherwise explain the error.",
                    )
                ]
            )

            validation_response = self.validation_model.invoke(
                {
                    "tool": result["tool"],
                    "expected_output": result.get("expected_output", ""),
                    "result": result["result"],
                }
            )

            validated_results.append(
                {
                    "tool": result["tool"],
                    "result": result["result"],
                    "validation": validation_response,
                }
            )

        return {"validated_results": validated_results}

# =============== AI MODELS ===============
llm_low = ChatOpenAI(model="gpt-4o-mini")  # Cheap AI for validation
llm = ChatOpenAI(model="gpt-4o")  # Main AI model

# =============== PROMPT ===============
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are 'J.A.R.V.ish', an AI assistant managing and assisting {user_info}.\n"
            "Use internal and external tools for execution.\n"
            "Expand queries if needed and ensure accuracy.\n\n"
            "Current user:\n<User>\n{user_info}\n</User>\n"
            "Current time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

assistant_runnable = primary_assistant_prompt | llm.bind_tools(internal_tools)

# =============== LANGGRAPH PIPELINE ===============
graph_builder = StateGraph(State)

# Define Nodes
graph_builder.add_node("planner", Planner(llm))
graph_builder.add_node("executor", Executor(internal_tools))
graph_builder.add_node("validator", Validator(llm_low))  # Use a cheap AI model for validation

# Define Execution Flow
graph_builder.add_edge(START, "planner")
graph_builder.add_edge("planner", "executor")
graph_builder.add_conditional_edges(
    "executor",
    lambda state: "executor" if state.get("step_index", 0) < len(state.get("execution_steps", [])) else "validator",
    {"executor": "executor", "validator": "validator"}
)
graph_builder.add_edge("validator", END)

# Compile Graph
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# =============== EXECUTION FUNCTION ===============
def run_assistant(user_message):
    initial_state = {
        "messages": [{"role": "user", "content": user_message}],
        "execution_steps": [],
        "step_index": 0,
        "retry_count": 0
    }

    config = RunnableConfig(
        configurable={
            "thread_id": str(uuid.uuid4()),  # Unique session ID
            "user_info": "Jaehak Kim",
        }
    )

    return graph.stream(initial_state, config)

# =============== RUNNING THE CHATBOT ===============
if __name__ == "__main__":
    print("J.A.R.V.ish AI Assistant is now running...")
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response = run_assistant(user_input)
        print("\nAssistant:", response)
