# orchestrator.py

# ---------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------

from typing import TypedDict, Annotated
from langchain_core.messages import SystemMessage

# LangChain message type
from langchain_core.messages import BaseMessage

# Local Ollama LLM integration
from langchain_ollama import ChatOllama

# LangGraph components
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages

# ToolNode executes the tools selected by the LLM
# tools_condition decides whether we should execute
# a tool or finish the graph.
from langgraph.prebuilt import ToolNode, tools_condition

# Import the tools we created earlier
from services import agenttools

SYSTEM_PROMPT = """
You are an chat  assistant for our company.

You have access to two tools:

1. sql_query
   Use this when the user asks about employee-specific or structured
   HR information stored in the SQL database.USE CompanyDB for employee-specific structured data such as:
   - salary
   - attendance percentage etc and make sure you know the table names correctly before querying the database. 

   Examples:
   - employee salary
   - employee attendance
   - employee leave balance
   - employee department
   - employee manager
   - employee details

2. vector_search
   Use this when the user asks about engineering,marketing,finance,general company information.
   The vector search results will only give results based on the role of the user. For example, if the user is from HR, they will only get results related to HR policies and not engineering or finance policies.
   So let user know that they will only get results based on their role and if they want to get results from other departments, they should contact the respective department.
   
   Examples:
   - leave policy
   - attendance policy
   - work from home policy
   - company benefits
   - HR rules
   - employee policies

Important rules:

- Do NOT ask for an employee ID when the user is asking about a
  general company policy.
- For general policy questions, call vector_search.
- For employee-specific information, call sql_query.
- If the question requires both employee data and company policy,
  use both tools.
- Do not make up company policies or employee information.
- Use the information returned by the tools to answer the user.
-If you get zero results from vector search, tell the user that you could not find any information related to their query and suggest them to contact the respective department for more information.
"""

# ---------------------------------------------------------
# 2. DEFINE THE GRAPH STATE
# ---------------------------------------------------------


class State(TypedDict):
    """
    State represents the information that flows through
    our LangGraph.

    'messages' contains the complete conversation:

        User question
            ↓
        Qwen response
            ↓
        Tool call
            ↓
        Tool result
            ↓
        Qwen final response
    """

    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------
# 3. CREATE THE QWEN MODEL
# ---------------------------------------------------------
from services.llm import get_llm

llm = get_llm()


# ---------------------------------------------------------
# 4. REGISTER OUR TOOLS
# ---------------------------------------------------------


def build_tools_for_role(current_user_role=None):
    role = (
        (current_user_role or agenttools.CURRENT_USER_ROLE or "general").strip().lower()
    )
    sql_tool = agenttools.get_sql_tool(role)
    print("CURRENT USER ROLE:", repr(role))

    tools = []
    if sql_tool is not None:
        tools.append(sql_tool)
    tools.append(agenttools.vector_search)
    return tools


# ---------------------------------------------------------
# 5. GIVE THE TOOLS TO QWEN
# ---------------------------------------------------------

"""
This does NOT execute the tools.

It tells Qwen:

    "These tools are available to you."

Qwen can then decide whether it wants to call:

    sql_query(...)
or
    vector_search(...)
"""


llm_with_tools = None
tools = []
tool_node = None
graph = None


def call_llm(state: State):

    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]

    response = llm_with_tools.invoke(messages)

    print("Tool calls:")
    print(response.tool_calls)

    return {"messages": [response]}


def refresh_agent_tools():
    global llm_with_tools, tool_node, graph, tools

    tools = build_tools_for_role(agenttools.CURRENT_USER_ROLE)
    llm_with_tools = llm.bind_tools(tools)
    for tool in tools:
        print("TOOL NAME:", tool.name)
        print("TOOL SCHEMA:", tool.args_schema.model_json_schema())

    tool_node = ToolNode(tools)

    graph_builder = StateGraph(State)
    graph_builder.add_node("llm", call_llm)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_edge(START, "llm")
    graph_builder.add_conditional_edges("llm", tools_condition)
    graph_builder.add_edge("tools", "llm")
    graph = graph_builder.compile()


refresh_agent_tools()


# ---------------------------------------------------------
# 13. FUNCTION TO ASK OUR AGENT A QUESTION
# ---------------------------------------------------------


def ask_agent(question: str):
    """
    Send a question to the LangGraph agent.
    """

    result = graph.invoke({"messages": [{"role": "user", "content": question}]})

    # The final message should contain Qwen's answer.
    return result["messages"][-1].content


# ---------------------------------------------------------
# 14. COMMAND-LINE INTERFACE
# ---------------------------------------------------------

if __name__ == "__main__":

    while True:

        question = input("You: ")

        # Exit the program
        if question.lower() == "exit":
            print("Goodbye!")
            break

        try:

            # Send question to LangGraph
            answer = ask_agent(question)

            print("\nAgent:")
            print(answer)
            print()

        except Exception as e:

            print("\nError:")
            print(e)
            print()
