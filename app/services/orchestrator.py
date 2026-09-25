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
from services.agenttools import sql_query, vector_search

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

tools = [sql_query, vector_search]


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

llm_with_tools = llm.bind_tools(tools)


# ---------------------------------------------------------
# 6. CREATE THE LLM NODE
# ---------------------------------------------------------


def call_llm(state: State):

    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]

    response = llm_with_tools.invoke(messages)

    # DEBUG: show exactly what Qwen generated

    # DEBUG: specifically show tool calls
    print("Tool calls:")
    print(response.tool_calls)

    return {"messages": [response]}


# ---------------------------------------------------------
# 7. CREATE THE TOOL NODE
# ---------------------------------------------------------

"""
ToolNode is responsible for actually executing tools.

For example, if Qwen generates:

    sql_query(
        "SELECT * FROM hr_data WHERE employee_id = 101"
    )

ToolNode executes our Python function:

    sql_query(...)

and puts the result back into the messages.
"""

tool_node = ToolNode(tools)


# ---------------------------------------------------------
# 8. BUILD THE LANGGRAPH
# ---------------------------------------------------------

graph_builder = StateGraph(State)


# Add our two types of nodes:

# Node 1:
# Qwen / LLM
graph_builder.add_node("llm", call_llm)

# Node 2:
# Tool executor
graph_builder.add_node("tools", tool_node)


# ---------------------------------------------------------
# 9. DEFINE THE GRAPH FLOW
# ---------------------------------------------------------

"""
The initial flow is:

START
  ↓
LLM
"""

graph_builder.add_edge(START, "llm")


# ---------------------------------------------------------
# 10. CONDITIONAL ROUTING
# ---------------------------------------------------------

"""
After Qwen responds, tools_condition checks:

Did Qwen request a tool?

        YES
         ↓
       tools

        NO
         ↓
       END
"""

graph_builder.add_conditional_edges("llm", tools_condition)


# ---------------------------------------------------------
# 11. AFTER TOOL EXECUTION, GO BACK TO QWEN
# ---------------------------------------------------------

"""
After a tool executes, its result goes back to Qwen.

For example:

User:
"Who is employee 101?"

        ↓

Qwen:
"I need SQL."

        ↓

SQL tool:
"Employee 101 is John..."

        ↓

Qwen:
"Employee 101 is John..."
"""

graph_builder.add_edge("tools", "llm")


# ---------------------------------------------------------
# 12. COMPILE THE GRAPH
# ---------------------------------------------------------

graph = graph_builder.compile()


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
