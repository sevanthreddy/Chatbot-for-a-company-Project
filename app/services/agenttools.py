import os
import urllib.parse

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from sqlalchemy import create_engine, text
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

CURRENT_USER_ROLE = None


import re

EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def protect_pii(text: str) -> str:
    """
    Redacts email addresses from tool results.
    """

    return EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)


def get_cached_huggingface_embeddings(model_name: str):
    """
    Loads and locks the HuggingFace model weights into the machine's global RAM.
    If called again during any subsequent script rerun, it returns instantly.
    """
    import streamlit as st

    # We wrap the inner call with st.cache_resource dynamically
    @st.cache_resource(show_spinner=False)
    def _load_model(name: str):
        print(
            f"🧠 MEMORY SEED: Permanently caching local model [{name}] in global RAM..."
        )
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=name, model_kwargs={"device": "cpu"})

    return _load_model(model_name)


def get_sql_tool(current_user_role):

    if not current_user_role or current_user_role.lower() != "hr":
        return None

    @tool
    def sql_query(sql_query: str) -> str:
        """
        Execute a read-only SQL query against the ai_schema.EmployeeData view.

        IMPORTANT:
        - The ONLY allowed table/view is ai_schema.EmployeeData.
        - NEVER use employees, dbo.Employees, or any other table.
        - The SQL must be a SELECT query.
        - Use T-SQL syntax.

        Example:

        SELECT *
        FROM ai_schema.EmployeeData
        WHERE employee_id = 'FINEMP1005'
        """

        try:
            connection_string = (
                f"mssql+pymssql://"
                f"{os.getenv('AI_DB_USER')}:"
                f"{urllib.parse.quote_plus(os.getenv('AI_DB_PASSWORD'))}"
                f"@localhost:1434/"
                f"{os.getenv('DB_DATABASE')}"
            )

            engine = create_engine(connection_string)

            with engine.connect() as connection:
                result = connection.execute(text(sql_query))

                columns = list(result.keys())
                rows = result.fetchall()

                data = [dict(zip(columns, row)) for row in rows]
                result = str(data)
                return protect_pii(result)
        except Exception as e:
            return f"Error executing SQL query: {e}"

    return sql_query


@tool
def vector_search(query: str) -> str:
    """
    Search the company's internal documentation stored in Pinecone.

    IMPORTANT:
    The query argument is REQUIRED.
    """

    try:
        global CURRENT_USER_ROLE

        normalized_role = (CURRENT_USER_ROLE or "general").strip().lower()
        allowed_roles = {"engineering", "finance", "general", "hr", "marketing"}
        if normalized_role not in allowed_roles:
            normalized_role = "general"

        embeddings = get_cached_huggingface_embeddings("BAAI/bge-m3")

        vector_store = PineconeVectorStore(
            index_name=os.getenv("PINECONE_INDEX_NAME"),
            embedding=embeddings,
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        )

        results = vector_store.similarity_search(
            query=query,
            k=5,
            filter={"department": normalized_role},
        )

        if not results:
            return (
                f"No relevant documents found for the '{normalized_role}' role. "
                "This department may not have matching documentation."
            )

        response = []

        for i, result in enumerate(results, start=1):
            response.append(
                f"Result {i}\n"
                f"Content: {result.page_content}\n"
                f"Metadata: {result.metadata}"
            )

        return "\n\n".join(response)

    except Exception as e:
        return f"Error searching Pinecone: {e}"


if __name__ == "__main__":
    sql_tool = get_sql_tool("hr")
    if sql_tool:
        print(
            sql_tool.invoke(
                {
                    "sql_query": "SELECT TOP 5 employee_id, full_name, salary FROM ai_schema.EmployeeData"
                }
            )
        )
