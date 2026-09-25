import os
import urllib.parse

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from sqlalchemy import create_engine, text
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

CURRENT_USER_ROLE = "general"


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


@tool
def sql_query(sql_query: str) -> str:
    """
    Executes a SQL SELECT query against the company HR database.

    Database:
    companydb

    Schema:
    ai_schema

    views:
    ai_schema.EmployeeData

    Columns available:
    employee_id,
    full_name,
    role,
    department,
    email,
    location,
    date_of_birth,
    date_of_joining,
    manager_id,
    salary,
    leave_balance,
    leaves_taken,
    attendance_pct,
    performance_rating,
    last_review_date

    Employee IDs are strings such as:
    FINEMP1000, FINEMP1001, FINEMP1002.

    Always use ai_schema.EmployeeData when querying employee information.

    Always write standard T-SQL queries.

    Examples:

    For employee salary:
    SELECT salary
    FROM ai_schema.EmployeeData
    WHERE employee_id = 'FINEMP1000'

    For employee leave balance:
    SELECT leave_balance
    FROM ai_schema.EmployeeData
    WHERE employee_id = 'FINEMP1000'
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

            return str(data)

    except Exception as e:
        return f"Error executing SQL query: {e}"


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
    print(
        sql_query.invoke(
            {
                "sql_query": "SELECT TOP 5 employee_id, full_name, salary FROM dbo.hr_data"
            }
        )
    )
