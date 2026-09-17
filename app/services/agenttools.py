import os
import urllib.parse

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from sqlalchemy import create_engine, text
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


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

    Table:
    dbo.hr_data

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

    Always use dbo.hr_data when querying employee information.

    Always write standard T-SQL queries.

    Examples:

    For employee salary:
    SELECT salary
    FROM dbo.hr_data
    WHERE employee_id = 'FINEMP1000'

    For employee leave balance:
    SELECT leave_balance
    FROM dbo.hr_data
    WHERE employee_id = 'FINEMP1000'
    """

    try:
        connection_string = (
            f"mssql+pymssql://"
            f"{os.getenv('DB_USER')}:"
            f"{urllib.parse.quote_plus(os.getenv('DB_PASSWORD'))}"
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

    Always provide a meaningful search query describing
    exactly what information you want to retrieve.

    Use this tool for:
    - company technical architecture
    - technology stack
    - engineering documentation
    - system architecture
    - APIs and services
    - infrastructure
    - engineering processes
    - HR policies
    - company policies
    - SOPs
    - internal documentation

    Examples:

    User:
    "What is the company's technical architecture?"

    Correct tool call:
    vector_search(
        query="company technical architecture technology stack systems APIs infrastructure"
    )

    User:
    "What is the leave policy?"

    Correct tool call:
    vector_search(
        query="company employee leave policy"
    )

    Never call vector_search without the query argument.
    """

    try:

        # Load model from cache
        embeddings = get_cached_huggingface_embeddings("BAAI/bge-m3")

        # Connect to existing Pinecone index
        vector_store = PineconeVectorStore(
            index_name=os.getenv("PINECONE_INDEX_NAME"),
            embedding=embeddings,
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        )

        # Similarity search
        results = vector_store.similarity_search(query=query, k=5)

        if not results:
            return "No relevant documents found."

        # Format results for the LLM
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
