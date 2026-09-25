import pymssql
import os

THREAD_ID = "0de3231a-3589-40f7-b04e-127bf1657327"


def save_message(role, content):

    connection = None
    cursor = None

    try:
        connection = pymssql.connect(
            server="localhost",
            port=1434,
            user=os.getenv("sqladminuser"),
            password=os.getenv("sqladminpassword"),
            database=os.getenv("DB_DATABASE"),
        )

        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO Messages
                (ThreadId, Role, Content)
            VALUES
                (%s, %s, %s)
            """,
            (THREAD_ID, role, content),
        )

        connection.commit()

    except Exception as e:
        print(f"Error saving message: {e}")

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()
