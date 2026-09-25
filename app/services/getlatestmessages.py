import pymssql
import os

THREAD_ID = "0de3231a-3589-40f7-b04e-127bf1657327"


def get_latest_messages():

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
            SELECT Role, Content, SequenceNumber
            FROM
            (
                SELECT TOP 10
                    Role,
                    Content,
                    SequenceNumber
                FROM Messages
                WHERE ThreadId = %s
                ORDER BY SequenceNumber DESC
            ) AS RecentMessages
            ORDER BY SequenceNumber ASC;
            """,
            (THREAD_ID,),
        )

        messages = cursor.fetchall()

        return [{"role": row[0], "content": row[1]} for row in messages]

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()
