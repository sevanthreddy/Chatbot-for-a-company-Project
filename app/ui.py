import streamlit as st

from services.agenttools import CURRENT_USER_ROLE
from services.saveconversations import save_message
from services.orchestrator import ask_agent

USERS = {
    "Tony": {"password": "password123", "role": "engineering"},
    "Bruce": {"password": "securepass", "role": "marketing"},
    "Sam": {"password": "financepass", "role": "finance"},
    "Peter": {"password": "pete123", "role": "engineering"},
    "Sid": {"password": "sidpass123", "role": "marketing"},
    "Natasha": {"password": "hrpass123", "role": "hr"},
}

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(page_title="HR Assistant", page_icon="🏢", layout="centered")


if "user" not in st.session_state:
    st.session_state.user = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------
# Login screen
# ---------------------------------------------------------

if st.session_state.user is None:
    st.title("🔐 HR Assistant Login")
    st.caption(
        "Sign in to ask questions about employees, policies, and company documentation."
    )

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            user = USERS.get(username)
            if user and user["password"] == password:
                st.session_state.user = {"username": username, "role": user["role"]}
                import services.agenttools as agenttools

                agenttools.CURRENT_USER_ROLE = user["role"]
                st.rerun()
            else:
                st.error("Invalid username or password")

    st.stop()


# ---------------------------------------------------------
# Authenticated layout
# ---------------------------------------------------------

user = st.session_state.user

st.sidebar.write(f"Logged in as: {user['username']}")
st.sidebar.write(f"Role: {user['role']}")

if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.session_state.messages = []
    st.rerun()

st.title("🏢 HR Assistant")
st.caption(
    f"Welcome {user['username']} ({user['role']}). Ask questions about employees, HR policies, and company documentation."
)


# ---------------------------------------------------------
# Display previous messages
# ---------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------------
# Chat input
# ---------------------------------------------------------

question = st.chat_input("Ask your question...")


if question:
    # ---------------------------------------------
    # Display user's message
    # ---------------------------------------------

    st.session_state.messages.append({"role": "user", "content": question})
    save_message("user", question)

    with st.chat_message("user"):
        st.markdown(question)

    # ---------------------------------------------
    # Call LangGraph agent
    # ---------------------------------------------

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = ask_agent(question)
                save_message("assistant", answer)
            except Exception as e:
                answer = f"Something went wrong: {e}"

        st.markdown(answer)

    # ---------------------------------------------
    # Store assistant response
    # ---------------------------------------------

    st.session_state.messages.append({"role": "assistant", "content": answer})
