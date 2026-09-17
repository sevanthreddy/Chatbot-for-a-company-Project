import streamlit as st

from services.orchestrator import ask_agent

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(page_title="HR Assistant", page_icon="🏢", layout="centered")


# ---------------------------------------------------------
# Page title
# ---------------------------------------------------------

st.title("🏢 HR Assistant")
st.caption("Ask questions about employees, HR policies, and company documentation.")


# ---------------------------------------------------------
# Initialize chat history
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


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

    with st.chat_message("user"):
        st.markdown(question)

    # ---------------------------------------------
    # Call LangGraph agent
    # ---------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:
                answer = ask_agent(question)

            except Exception as e:
                answer = f"Something went wrong: {e}"

        st.markdown(answer)

    # ---------------------------------------------
    # Store assistant response
    # ---------------------------------------------

    st.session_state.messages.append({"role": "assistant", "content": answer})
