import streamlit as st
from assistant import Assistant
from langchain_core.messages import SystemMessage
import sys
from dotenv import load_dotenv
import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

def cli_chat(assistant):
    print("Welcome to the AI Assistant. Type 'exit' to end the conversation.")
    messages = [SystemMessage(content=assistant.system_message)]
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        try:
            response, messages = assistant.chat(user_input, messages)
            print(f"AI: {response}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Error details:")
            print(traceback.format_exc())
            print("Please try again or type 'exit' to quit.")

def streamlit_chat():
    st.title("AI Assistant")

    # Initialize assistant
    if "assistant" not in st.session_state:
        try:
            st.session_state.assistant = Assistant()
        except Exception as e:
            st.error(f"Error initializing the assistant: {str(e)}")
            st.error("Error details:")
            st.error(traceback.format_exc())
            return

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content=st.session_state.assistant.system_message)
        ]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages[1:]:  # Skip the system message
        with st.chat_message(message.type):
            st.markdown(message.content)

    # React to user input
    if prompt := st.chat_input("What would you like to do today?"):
        # Display user message in chat message container
        st.chat_message("human").markdown(prompt)

        # Get AI response
        with st.chat_message("ai"):
            try:
                response, st.session_state.messages = st.session_state.assistant.chat(prompt, st.session_state.messages)
                st.markdown(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Error details:")
                st.error(traceback.format_exc())

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--cli':
        try:
            assistant = Assistant()
            cli_chat(assistant)
        except Exception as e:
            print(f"Error initializing the assistant: {str(e)}")
            print("Error details:")
            print(traceback.format_exc())
            print("Please check your .env file or set the required environment variables and try again.")
    else:
        streamlit_chat()

if __name__ == "__main__":
    main()