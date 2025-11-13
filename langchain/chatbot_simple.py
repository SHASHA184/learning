#!/usr/bin/env python3
"""
Simple Conversational Chatbot with Memory

A production-ready chatbot implementation using LangChain with:
- Streaming responses for better UX
- Conversation memory using RunnableWithMessageHistory
- Clean error handling and graceful shutdown
- CLI interface with colored output

Usage:
    python chatbot_simple.py

    Set environment variable OPENAI_API_KEY or create a .env file:
    export OPENAI_API_KEY="your-key-here"

Requirements:
    pip install langchain langchain-openai python-dotenv
"""

import os
import sys
from typing import Optional
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory


# Color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    USER = '\033[94m'  # Blue
    BOT = '\033[92m'   # Green
    SYSTEM = '\033[93m'  # Yellow
    ERROR = '\033[91m'  # Red


# In-memory chat history store
chat_history_store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Retrieve or create chat history for a session.

    Args:
        session_id: Unique identifier for the conversation session

    Returns:
        BaseChatMessageHistory: Chat history object for the session
    """
    if session_id not in chat_history_store:
        chat_history_store[session_id] = ChatMessageHistory()
    return chat_history_store[session_id]


def load_environment() -> bool:
    """
    Load environment variables from .env file if present.

    Returns:
        bool: True if API key is available, False otherwise
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed, rely on environment variables

    return bool(os.getenv("OPENAI_API_KEY"))


def create_chatbot(
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    streaming: bool = True
) -> RunnableWithMessageHistory:
    """
    Create a conversational chatbot with memory.

    Args:
        model_name: OpenAI model to use (default: gpt-3.5-turbo)
        temperature: Controls randomness (0.0-1.0, default: 0.7)
        streaming: Enable streaming responses (default: True)

    Returns:
        RunnableWithMessageHistory: Chatbot with conversation memory
    """
    # Initialize the language model
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        streaming=streaming
    )

    # Create prompt template with system message and conversation history
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a helpful, friendly AI assistant. "
            "Provide clear, concise answers and engage naturally in conversation. "
            "Remember the conversation history and maintain context."
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # Create the conversation chain
    chain = prompt | llm

    # Add message history to the chain
    chatbot = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )

    return chatbot


def print_colored(text: str, color: str = Colors.RESET) -> None:
    """Print text with color to terminal."""
    print(f"{color}{text}{Colors.RESET}")


def stream_response(chatbot: RunnableWithMessageHistory, user_input: str, session_id: str) -> str:
    """
    Stream the chatbot response and return the complete message.

    Args:
        chatbot: The chatbot instance
        user_input: User's message
        session_id: Session identifier for conversation history

    Returns:
        str: Complete response text
    """
    print_colored("\nAssistant: ", Colors.BOT + Colors.BOLD)

    full_response = ""
    try:
        # Stream the response
        for chunk in chatbot.stream(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        ):
            if hasattr(chunk, 'content'):
                content = chunk.content
                print(content, end='', flush=True)
                full_response += content

        print()  # New line after response
        return full_response

    except Exception as e:
        error_msg = f"\n{Colors.ERROR}Error generating response: {e}{Colors.RESET}"
        print(error_msg)
        return ""


def display_help() -> None:
    """Display help information for the chatbot."""
    help_text = """
Available commands:
    /help     - Show this help message
    /clear    - Clear conversation history
    /history  - Show conversation history
    /quit     - Exit the chatbot

Just type your message to chat with the assistant.
"""
    print_colored(help_text, Colors.SYSTEM)


def display_history(session_id: str) -> None:
    """Display the conversation history."""
    history = get_session_history(session_id)
    messages = history.messages

    if not messages:
        print_colored("\nNo conversation history yet.", Colors.SYSTEM)
        return

    print_colored("\n=== Conversation History ===", Colors.SYSTEM + Colors.BOLD)
    for msg in messages:
        if isinstance(msg, HumanMessage):
            print_colored(f"\nYou: {msg.content}", Colors.USER)
        elif isinstance(msg, AIMessage):
            print_colored(f"Assistant: {msg.content}", Colors.BOT)
    print_colored("\n=== End of History ===\n", Colors.SYSTEM + Colors.BOLD)


def clear_history(session_id: str) -> None:
    """Clear the conversation history."""
    if session_id in chat_history_store:
        chat_history_store[session_id].clear()
    print_colored("\nConversation history cleared.\n", Colors.SYSTEM)


def run_chatbot_cli(session_id: str = "default_session") -> None:
    """
    Run the chatbot in CLI mode with interactive conversation.

    Args:
        session_id: Unique identifier for the conversation session
    """
    # Check for API key
    if not load_environment():
        print_colored(
            "Error: OPENAI_API_KEY not found in environment variables.\n"
            "Please set it or create a .env file with your API key.",
            Colors.ERROR
        )
        sys.exit(1)

    # Create the chatbot
    print_colored("Initializing chatbot...", Colors.SYSTEM)
    try:
        chatbot = create_chatbot()
    except Exception as e:
        print_colored(f"Error initializing chatbot: {e}", Colors.ERROR)
        sys.exit(1)

    # Welcome message
    print_colored("\n" + "="*60, Colors.BOLD)
    print_colored("Welcome to the AI Chatbot!", Colors.BOLD)
    print_colored("="*60 + "\n", Colors.BOLD)
    print_colored("Type '/help' for commands or just start chatting!", Colors.SYSTEM)
    print_colored("Type '/quit' to exit.\n", Colors.SYSTEM)

    # Main conversation loop
    while True:
        try:
            # Get user input
            user_input = input(f"{Colors.USER}{Colors.BOLD}You: {Colors.RESET}").strip()

            # Handle empty input
            if not user_input:
                continue

            # Handle commands
            if user_input.startswith('/'):
                command = user_input.lower()

                if command == '/quit' or command == '/exit':
                    print_colored("\nGoodbye! Have a great day!\n", Colors.SYSTEM)
                    break

                elif command == '/help':
                    display_help()

                elif command == '/clear':
                    clear_history(session_id)

                elif command == '/history':
                    display_history(session_id)

                else:
                    print_colored(f"Unknown command: {user_input}", Colors.ERROR)
                    print_colored("Type '/help' for available commands.", Colors.SYSTEM)

                continue

            # Get and stream the response
            stream_response(chatbot, user_input, session_id)
            print()  # Extra newline for spacing

        except KeyboardInterrupt:
            print_colored("\n\nInterrupted. Type '/quit' to exit or continue chatting.", Colors.SYSTEM)
            print()

        except EOFError:
            print_colored("\n\nGoodbye!\n", Colors.SYSTEM)
            break

        except Exception as e:
            print_colored(f"\nUnexpected error: {e}", Colors.ERROR)
            print_colored("Please try again or type '/quit' to exit.\n", Colors.SYSTEM)


def main() -> None:
    """Main entry point for the chatbot."""
    try:
        run_chatbot_cli()
    except Exception as e:
        print_colored(f"Fatal error: {e}", Colors.ERROR)
        sys.exit(1)


if __name__ == "__main__":
    main()
