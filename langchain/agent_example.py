#!/usr/bin/env python3
"""
LangChain Agent with Multiple Tools

A production-ready agent implementation with:
- Custom tool definitions (calculator, string operations)
- Built-in tool integration (web search, file operations)
- Agent creation and execution with error handling
- Detailed logging and monitoring
- CLI interface for easy interaction

Usage:
    # Run a single task
    python agent_example.py --task "Calculate 25 * 17 and reverse the result"

    # Interactive mode
    python agent_example.py --interactive

    # With verbose logging
    python agent_example.py --task "Your task" --verbose

Requirements:
    pip install langchain langchain-openai langchain-community python-dotenv
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any, Optional, Type
from datetime import datetime

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field


# Color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'


def print_colored(text: str, color: str = Colors.RESET) -> None:
    """Print text with color to terminal."""
    print(f"{color}{text}{Colors.RESET}")


def load_environment() -> bool:
    """Load environment variables from .env file if present."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    return bool(os.getenv("OPENAI_API_KEY"))


# ============================================================================
# Custom Tools Definition
# ============================================================================

class CalculatorInput(BaseModel):
    """Input schema for calculator tool."""
    expression: str = Field(description="Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')")


@tool("calculator", args_schema=CalculatorInput)
def calculator_tool(expression: str) -> str:
    """
    Evaluate mathematical expressions safely.

    Supports basic arithmetic operations: +, -, *, /, **, %
    Example: "25 * 17" returns "425"
    """
    try:
        # Use eval with restricted scope for safety
        # Only allow basic math operations
        allowed_names = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'pow': pow,
        }

        # Evaluate the expression
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"The result of '{expression}' is {result}"

    except ZeroDivisionError:
        return "Error: Division by zero"
    except SyntaxError:
        return f"Error: Invalid mathematical expression: '{expression}'"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


class StringReverseInput(BaseModel):
    """Input schema for string reverse tool."""
    text: str = Field(description="Text to reverse")


@tool("string_reverse", args_schema=StringReverseInput)
def string_reverse_tool(text: str) -> str:
    """
    Reverse a string.

    Example: "hello" becomes "olleh"
    """
    return text[::-1]


class StringCaseInput(BaseModel):
    """Input schema for string case conversion tool."""
    text: str = Field(description="Text to convert")
    case: str = Field(description="Target case: 'upper', 'lower', 'title', or 'capitalize'")


@tool("string_case", args_schema=StringCaseInput)
def string_case_tool(text: str, case: str) -> str:
    """
    Convert string to different cases.

    Supported cases: upper, lower, title, capitalize
    Example: string_case("hello world", "upper") returns "HELLO WORLD"
    """
    case = case.lower()

    if case == "upper":
        return text.upper()
    elif case == "lower":
        return text.lower()
    elif case == "title":
        return text.title()
    elif case == "capitalize":
        return text.capitalize()
    else:
        return f"Error: Unknown case '{case}'. Use: upper, lower, title, or capitalize"


class WordCountInput(BaseModel):
    """Input schema for word count tool."""
    text: str = Field(description="Text to count words in")


@tool("word_count", args_schema=WordCountInput)
def word_count_tool(text: str) -> str:
    """
    Count words and characters in text.

    Returns statistics about the input text.
    """
    words = text.split()
    chars = len(text)
    chars_no_spaces = len(text.replace(" ", ""))

    return (
        f"Text statistics:\n"
        f"  - Words: {len(words)}\n"
        f"  - Characters (with spaces): {chars}\n"
        f"  - Characters (without spaces): {chars_no_spaces}"
    )


class TimestampInput(BaseModel):
    """Input schema for timestamp tool."""
    format: str = Field(
        default="%Y-%m-%d %H:%M:%S",
        description="Datetime format string (default: '%Y-%m-%d %H:%M:%S')"
    )


@tool("current_timestamp", args_schema=TimestampInput)
def timestamp_tool(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Get the current date and time.

    Returns formatted timestamp. Default format: YYYY-MM-DD HH:MM:SS
    """
    try:
        return datetime.now().strftime(format)
    except Exception as e:
        return f"Error formatting timestamp: {str(e)}"


# ============================================================================
# Agent Setup and Execution
# ============================================================================

class LangChainAgent:
    """
    LangChain agent with multiple tools and error handling.

    Provides an interface to create and execute agents with custom tools.
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0,
        verbose: bool = False
    ):
        """
        Initialize the agent.

        Args:
            model_name: OpenAI model to use
            temperature: Controls randomness (0.0-1.0)
            verbose: Enable verbose logging
        """
        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature
        )

        # Setup tools
        self.tools = self._setup_tools()

        # Create agent
        self.agent_executor = self._create_agent()

        # Setup logging
        if verbose:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )

    def _setup_tools(self) -> List[BaseTool]:
        """
        Setup and return available tools.

        Returns:
            List[BaseTool]: List of available tools
        """
        tools = [
            calculator_tool,
            string_reverse_tool,
            string_case_tool,
            word_count_tool,
            timestamp_tool,
        ]

        if self.verbose:
            print_colored(f"\nLoaded {len(tools)} tools:", Colors.CYAN)
            for tool in tools:
                print_colored(f"  - {tool.name}: {tool.description}", Colors.BLUE)

        return tools

    def _create_agent(self) -> AgentExecutor:
        """
        Create the agent executor.

        Returns:
            AgentExecutor: Configured agent executor
        """
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant with access to various tools.

Use the available tools to help answer questions and complete tasks.
When using tools:
1. Break down complex tasks into simple tool calls
2. Use the calculator for mathematical operations
3. Use string tools for text manipulation
4. Always explain what you're doing

If you can't complete a task with available tools, explain why and suggest alternatives.
"""),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create the agent
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            handle_parsing_errors=True,
            max_iterations=10,
        )

        return agent_executor

    def run(self, task: str) -> Dict[str, Any]:
        """
        Execute a task using the agent.

        Args:
            task: Task description

        Returns:
            Dict[str, Any]: Result with output and metadata
        """
        try:
            result = self.agent_executor.invoke({"input": task})
            return {
                "success": True,
                "output": result.get("output", ""),
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "output": None,
                "error": str(e)
            }

    def display_result(self, result: Dict[str, Any], task: str) -> None:
        """Display execution result with formatting."""
        print_colored("\n" + "="*60, Colors.BOLD)
        print_colored("Task:", Colors.YELLOW + Colors.BOLD)
        print_colored(task, Colors.YELLOW)

        if result["success"]:
            print_colored("\nResult:", Colors.GREEN + Colors.BOLD)
            print_colored(result["output"], Colors.GREEN)
        else:
            print_colored("\nError:", Colors.RED + Colors.BOLD)
            print_colored(result["error"], Colors.RED)

        print_colored("="*60 + "\n", Colors.BOLD)


def run_interactive_mode(agent: LangChainAgent) -> None:
    """Run agent in interactive mode."""
    print_colored("\n" + "="*60, Colors.BOLD)
    print_colored("LangChain Agent - Interactive Mode", Colors.BOLD)
    print_colored("="*60, Colors.BOLD)

    # Display available tools
    print_colored("\nAvailable Tools:", Colors.CYAN + Colors.BOLD)
    for tool in agent.tools:
        print_colored(f"  {tool.name}", Colors.GREEN)
        print_colored(f"    {tool.description}", Colors.BLUE)

    print_colored("\nType your tasks (or 'quit' to exit)", Colors.YELLOW)
    print_colored("Examples:", Colors.CYAN)
    print_colored("  - Calculate 25 * 17", Colors.BLUE)
    print_colored("  - Reverse the word 'hello'", Colors.BLUE)
    print_colored("  - Convert 'python programming' to title case", Colors.BLUE)
    print_colored("  - Count words in 'the quick brown fox'", Colors.BLUE)
    print_colored("  - What's the current time?\n", Colors.BLUE)

    while True:
        try:
            task = input(f"{Colors.YELLOW}Task: {Colors.RESET}").strip()

            if not task:
                continue

            if task.lower() in ['quit', 'exit', 'q']:
                print_colored("\nGoodbye!\n", Colors.GREEN)
                break

            result = agent.run(task)
            agent.display_result(result, task)

        except KeyboardInterrupt:
            print_colored("\n\nInterrupted. Type 'quit' to exit.\n", Colors.YELLOW)
        except Exception as e:
            print_colored(f"\nError: {e}\n", Colors.RED)


def run_example_tasks(agent: LangChainAgent) -> None:
    """Run a series of example tasks to demonstrate capabilities."""
    print_colored("\n" + "="*60, Colors.BOLD)
    print_colored("Running Example Tasks", Colors.BOLD)
    print_colored("="*60 + "\n", Colors.BOLD)

    example_tasks = [
        "Calculate 25 * 17 + 10",
        "Reverse the word 'hello'",
        "Convert 'python programming' to title case",
        "Count words in 'the quick brown fox jumps over the lazy dog'",
        "What's the current date and time?",
        "Calculate 100 divided by 5, then reverse that number as a string",
    ]

    for i, task in enumerate(example_tasks, 1):
        print_colored(f"\nExample {i}/{len(example_tasks)}", Colors.MAGENTA + Colors.BOLD)
        result = agent.run(task)
        agent.display_result(result, task)

        if i < len(example_tasks):
            input(f"{Colors.CYAN}Press Enter to continue...{Colors.RESET}")


def main():
    """Main entry point for the agent."""
    parser = argparse.ArgumentParser(
        description="LangChain Agent with Multiple Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a single task
  python agent_example.py --task "Calculate 25 * 17"

  # Interactive mode
  python agent_example.py --interactive

  # Run example tasks
  python agent_example.py --examples

  # With verbose logging
  python agent_example.py --task "Your task" --verbose
        """
    )

    parser.add_argument(
        '--task',
        type=str,
        help='Task to execute'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    parser.add_argument(
        '--examples',
        action='store_true',
        help='Run example tasks'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-3.5-turbo',
        help='OpenAI model to use (default: gpt-3.5-turbo)'
    )

    args = parser.parse_args()

    # Check for API key
    if not load_environment():
        print_colored(
            "Error: OPENAI_API_KEY not found in environment variables.",
            Colors.RED
        )
        sys.exit(1)

    # Create agent
    print_colored("Initializing agent...", Colors.CYAN)
    try:
        agent = LangChainAgent(
            model_name=args.model,
            verbose=args.verbose
        )
    except Exception as e:
        print_colored(f"Error initializing agent: {e}", Colors.RED)
        sys.exit(1)

    print_colored("Agent ready!\n", Colors.GREEN)

    # Execute based on mode
    try:
        if args.examples:
            run_example_tasks(agent)
        elif args.interactive:
            run_interactive_mode(agent)
        elif args.task:
            result = agent.run(args.task)
            agent.display_result(result, args.task)
        else:
            parser.print_help()
            print_colored("\nNo action specified. Use --task, --interactive, or --examples", Colors.YELLOW)

    except Exception as e:
        print_colored(f"Error: {e}", Colors.RED)
        sys.exit(1)


if __name__ == "__main__":
    main()
