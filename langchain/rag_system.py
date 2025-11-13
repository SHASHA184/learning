#!/usr/bin/env python3
"""
Complete RAG (Retrieval-Augmented Generation) System

A production-ready RAG implementation using LangChain with:
- Document ingestion from multiple sources
- Vector store creation and persistence (Chroma)
- Similarity search and retrieval
- Answer generation with source citations
- Performance metrics and monitoring
- Command-line interface

Usage:
    # Index documents
    python rag_system.py --mode index --source ./docs/

    # Query the system
    python rag_system.py --mode query --question "What is machine learning?"

    # Interactive mode
    python rag_system.py --mode interactive

Requirements:
    pip install langchain langchain-openai langchain-community chromadb pypdf python-dotenv
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


@dataclass
class QueryResult:
    """Structure for RAG query results with metrics."""
    question: str
    answer: str
    sources: List[str]
    retrieval_time: float
    generation_time: float
    total_time: float
    num_chunks_retrieved: int


class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'


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


class RAGSystem:
    """
    Retrieval-Augmented Generation System.

    Handles document ingestion, vector storage, and question answering
    with proper error handling and performance monitoring.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-3.5-turbo",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k_retrieval: int = 4
    ):
        """
        Initialize the RAG system.

        Args:
            persist_directory: Directory to persist the vector store
            embedding_model: OpenAI embedding model name
            llm_model: OpenAI LLM model name
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            k_retrieval: Number of chunks to retrieve for context
        """
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_retrieval = k_retrieval

        # Initialize components
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )

        self.vectorstore: Optional[Chroma] = None
        self.qa_chain = None

    def load_documents(self, source_path: str) -> List[Document]:
        """
        Load documents from a file or directory.

        Args:
            source_path: Path to file or directory containing documents

        Returns:
            List[Document]: Loaded documents

        Raises:
            FileNotFoundError: If source path doesn't exist
            ValueError: If no documents could be loaded
        """
        path = Path(source_path)

        if not path.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")

        documents = []

        try:
            if path.is_file():
                # Load single file
                if path.suffix.lower() == '.pdf':
                    loader = PyPDFLoader(str(path))
                else:
                    loader = TextLoader(str(path))
                documents = loader.load()

            elif path.is_dir():
                # Load all text files from directory
                text_loader = DirectoryLoader(
                    str(path),
                    glob="**/*.txt",
                    loader_cls=TextLoader
                )
                documents.extend(text_loader.load())

                # Load all PDF files from directory
                pdf_loader = DirectoryLoader(
                    str(path),
                    glob="**/*.pdf",
                    loader_cls=PyPDFLoader
                )
                documents.extend(pdf_loader.load())

            if not documents:
                raise ValueError(f"No documents found in {source_path}")

            print_colored(f"Loaded {len(documents)} documents", Colors.GREEN)
            return documents

        except Exception as e:
            raise ValueError(f"Error loading documents: {e}")

    def create_sample_documents(self, output_dir: str = "./sample_docs") -> str:
        """
        Create sample documents for demonstration.

        Args:
            output_dir: Directory to create sample documents

        Returns:
            str: Path to created documents directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        sample_docs = {
            "machine_learning.txt": """
Machine Learning Overview

Machine learning is a subset of artificial intelligence that enables systems to learn
and improve from experience without being explicitly programmed. It focuses on the
development of computer programs that can access data and use it to learn for themselves.

Types of Machine Learning:
1. Supervised Learning: Learning from labeled data
2. Unsupervised Learning: Finding patterns in unlabeled data
3. Reinforcement Learning: Learning through interaction with an environment

Common algorithms include linear regression, decision trees, neural networks, and
support vector machines. Machine learning is used in various applications such as
image recognition, natural language processing, and recommendation systems.
""",
            "python_programming.txt": """
Python Programming Language

Python is a high-level, interpreted programming language known for its simplicity
and readability. Created by Guido van Rossum and first released in 1991, Python
emphasizes code readability with its notable use of significant indentation.

Key Features:
- Easy to learn and use
- Extensive standard library
- Dynamic typing
- Object-oriented programming support
- Large ecosystem of third-party packages

Python is widely used in web development, data science, machine learning, automation,
and scientific computing. Popular frameworks include Django, Flask, NumPy, and Pandas.
""",
            "data_structures.txt": """
Data Structures

Data structures are specialized formats for organizing, processing, and storing data.
They are fundamental to computer science and software engineering.

Common Data Structures:
1. Arrays: Fixed-size sequential collections
2. Linked Lists: Dynamic sequential collections with nodes
3. Stacks: Last-In-First-Out (LIFO) structures
4. Queues: First-In-First-Out (FIFO) structures
5. Trees: Hierarchical structures with parent-child relationships
6. Graphs: Networks of nodes connected by edges
7. Hash Tables: Key-value pair storage with fast lookup

Choosing the right data structure is crucial for algorithm efficiency and can
significantly impact the performance of software applications.
"""
        }

        for filename, content in sample_docs.items():
            file_path = output_path / filename
            file_path.write_text(content.strip())

        print_colored(f"Created {len(sample_docs)} sample documents in {output_dir}", Colors.GREEN)
        return str(output_path)

    def index_documents(self, documents: List[Document]) -> None:
        """
        Split documents and create vector store.

        Args:
            documents: List of documents to index
        """
        print_colored("Splitting documents into chunks...", Colors.CYAN)

        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        print_colored(f"Created {len(chunks)} chunks", Colors.GREEN)

        # Create vector store
        print_colored("Creating vector store (this may take a moment)...", Colors.CYAN)
        start_time = time.time()

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

        elapsed = time.time() - start_time
        print_colored(f"Vector store created in {elapsed:.2f} seconds", Colors.GREEN)
        print_colored(f"Persisted to: {self.persist_directory}", Colors.BLUE)

    def load_vectorstore(self) -> bool:
        """
        Load existing vector store from disk.

        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print_colored("Vector store loaded successfully", Colors.GREEN)
            return True
        except Exception as e:
            print_colored(f"Error loading vector store: {e}", Colors.RED)
            return False

    def setup_qa_chain(self) -> None:
        """Set up the question-answering chain."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Load or create a vector store first.")

        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.k_retrieval}
        )

        # Create prompt template
        template = """Answer the question based only on the following context.
If you cannot answer the question based on the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer: """

        prompt = ChatPromptTemplate.from_template(template)

        # Create the chain
        self.qa_chain = (
            {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_docs(self, docs: List[Document]) -> str:
        """Format retrieved documents for context."""
        return "\n\n".join(doc.page_content for doc in docs)

    def query(self, question: str) -> QueryResult:
        """
        Query the RAG system with a question.

        Args:
            question: Question to answer

        Returns:
            QueryResult: Answer with metadata and performance metrics
        """
        if not self.qa_chain:
            self.setup_qa_chain()

        start_total = time.time()

        # Retrieve relevant documents
        start_retrieval = time.time()
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.k_retrieval}
        )
        relevant_docs = retriever.invoke(question)
        retrieval_time = time.time() - start_retrieval

        # Generate answer
        start_generation = time.time()
        answer = self.qa_chain.invoke(question)
        generation_time = time.time() - start_generation

        total_time = time.time() - start_total

        # Extract sources
        sources = list(set([
            doc.metadata.get('source', 'Unknown')
            for doc in relevant_docs
        ]))

        return QueryResult(
            question=question,
            answer=answer,
            sources=sources,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time,
            num_chunks_retrieved=len(relevant_docs)
        )

    def display_result(self, result: QueryResult) -> None:
        """Display query result with formatting."""
        print_colored("\n" + "="*60, Colors.BOLD)
        print_colored("Question:", Colors.YELLOW + Colors.BOLD)
        print_colored(result.question, Colors.YELLOW)

        print_colored("\nAnswer:", Colors.GREEN + Colors.BOLD)
        print_colored(result.answer, Colors.GREEN)

        print_colored("\nSources:", Colors.BLUE + Colors.BOLD)
        for source in result.sources:
            print_colored(f"  - {source}", Colors.BLUE)

        print_colored("\nPerformance Metrics:", Colors.CYAN + Colors.BOLD)
        print_colored(f"  Retrieval Time: {result.retrieval_time:.3f}s", Colors.CYAN)
        print_colored(f"  Generation Time: {result.generation_time:.3f}s", Colors.CYAN)
        print_colored(f"  Total Time: {result.total_time:.3f}s", Colors.CYAN)
        print_colored(f"  Chunks Retrieved: {result.num_chunks_retrieved}", Colors.CYAN)
        print_colored("="*60 + "\n", Colors.BOLD)


def run_interactive_mode(rag_system: RAGSystem) -> None:
    """Run RAG system in interactive query mode."""
    print_colored("\n" + "="*60, Colors.BOLD)
    print_colored("RAG System - Interactive Mode", Colors.BOLD)
    print_colored("="*60, Colors.BOLD)
    print_colored("\nType your questions (or 'quit' to exit)\n", Colors.YELLOW)

    while True:
        try:
            question = input(f"{Colors.YELLOW}Question: {Colors.RESET}").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print_colored("\nGoodbye!\n", Colors.GREEN)
                break

            result = rag_system.query(question)
            rag_system.display_result(result)

        except KeyboardInterrupt:
            print_colored("\n\nInterrupted. Type 'quit' to exit.\n", Colors.YELLOW)
        except Exception as e:
            print_colored(f"\nError: {e}\n", Colors.RED)


def main():
    """Main entry point for the RAG system."""
    parser = argparse.ArgumentParser(
        description="RAG System - Document Question Answering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create sample documents and index them
  python rag_system.py --mode index --create-samples

  # Index documents from a directory
  python rag_system.py --mode index --source ./docs/

  # Query the system
  python rag_system.py --mode query --question "What is machine learning?"

  # Interactive mode
  python rag_system.py --mode interactive
        """
    )

    parser.add_argument(
        '--mode',
        choices=['index', 'query', 'interactive'],
        required=True,
        help='Operation mode'
    )
    parser.add_argument(
        '--source',
        type=str,
        help='Source path for documents (file or directory)'
    )
    parser.add_argument(
        '--question',
        type=str,
        help='Question to ask (query mode only)'
    )
    parser.add_argument(
        '--persist-dir',
        type=str,
        default='./chroma_db',
        help='Directory to persist vector store'
    )
    parser.add_argument(
        '--create-samples',
        action='store_true',
        help='Create sample documents for demonstration'
    )

    args = parser.parse_args()

    # Check for API key
    if not load_environment():
        print_colored(
            "Error: OPENAI_API_KEY not found in environment variables.",
            Colors.RED
        )
        sys.exit(1)

    # Initialize RAG system
    rag_system = RAGSystem(persist_directory=args.persist_dir)

    try:
        if args.mode == 'index':
            # Create sample documents if requested
            if args.create_samples:
                args.source = rag_system.create_sample_documents()

            if not args.source:
                print_colored(
                    "Error: --source required for index mode (or use --create-samples)",
                    Colors.RED
                )
                sys.exit(1)

            # Load and index documents
            documents = rag_system.load_documents(args.source)
            rag_system.index_documents(documents)
            print_colored("\nIndexing complete! You can now query the system.", Colors.GREEN)

        elif args.mode == 'query':
            if not args.question:
                print_colored("Error: --question required for query mode", Colors.RED)
                sys.exit(1)

            # Load vector store and query
            if not rag_system.load_vectorstore():
                print_colored(
                    "No indexed documents found. Run with --mode index first.",
                    Colors.RED
                )
                sys.exit(1)

            result = rag_system.query(args.question)
            rag_system.display_result(result)

        elif args.mode == 'interactive':
            # Load vector store and enter interactive mode
            if not rag_system.load_vectorstore():
                print_colored(
                    "No indexed documents found. Run with --mode index first.",
                    Colors.RED
                )
                sys.exit(1)

            run_interactive_mode(rag_system)

    except Exception as e:
        print_colored(f"Error: {e}", Colors.RED)
        sys.exit(1)


if __name__ == "__main__":
    main()
