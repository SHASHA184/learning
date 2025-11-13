# Code Architecture

This document describes the overall architecture and patterns used in the learning repository.

## Learning Module Pattern

Each learning area follows a consistent pattern:
- **Jupyter notebooks** (.ipynb) for interactive exploration and documented learning
- **Python scripts** (.py) for practical implementations and performance testing
- **Self-contained examples** that can be run independently

## Project Structure

The repository is organized into four main learning areas:

### Concurrency Module (`concurrency/`)
Comprehensive study of Python concurrency patterns including:
- **Multithreading** - Shared memory, lightweight context switching
- **Multiprocessing** - Separate memory spaces, true parallelism
- **Asyncio** - Single-threaded, cooperative multitasking

Each approach includes both educational notebooks explaining concepts and practical calculator implementations for performance comparison.

### Design Patterns Module (`design_patterns/`)
Implementation and exploration of common design patterns:
- Factory Pattern
- Builder Pattern
- Abstract Factory Pattern

Implements classical GoF patterns with:
- Abstract base classes and concrete implementations
- Client code examples demonstrating usage
- Comprehensive documentation in notebook format

### Interview Preparation Module (`interview/`)
Python interview preparation materials covering advanced topics:
- Generators and iterators
- Descriptors
- Method Resolution Order (MRO)

### Machine Learning Module (`ml/`)
Machine learning fundamentals focusing on mathematical foundations:
- Vector operations and manipulations
- Matrix mathematics
- Core mathematical operations using NumPy

## File Naming Conventions

- `*.ipynb` - Interactive learning notebooks with explanations and examples
- `*_calculator.py` - Performance comparison implementations
- `*_test.py` - Testing and validation scripts
- Pattern-specific naming for design pattern implementations

## Development Notes

- All notebooks are self-contained and can be run independently
- Performance testing scripts compare different concurrency approaches
- Virtual environment (`venv/`) contains all necessary dependencies
- Git ignores standard Python artifacts, Jupyter checkpoints, and performance test outputs