# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a personal learning repository focused on Python programming concepts, organized into thematic learning modules. The codebase is primarily educational, using Jupyter notebooks for interactive exploration and standalone Python scripts for practical implementations.

## Quick Reference

- **Setup**: See [docs/setup.md](docs/setup.md) for environment setup and development commands
- **Architecture**: See [docs/architecture.md](docs/architecture.md) for code structure and patterns
- **Learning Plans**: See [learning-plans/README.md](learning-plans/README.md) for structured study guides
- **OOP Principles**: See [docs/oop-principles.md](docs/oop-principles.md) for the four pillars of object-oriented programming
- **Concurrency**: See [docs/concurrency.md](docs/concurrency.md) for threading, multiprocessing, and asyncio
- **Design Patterns**: See [docs/design-patterns.md](docs/design-patterns.md) for GoF pattern implementations
- **Interview Prep**: See [docs/interview-prep.md](docs/interview-prep.md) for advanced Python concepts
- **Machine Learning**: See [docs/machine-learning.md](docs/machine-learning.md) for mathematical foundations

## Project Structure

The repository is organized into four main learning areas:

- **`concurrency/`** - Python concurrency patterns (threading, multiprocessing, asyncio)
- **`design_patterns/`** - GoF design pattern implementations
- **`interview/`** - Advanced Python concepts for interview preparation
- **`ml/`** - Mathematical foundations for machine learning
- **`docs/`** - Detailed documentation for each module
- **`learning-plans/`** - Structured learning plans with timelines and progress tracking

## Quick Start

```bash
# Setup environment
source venv/bin/activate
pip install -r requirements.txt

# Start Jupyter for interactive learning
jupyter lab

# Run performance comparisons
python concurrency/multithreading_calculator.py
python concurrency/multiprocessing_calculator.py
```