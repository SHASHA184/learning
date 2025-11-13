# Environment Setup

This document covers the setup and environment configuration for the learning repository.

## Virtual Environment

This project uses a Python virtual environment with specific dependencies:

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Key Dependencies

- `requests==2.32.3` - For HTTP operations in practical examples
- `tensorflow==2.20.0` - For machine learning explorations

## Development Environment

### Running Jupyter Notebooks
```bash
# Start Jupyter Lab/Notebook server
jupyter lab
# or
jupyter notebook
```

### Running Python Scripts
```bash
# Execute standalone Python files
python concurrency/multiprocessing_calculator.py
python concurrency/multithreading_calculator.py
python concurrency/multithreading_test.py
```

### Testing Implementations
```bash
# Run specific concurrency examples
python concurrency/multithreading_calculator.py
python concurrency/multiprocessing_calculator.py
```