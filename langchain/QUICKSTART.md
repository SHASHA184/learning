# LangChain Examples - Quick Start Guide

Get up and running with these LangChain examples in 5 minutes.

## Step 1: Install Dependencies

```bash
cd /home/sasha/learning/langchain
pip install -r requirements.txt
```

## Step 2: Set Up API Key

Create a `.env` file with your OpenAI API key:

```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

Or export it as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Step 3: Try the Examples

### Chatbot (Easiest to Start)

```bash
python chatbot_simple.py
```

Try these interactions:
- Ask: "What is Python?"
- Ask: "What are its main features?" (tests memory)
- Type: `/history` (view conversation)
- Type: `/quit` (exit)

### Agent (Most Interactive)

```bash
# Run examples
python agent_example.py --examples

# Or interactive mode
python agent_example.py --interactive
```

Try these tasks:
- "Calculate 25 * 17"
- "Reverse the word 'hello'"
- "Convert 'python programming' to title case"
- "What's the current time?"

### RAG System (Document Q&A)

```bash
# Create sample documents and index them
python rag_system.py --mode index --create-samples

# Query the documents
python rag_system.py --mode query --question "What is machine learning?"

# Or use interactive mode
python rag_system.py --mode interactive
```

## Common Commands Cheat Sheet

### Chatbot
```bash
# Start chatbot
./chatbot_simple.py

# In-chat commands
/help      # Show help
/history   # View conversation
/clear     # Clear history
/quit      # Exit
```

### RAG System
```bash
# Index your documents
python rag_system.py --mode index --source /path/to/docs/

# Single query
python rag_system.py --mode query --question "Your question?"

# Interactive mode
python rag_system.py --mode interactive
```

### Agent
```bash
# Single task
python agent_example.py --task "Calculate 25 * 17"

# Interactive mode
python agent_example.py --interactive

# With detailed logging
python agent_example.py --task "Your task" --verbose

# Run examples
python agent_example.py --examples
```

## Troubleshooting

### "API key not found"
Make sure you've set `OPENAI_API_KEY` in `.env` or as an environment variable.

### "No module named 'langchain'"
Run `pip install -r requirements.txt`

### "Vector store not found" (RAG)
Run `python rag_system.py --mode index --create-samples` first

## Next Steps

1. Read [README.md](README.md) for detailed documentation
2. Explore the code in each script
3. Modify and experiment with parameters
4. Try with your own documents (RAG)
5. Create custom tools (Agent)

## File Overview

- `chatbot_simple.py` (298 lines) - Conversational chatbot with memory
- `rag_system.py` (551 lines) - Document question-answering system
- `agent_example.py` (504 lines) - Multi-tool agent
- `README.md` (520 lines) - Comprehensive documentation
- `requirements.txt` - Python dependencies

Total: 1,873 lines of production-ready code!
