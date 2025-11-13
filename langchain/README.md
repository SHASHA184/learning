# LangChain Tutorial Series

Complete educational series for learning LangChain from basics to advanced topics.

## Notebooks Overview

### 1. **langchain_basics.ipynb** (17KB)
Introduction to LangChain fundamentals
- What is LangChain and when to use it
- Runnable interface (.invoke, .stream, .batch, .ainvoke)
- Provider independence (OpenAI, Anthropic, Google)
- Simple chains with LCEL pipe operator
- Chat messages and streaming
- First "Hello World" examples

### 2. **langchain_models.ipynb** (25KB)
Deep dive into models and providers
- Chat Models vs LLMs (legacy)
- OpenAI, Anthropic, Google Gemini integration
- Model parameters (temperature, max_tokens)
- Streaming and async execution
- Multi-model comparison
- Token usage tracking and caching
- Function calling and embeddings
- Provider-specific features

### 3. **langchain_prompts.ipynb** (32KB)
Comprehensive prompt engineering
- ChatPromptTemplate basics
- System, Human, AI message types
- MessagesPlaceholder for dynamic history
- Few-shot prompting with examples
- Output parsers (String, JSON, Pydantic)
- Partial prompts and composition
- Best practices and common pitfalls

### 4. **langchain_lcel.ipynb** (32KB)
LangChain Expression Language mastery
- Pipe operator (`|`) fundamentals
- RunnablePassthrough for data flow
- RunnableParallel for parallel execution
- RunnableBranch for conditional logic
- RunnableLambda for custom functions
- Error handling (fallbacks, retries)
- Real-world chain examples
- Performance optimization

### 5. **langchain_rag.ipynb** (36KB)
Complete RAG (Retrieval-Augmented Generation) pipeline
- RAG architecture and use cases
- Document loaders (Text, PDF, Web)
- Text splitting strategies
- Embeddings (OpenAI, HuggingFace)
- Vector stores (Chroma, FAISS, Pinecone)
- Retrieval strategies (similarity, MMR, threshold)
- create_retrieval_chain usage
- RAG with chat history
- Advanced patterns (multi-query, parent document)

### 6. **langchain_agents.ipynb** (26KB)
Agents and intelligent tool use
- What are agents (ReAct pattern)
- Creating custom tools with @tool decorator
- Built-in tools (DuckDuckGo, Wikipedia)
- create_openai_functions_agent
- AgentExecutor with safety limits
- Agent types comparison
- Memory integration with agents
- Best practices and warnings

### 7. **langchain_memory.ipynb** (31KB)
Conversation memory patterns
- Memory types comparison
- ConversationBufferMemory (all messages)
- ConversationWindowMemory (last N)
- ConversationSummaryMemory (summarized)
- ConversationSummaryBufferMemory (hybrid)
- RunnableWithMessageHistory (modern approach)
- Persistent memory (SQLite, Redis, Postgres)
- Multi-session management
- Production memory patterns

## Learning Path

### Beginner (Start Here)
1. `langchain_basics.ipynb` - Get started with core concepts
2. `langchain_models.ipynb` - Learn about different models
3. `langchain_prompts.ipynb` - Master prompt engineering

### Intermediate
4. `langchain_lcel.ipynb` - Build complex chains
5. `langchain_rag.ipynb` - Create document Q&A systems

### Advanced
6. `langchain_agents.ipynb` - Build autonomous agents
7. `langchain_memory.ipynb` - Add conversation memory

## Prerequisites

```bash
# Install required packages
pip install langchain langchain-openai langchain-anthropic langchain-community
pip install chromadb faiss-cpu pypdf
pip install duckduckgo-search wikipedia tiktoken
```

## API Keys Required

Set environment variables:
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"  # Optional
export GOOGLE_API_KEY="your-google-key"        # Optional
```

Or set in notebooks when prompted.

## Features

Each notebook includes:
- ✅ Practical, runnable code examples
- ✅ Clear explanations and markdown cells
- ✅ Hands-on exercises at the end
- ✅ Best practices sections
- ✅ Common pitfalls and solutions
- ✅ Real-world use cases
- ✅ Resource links for deeper learning

## Notebook Format

All notebooks follow consistent structure:
1. **Introduction** - Concept overview and use cases
2. **Setup** - Installation and configuration
3. **Examples** - Progressive difficulty (10-15 examples each)
4. **Best Practices** - Do's and don'ts
5. **Common Pitfalls** - Mistakes to avoid
6. **Exercises** - Practice problems
7. **Key Takeaways** - Summary of learnings
8. **Resources** - Additional reading

## Quick Start

```bash
# Navigate to directory
cd /home/sasha/learning/langchain/

# Start Jupyter
jupyter lab

# Open langchain_basics.ipynb to begin
```

## Topics Covered

### Core Concepts
- Runnables and LCEL
- Chains and composition
- Prompts and templates
- Output parsing

### Models & Providers
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Google (Gemini)
- Provider switching

### Advanced Patterns
- RAG pipelines
- Agent systems
- Memory management
- Streaming responses
- Async/parallel execution

### Production Considerations
- Error handling
- Token management
- Caching strategies
- Monitoring and debugging
- Security best practices

## Learning Outcomes

After completing this series, you will be able to:

1. **Build LLM applications** with LangChain
2. **Design effective prompts** for different tasks
3. **Create RAG systems** for document Q&A
4. **Develop agents** that use tools intelligently
5. **Implement memory** for conversational AI
6. **Optimize chains** for performance and cost
7. **Deploy to production** with best practices

## Next Steps

After completing these notebooks:
- Build a complete LLM application
- Explore LangGraph for complex workflows
- Implement production monitoring with LangSmith
- Contribute to LangChain ecosystem
- Build custom tools and integrations

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [LangSmith](https://smith.langchain.com/) - Monitoring & debugging
- [LangChain Blog](https://blog.langchain.dev/)
- [Discord Community](https://discord.gg/langchain)

## License

Educational materials for personal learning.

---

**Happy Learning! 🚀**
