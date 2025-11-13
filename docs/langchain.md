# LangChain - Complete Learning Guide

A comprehensive guide to building LLM applications with LangChain, covering RAG systems, agents, chains, and production best practices.

## Table of Contents

- [Overview](#overview)
- [Core Concepts](#core-concepts)
- [Components](#components)
- [When to Use What](#when-to-use-what)
- [Common Patterns](#common-patterns)
- [Provider Comparison](#provider-comparison)
- [Performance Considerations](#performance-considerations)
- [Security Best Practices](#security-best-practices)
- [Production Checklist](#production-checklist)
- [Common Pitfalls](#common-pitfalls)
- [Resources](#resources)

---

## Overview

LangChain is a framework for developing applications powered by language models. It provides abstractions and tools to build complex LLM applications through composable components.

### What is LangChain?

LangChain enables developers to:
- **Connect LLMs to external data** (RAG - Retrieval Augmented Generation)
- **Build autonomous agents** that can use tools and make decisions
- **Create complex workflows** by chaining together components
- **Maintain conversation state** across interactions
- **Switch between providers** with minimal code changes

### Why LangChain in 2025?

- **Provider-Agnostic**: Write once, switch between OpenAI, Anthropic, Google easily
- **Production-Ready**: Built-in patterns for RAG, agents, memory
- **Modern Architecture**: LangGraph for complex workflows, LCEL for composability
- **Rich Ecosystem**: 100+ integrations with vector stores, tools, APIs
- **Active Development**: Regular updates, strong community support

### When to Use LangChain

| ✅ Use LangChain For | ❌ Don't Use For |
|---------------------|------------------|
| RAG applications | Simple single LLM calls |
| Multi-step workflows | Maximum performance critical paths |
| Agents with tools | Static predefined responses |
| Provider flexibility | When you only use one provider |
| Complex state management | Simple scripts |
| Production applications | Quick prototypes |

---

## Core Concepts

### 1. Runnables: The Foundation

Everything in LangChain implements the **Runnable** interface:

```python
.invoke(input)      # Synchronous execution
.stream(input)      # Streaming results
.batch(inputs)      # Batch processing
.ainvoke(input)     # Async execution
```

### 2. LCEL (LangChain Expression Language)

Modern way to compose components using the pipe operator:

```python
chain = prompt | model | output_parser
```

Benefits:
- **Intuitive**: Reads like a data flow
- **Composable**: Mix and match components
- **Async Support**: Automatic async execution
- **Streaming**: Built-in streaming support
- **Parallel Execution**: RunnableParallel for concurrency

### 3. Core Components

#### Models
- **Chat Models**: Message-based interface (recommended)
- **LLMs**: Text in, text out (legacy)
- **Embeddings**: Text to vectors

#### Prompts
- **ChatPromptTemplate**: Message-based prompts
- **FewShotPromptTemplate**: Examples-based prompting
- **MessagesPlaceholder**: Dynamic message insertion

#### Output Parsers
- **StrOutputParser**: Extract string
- **JsonOutputParser**: Parse JSON
- **PydanticOutputParser**: Validate with Pydantic

#### Chains
- **Simple Chains**: Linear workflows
- **Retrieval Chains**: RAG pipelines
- **Custom Chains**: Complex logic

#### Agents
- **ReAct**: Reasoning + Acting pattern
- **OpenAI Functions**: Structured outputs
- **Custom Agents**: Your own logic

#### Memory
- **Buffer**: Store all messages
- **Window**: Keep last N messages
- **Summary**: Summarize old messages
- **SQL/Redis**: Persistent storage

#### Retrievers
- **Vector Store**: Semantic search
- **Multi-Query**: Generate variations
- **Parent Document**: Retrieve context

---

## Components

### Models & Providers

#### Chat Models (Recommended)

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatOpenAI(model="gpt-4", temperature=0.7)
```

**Supported Providers**:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude 3/4)
- Google (Gemini Pro)
- Cohere (Command R+)
- Mistral (Mistral Large)
- Local (Ollama, LM Studio)

#### Embeddings

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

**Embedding Models**:
- OpenAI: text-embedding-3-small/large
- Cohere: embed-english-v3.0
- HuggingFace: all-MiniLM-L6-v2 (free)

### Prompts

#### ChatPromptTemplate

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} assistant."),
    ("human", "{input}")
])
```

#### Few-Shot Prompting

```python
from langchain.prompts import FewShotChatMessagePromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"}
]

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_template
)
```

### Chains

#### Simple Chain

```python
chain = prompt | model | output_parser
result = chain.invoke({"input": "Hello"})
```

#### Parallel Chains

```python
from langchain_core.runnables import RunnableParallel

chain = RunnableParallel({
    "summary": summarize_chain,
    "sentiment": sentiment_chain
})
```

#### Conditional Chains

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: len(x) > 100, long_chain),
    (lambda x: len(x) > 50, medium_chain),
    short_chain
)
```

### RAG (Retrieval Augmented Generation)

#### Complete RAG Pipeline

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain

# 1. Load documents
loader = TextLoader("data.txt")
docs = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
splits = splitter.split_documents(docs)

# 3. Create vector store
vectorstore = Chroma.from_documents(splits, OpenAIEmbeddings())

# 4. Create retriever
retriever = vectorstore.as_retriever()

# 5. Build RAG chain
rag_chain = create_retrieval_chain(retriever, llm_chain)

# 6. Query
response = rag_chain.invoke({"input": "What is the topic?"})
```

#### Vector Stores

| Store | Best For | Persistence | Scalability | Cost |
|-------|----------|-------------|-------------|------|
| Chroma | Development | Local | Medium | Free |
| FAISS | Fast search | Optional | High | Free |
| Pinecone | Production | Cloud | Very High | Paid |
| Weaviate | Enterprise | Self-hosted | Very High | Free/Paid |
| Qdrant | Production | Self-hosted | High | Free/Paid |

### Agents

#### Simple Agent

```python
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))

tools = [calculator]
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

result = agent_executor.invoke({"input": "What's 25 * 17?"})
```

#### Agent Types (2025)

| Type | Use Case | Complexity | Recommended |
|------|----------|------------|-------------|
| OpenAI Functions | Structured outputs | Low | ✅ Yes |
| ReAct | General purpose | Medium | ✅ Yes |
| LangGraph | Complex workflows | High | ✅ Yes (production) |
| Legacy Agents | Old APIs | Low | ❌ No (deprecated) |

### Memory

#### Modern Approach (RunnableWithMessageHistory)

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

def get_session_history(session_id: str):
    return SQLChatMessageHistory(session_id, "sqlite:///memory.db")

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)
```

#### Memory Types

| Type | Storage | Best For | Token Usage |
|------|---------|----------|-------------|
| Buffer | All messages | Short conversations | High |
| Window | Last N messages | Long conversations | Medium |
| Summary | Summarized history | Very long conversations | Low |
| Entity | Tracked entities | Complex relationships | Variable |

---

## When to Use What

### Decision Tree

```
Need LLM functionality?
├── Simple single call → Use provider SDK directly
└── Complex workflow → Use LangChain
    ├── Need external data? → Use RAG
    │   ├── Small dataset → Chroma/FAISS
    │   └── Large dataset → Pinecone/Weaviate
    ├── Need tools/actions? → Use Agents
    │   ├── Simple tools → OpenAI Functions Agent
    │   └── Complex workflows → LangGraph
    ├── Need conversation history? → Use Memory
    │   ├── Short conversations → Buffer Memory
    │   └── Long conversations → Summary Memory
    └── Need custom logic? → Use LCEL
```

### Pattern Selection Guide

**Use RAG when**:
- Need to query private/proprietary data
- Data changes frequently (can't fine-tune)
- Need citations/sources
- Want explainable results

**Use Agents when**:
- Need dynamic tool selection
- Multi-step problem solving
- External API integration
- Autonomous task completion

**Use Simple Chains when**:
- Linear workflows
- Predictable steps
- No branching logic
- Fast execution needed

**Use LangGraph when**:
- Complex state management
- Cycles/loops needed
- Human-in-the-loop
- Multi-agent systems

---

## Common Patterns

### Pattern 1: RAG with Conversation History

```python
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Context-aware retriever
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given chat history, rephrase question for retrieval."),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_prompt
)

# QA chain
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using context: {context}"),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

qa_chain = create_retrieval_chain(history_aware_retriever, qa_prompt)
```

### Pattern 2: Multi-Query RAG

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

# Generates multiple query variations for better retrieval
docs = retriever.get_relevant_documents("What is LangChain?")
```

### Pattern 3: Agent with Memory

```python
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)
```

### Pattern 4: Parallel Processing

```python
from langchain_core.runnables import RunnableParallel

analysis_chain = RunnableParallel({
    "summary": summarize_chain,
    "sentiment": sentiment_chain,
    "keywords": keyword_chain,
    "category": category_chain
})

results = analysis_chain.invoke({"text": document})
```

### Pattern 5: Fallback Chain

```python
primary_chain = prompt | ChatOpenAI(model="gpt-4") | parser
fallback_chain = prompt | ChatAnthropic(model="claude-sonnet-4-5-20250929") | parser

chain_with_fallback = primary_chain.with_fallbacks([fallback_chain])
```

---

## Provider Comparison

### OpenAI

**Best For**: General purpose, function calling, proven reliability

**Strengths**:
- ✅ Best ecosystem support
- ✅ Function calling
- ✅ JSON mode
- ✅ Vision (GPT-4V)
- ✅ Fast inference
- ✅ Extensive documentation

**Weaknesses**:
- ❌ Context window smaller than Claude
- ❌ Higher cost than some alternatives
- ❌ Data privacy concerns

**Models**:
- GPT-4: Best quality
- GPT-3.5-turbo: Fast and cheap

### Anthropic Claude

**Best For**: Long context, reasoning, safety-critical applications

**Strengths**:
- ✅ 200K token context window
- ✅ Superior reasoning
- ✅ Strong safety guardrails
- ✅ Constitutional AI
- ✅ Better at refusing harmful requests

**Weaknesses**:
- ❌ Smaller ecosystem than OpenAI
- ❌ No function calling (uses tools differently)
- ❌ Slower adoption

**Models**:
- Claude 4 Sonnet: Best balance
- Claude 3.5 Opus: Highest quality

### Google Gemini

**Best For**: Multimodal, free tier, Google integration

**Strengths**:
- ✅ Multimodal (text, images, video)
- ✅ Free tier available
- ✅ Fast inference
- ✅ Google services integration
- ✅ Long context (1M tokens in Flash)

**Weaknesses**:
- ❌ Less proven than GPT/Claude
- ❌ Smaller community
- ❌ API stability concerns

**Models**:
- Gemini Pro: Free tier
- Gemini Ultra: Highest capability

### Cost Comparison (2025)

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| GPT-4 | $30 | $60 |
| GPT-3.5-turbo | $0.50 | $1.50 |
| Claude 4 Sonnet | $3 | $15 |
| Claude 3.5 Opus | $15 | $75 |
| Gemini Pro | Free (with limits) | Free (with limits) |

---

## Performance Considerations

### Token Optimization

**Strategies**:
1. **Use concise prompts**: Remove unnecessary words
2. **Summarize conversation history**: Don't send full history
3. **Use smaller models**: GPT-3.5 instead of GPT-4 when possible
4. **Cache responses**: Save repeated queries
5. **Batch requests**: Process multiple inputs together

### Latency Reduction

**Techniques**:
1. **Stream responses**: Better perceived performance
2. **Use async**: Non-blocking execution
3. **Parallel processing**: RunnableParallel
4. **Provider selection**: Some providers are faster
5. **Local embeddings**: Faster than API calls

### Scaling Patterns

**Development → Production**:

| Component | Development | Production |
|-----------|------------|------------|
| Vector Store | Chroma (local) | Pinecone/Weaviate |
| Memory | In-memory | Redis/PostgreSQL |
| Caching | In-memory | Redis |
| LLM Calls | Synchronous | Async with rate limiting |
| Error Handling | Basic | Retry with exponential backoff |

### Monitoring

**Key Metrics**:
- Token usage and costs
- Response latency (p50, p95, p99)
- Error rates
- Cache hit rates
- Retrieval accuracy

**Tools**:
- **LangSmith**: Official monitoring/debugging
- **Weights & Biases**: Experiment tracking
- **Custom logging**: CloudWatch, Datadog, etc.

---

## Security Best Practices

### 1. API Key Management

❌ **Don't**:
```python
llm = ChatOpenAI(api_key="sk-...")  # Hardcoded
```

✅ **Do**:
```python
import os
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

### 2. Input Validation

❌ **Don't**:
```python
# Direct user input to LLM
result = chain.invoke({"input": user_input})
```

✅ **Do**:
```python
# Validate and sanitize
if len(user_input) > 10000:
    raise ValueError("Input too long")

# Check for prompt injection
if "ignore previous instructions" in user_input.lower():
    raise ValueError("Suspicious input")

result = chain.invoke({"input": user_input})
```

### 3. Agent Sandboxing

❌ **Don't**:
```python
@tool
def execute_code(code: str) -> str:
    return exec(code)  # Dangerous!
```

✅ **Do**:
```python
@tool
def calculator(expression: str) -> str:
    # Whitelist allowed operations
    allowed = {'+', '-', '*', '/', '(', ')', ' ', '.', '0-9'}
    if not all(c in allowed for c in expression):
        raise ValueError("Invalid expression")
    return str(eval(expression))
```

### 4. Output Filtering

```python
def filter_sensitive_data(text: str) -> str:
    """Remove sensitive information from outputs."""
    import re
    # Remove emails
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    # Remove phone numbers
    text = re.sub(r'\d{3}-\d{3}-\d{4}', '[PHONE]', text)
    # Remove API keys
    text = re.sub(r'sk-[A-Za-z0-9]{48}', '[API_KEY]', text)
    return text
```

### 5. Rate Limiting

```python
from functools import wraps
from time import time, sleep

def rate_limit(max_per_minute):
    """Decorator to rate limit function calls."""
    calls = []

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time()
            calls[:] = [c for c in calls if now - c < 60]

            if len(calls) >= max_per_minute:
                sleep_time = 60 - (now - calls[0])
                sleep(sleep_time)

            calls.append(time())
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(max_per_minute=10)
def query_llm(prompt):
    return llm.invoke(prompt)
```

---

## Production Checklist

### Before Deployment

- [ ] **Environment variables** set up correctly
- [ ] **Error handling** for all LLM calls
- [ ] **Rate limiting** implemented
- [ ] **Retry logic** with exponential backoff
- [ ] **Logging** configured (structured logs)
- [ ] **Monitoring** set up (metrics, alerts)
- [ ] **Cost tracking** enabled
- [ ] **Input validation** implemented
- [ ] **Output filtering** for sensitive data
- [ ] **Timeout settings** configured
- [ ] **Caching** strategy decided
- [ ] **Vector store** scaled for production
- [ ] **Memory storage** persistent
- [ ] **API key rotation** process
- [ ] **Backup strategy** for data
- [ ] **Load testing** completed
- [ ] **Security audit** done
- [ ] **Documentation** updated

### Production Code Template

```python
import os
import logging
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionLLMService:
    """Production-ready LLM service with error handling and monitoring."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            max_retries=3,
            timeout=30,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.prompt = ChatPromptTemplate.from_template("{input}")
        self.chain = self.prompt | self.llm

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def query(self, user_input: str) -> Optional[str]:
        """Query LLM with retry logic and error handling."""
        try:
            # Validate input
            if len(user_input) > 10000:
                raise ValueError("Input too long")

            # Log request
            logger.info(f"Processing query: {user_input[:100]}...")

            # Execute
            response = self.chain.invoke({"input": user_input})

            # Log response
            logger.info(f"Response generated: {len(response.content)} chars")

            return response.content

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

# Usage
service = ProductionLLMService()
result = service.query("What is LangChain?")
```

---

## Common Pitfalls

### ❌ Pitfall 1: Not Handling Rate Limits

**Problem**:
```python
# Sending too many requests too fast
for i in range(1000):
    llm.invoke("Hello")  # Will hit rate limit
```

**Solution**:
```python
from time import sleep

for i in range(1000):
    try:
        llm.invoke("Hello")
        sleep(0.1)  # Rate limiting
    except Exception as e:
        logger.error(f"Request {i} failed: {e}")
        sleep(1)  # Backoff on error
```

### ❌ Pitfall 2: Ignoring Token Limits

**Problem**:
```python
# Sending huge context without checking
huge_text = "..." * 100000
llm.invoke(huge_text)  # Will fail
```

**Solution**:
```python
import tiktoken

def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

if count_tokens(text) > 8000:
    # Truncate or summarize
    text = text[:8000]
```

### ❌ Pitfall 3: Not Implementing Timeouts

**Problem**:
```python
# No timeout - can hang forever
llm.invoke(prompt)
```

**Solution**:
```python
llm = ChatOpenAI(
    model="gpt-4",
    timeout=30,  # 30 second timeout
    max_retries=2
)
```

### ❌ Pitfall 4: Forgetting to Stream Long Responses

**Problem**:
```python
# User waits for entire response
response = llm.invoke("Write a long essay...")
print(response.content)  # Nothing shown until complete
```

**Solution**:
```python
# Stream for better UX
for chunk in llm.stream("Write a long essay..."):
    print(chunk.content, end="", flush=True)
```

### ❌ Pitfall 5: Hardcoding Prompts

**Problem**:
```python
# Hard to modify and test
response = llm.invoke("You are a helpful assistant. " + user_input)
```

**Solution**:
```python
# Use templates
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} assistant."),
    ("human", "{input}")
])
chain = prompt | llm
```

### ❌ Pitfall 6: Not Validating Agent Tool Usage

**Problem**:
```python
@tool
def execute_command(cmd: str) -> str:
    return os.system(cmd)  # Dangerous!
```

**Solution**:
```python
@tool
def safe_calculator(expression: str) -> str:
    """Only allow mathematical operations."""
    # Whitelist approach
    allowed_chars = set("0123456789+-*/.()")
    if not all(c in allowed_chars for c in expression.replace(" ", "")):
        raise ValueError("Invalid expression")
    return str(eval(expression))
```

### ❌ Pitfall 7: Inefficient RAG Chunking

**Problem**:
```python
# Too large chunks - poor retrieval
splitter = RecursiveCharacterTextSplitter(chunk_size=5000)
```

**Solution**:
```python
# Optimal chunk size with overlap
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Good for most cases
    chunk_overlap=200,  # Preserve context
    separators=["\n\n", "\n", ".", " ", ""]
)
```

### ❌ Pitfall 8: Not Monitoring Costs

**Problem**:
```python
# No cost tracking
for _ in range(1000):
    llm.invoke(long_prompt)
```

**Solution**:
```python
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    for _ in range(1000):
        llm.invoke(long_prompt)

    print(f"Total cost: ${cb.total_cost:.2f}")
    if cb.total_cost > 10:
        logger.warning("High cost detected!")
```

---

## Resources

### Official Documentation

- [LangChain Python Docs](https://python.langchain.com/)
- [LangChain JavaScript Docs](https://js.langchain.com/)
- [LangSmith (Debugging)](https://smith.langchain.com/)
- [LangChain Blog](https://blog.langchain.dev/)

### Learning Resources

- [LangChain Tutorials (Official)](https://python.langchain.com/docs/tutorials/)
- [LangChain Cookbook](https://github.com/langchain-ai/langchain/tree/master/cookbook)
- [LangChain Templates](https://python.langchain.com/docs/templates/)

### Community

- [GitHub Discussions](https://github.com/langchain-ai/langchain/discussions)
- [Discord Community](https://discord.gg/langchain)
- [Twitter](https://twitter.com/langchainai)

### Provider Documentation

- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Anthropic Claude Docs](https://docs.anthropic.com/)
- [Google Gemini Docs](https://ai.google.dev/docs)

### Tools & Integrations

- [LangSmith](https://smith.langchain.com/) - Debugging and monitoring
- [LangServe](https://python.langchain.com/docs/langserve) - Deploy as REST API
- [Vector Store Integrations](https://python.langchain.com/docs/integrations/vectorstores/)

---

## Learning Path

Recommended order for the notebooks:

1. **langchain_basics.ipynb** - Core concepts and first examples
2. **langchain_models.ipynb** - Models and providers
3. **langchain_prompts.ipynb** - Prompt engineering
4. **langchain_lcel.ipynb** - Expression Language
5. **langchain_rag.ipynb** - RAG pipelines
6. **langchain_agents.ipynb** - Agents and tools
7. **langchain_memory.ipynb** - Conversation memory
8. **langchain_integration.ipynb** - Full application

Estimated time: 3-4 weeks (2-3 hours per module)

---

**Last Updated**: 2025-01-08
**LangChain Version**: 0.3.x
