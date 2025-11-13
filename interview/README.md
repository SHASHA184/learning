# Interview Preparation - Problem Sets

This directory contains structured problem-solution examples for technical interviews, organized by topic with detailed explanations and complexity analysis.

## Quick Reference

| Notebook | Topics Covered | Problem Count | Difficulty Range |
|----------|---------------|---------------|------------------|
| [problems_data_structures.ipynb](problems_data_structures.ipynb) | Arrays, Strings, Linked Lists, Trees, Graphs, Hash Tables | 15 | Easy - Hard |
| [problems_algorithms.ipynb](problems_algorithms.ipynb) | Sorting, Searching, DP, Recursion, Backtracking | 15 | Easy - Hard |
| [problems_python_specific.ipynb](problems_python_specific.ipynb) | Decorators, Generators, Context Managers, Metaclasses | 10 | Medium - Hard |
| [problems_system_design.ipynb](problems_system_design.ipynb) | OOP Design, Design Patterns, Scalability | 8 | Medium - Hard |

## Python Fundamentals & Advanced Concepts

| Notebook | Focus Area | Level |
|----------|-----------|-------|
| [python_basics.ipynb](python_basics.ipynb) | Python fundamentals: data types, functions, scope, built-ins, common gotchas | Beginner-Intermediate |
| [oop_principles.ipynb](oop_principles.ipynb) | The four pillars of OOP (Encapsulation, Inheritance, Polymorphism, Abstraction) | Intermediate |
| [solid_principles.ipynb](solid_principles.ipynb) | SOLID design principles with practical Python examples and refactoring patterns | Intermediate-Advanced |
| [generators.ipynb](generators.ipynb) | Generator functions and expressions | Intermediate |
| [iterators.ipynb](iterators.ipynb) | Iterator protocol and custom iterators | Intermediate |
| [descriptor.ipynb](descriptor.ipynb) | Descriptors and attribute access control | Advanced |
| [mro.ipynb](mro.ipynb) | Method Resolution Order in multiple inheritance | Advanced |

## Problem Index by Difficulty

### Easy (Good for Warm-up)
- Two Sum (DS)
- Valid Parentheses (DS)
- Palindrome Check (DS)
- Reverse Linked List (DS)
- Binary Search (Algo)
- Bubble Sort (Algo)
- FizzBuzz (Algo)

### Medium (Core Interview Level)
- Group Anagrams (DS)
- Merge Intervals (DS)
- Binary Tree Level Order Traversal (DS)
- LRU Cache (DS)
- Longest Substring Without Repeating Characters (DS)
- Coin Change (Algo)
- Word Break (Algo)
- Permutations (Algo)
- Custom Retry Decorator (Python)
- Context Manager for Database Connection (Python)
- Singleton Design Pattern (Design)
- Factory Pattern Implementation (Design)

### Hard (Advanced/Senior Level)
- Median of Two Sorted Arrays (DS)
- Serialize/Deserialize Binary Tree (DS)
- Word Ladder (Algo)
- N-Queens Problem (Algo)
- Metaclass for Auto-validation (Python)
- Async Rate Limiter (Python)
- Design a Distributed Cache (Design)
- Design URL Shortener (Design)

## Problem Format

Each problem follows this structure:

```python
# Problem Statement
"""
Clear description of the problem with examples and constraints
"""

# Examples
"""
Input: [example input]
Output: [example output]
Explanation: [why this is the answer]
"""

# Solution 1: Brute Force
def solution_brute_force(input):
    """
    Time Complexity: O(?)
    Space Complexity: O(?)
    """
    pass

# Solution 2: Optimized
def solution_optimized(input):
    """
    Time Complexity: O(?)
    Space Complexity: O(?)
    Approach: [brief explanation]
    """
    pass

# Test Cases
assert solution_optimized([test_input]) == expected_output

# Common Pitfalls
"""
- Edge case 1
- Edge case 2
"""

# Follow-up Questions
"""
- What if constraint X changes?
- How would you handle Y?
"""
```

## Interview Preparation Tips

### 1. **Problem-Solving Framework**
- Clarify the problem and constraints
- Walk through examples
- Discuss approaches (brute force → optimized)
- Code the solution
- Test with edge cases
- Analyze complexity

### 2. **Time Management**
- 5 min: Understanding and clarification
- 10 min: Solution design and discussion
- 20 min: Implementation
- 5 min: Testing and refinement

### 3. **Communication**
- Think out loud
- Explain your reasoning
- Discuss trade-offs
- Ask clarifying questions

### 4. **Common Patterns**
- Two Pointers
- Sliding Window
- Fast & Slow Pointers
- Binary Search variants
- DFS/BFS for trees and graphs
- Dynamic Programming states
- Hash maps for O(1) lookups

## Complexity Reference

### Time Complexity
- **O(1)** - Constant: Hash table lookup, array access
- **O(log n)** - Logarithmic: Binary search, balanced tree operations
- **O(n)** - Linear: Single pass through data
- **O(n log n)** - Linearithmic: Efficient sorting (merge sort, heap sort)
- **O(n²)** - Quadratic: Nested loops, bubble sort
- **O(2ⁿ)** - Exponential: Recursive fibonacci, subset generation
- **O(n!)** - Factorial: Permutations

### Space Complexity
- **O(1)** - Constant: Fixed variables
- **O(n)** - Linear: Additional array/hash map
- **O(log n)** - Logarithmic: Recursion stack for binary search
- **O(n²)** - Quadratic: 2D matrix

## Study Plan

### Week 1-2: Fundamentals
- Focus on Easy problems from Data Structures
- Master array and string manipulation
- Practice hash table usage

### Week 3-4: Core Algorithms
- Medium problems from Algorithms
- Dynamic Programming patterns
- Tree and graph traversals

### Week 5-6: Python Mastery
- Python-specific problems
- Advanced language features
- Performance optimization

### Week 7-8: System Design
- Design pattern implementation
- OOP principles
- Scalability considerations

## Resources

- [LeetCode](https://leetcode.com) - Practice problems
- [Python Official Docs](https://docs.python.org/3/) - Language reference
- [Big-O Cheat Sheet](https://www.bigocheatsheet.com/) - Complexity reference
- [Cracking the Coding Interview](http://www.crackingthecodinginterview.com/) - Interview prep book

## Contributing

When adding new problems:
1. Follow the established format
2. Include multiple solution approaches
3. Add comprehensive test cases
4. Explain complexity analysis
5. Update this README with the new problem
