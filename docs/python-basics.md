# Python Basics - Quick Interview Reference

Fast reference guide for fundamental Python concepts commonly tested in technical interviews.

## Table of Contents

- [Data Types & Structures](#data-types--structures)
- [Functions & Scope](#functions--scope)
- [Common Gotchas](#common-gotchas)
- [Built-in Functions](#built-in-functions)
- [OOP Basics](#oop-basics)
- [Error Handling](#error-handling)
- [Python Conventions](#python-conventions)
- [Interview Cheat Sheet](#interview-cheat-sheet)

---

## Data Types & Structures

### Basic Data Types

| Type | Example | Mutable? | Notes |
|------|---------|----------|-------|
| `int` | `42` | No | Unlimited precision |
| `float` | `3.14` | No | IEEE 754 double precision |
| `str` | `"hello"` | No | Unicode sequences |
| `bool` | `True`/`False` | No | Subclass of `int` |
| `None` | `None` | No | Singleton null value |
| `list` | `[1, 2, 3]` | Yes | Ordered sequence |
| `tuple` | `(1, 2, 3)` | No | Immutable list |
| `dict` | `{"a": 1}` | Yes | Key-value mapping |
| `set` | `{1, 2, 3}` | Yes | Unordered, unique elements |

### When to Use Which Collection?

**Use `list`** when:
- Need ordered sequence
- Need to modify (append, remove)
- Duplicates allowed

**Use `tuple`** when:
- Need immutable sequence
- Want to use as dict key
- Returning multiple values from function

**Use `set`** when:
- Need unique elements only
- Fast membership testing (`in`)
- Set operations (union, intersection)

**Use `dict`** when:
- Need key-value mapping
- O(1) lookups by key
- Counting, grouping, caching

### List/Dict/Set Comprehensions

```python
# List comprehension
squares = [x**2 for x in range(10)]
evens = [x for x in range(10) if x % 2 == 0]

# Dict comprehension
squared_dict = {x: x**2 for x in range(5)}

# Set comprehension
unique_lengths = {len(word) for word in words}

# Nested comprehension
matrix = [[i*j for j in range(3)] for i in range(3)]
```

### Slicing

**Syntax**: `sequence[start:stop:step]`

```python
nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

nums[2:5]    # [2, 3, 4] - elements 2 to 4
nums[:5]     # [0, 1, 2, 3, 4] - first 5
nums[5:]     # [5, 6, 7, 8, 9] - from 5 to end
nums[::2]    # [0, 2, 4, 6, 8] - every 2nd
nums[::-1]   # [9, 8, 7, ..., 0] - reverse
nums[-3:]    # [7, 8, 9] - last 3
```

---

## Functions & Scope

### `*args` and `**kwargs`

```python
def func(required, *args, **kwargs):
    # required: mandatory positional
    # args: extra positional (tuple)
    # kwargs: extra keyword (dict)
    pass

func(1, 2, 3, key='value')
# required=1, args=(2, 3), kwargs={'key': 'value'}

# Unpacking
numbers = [1, 2, 3]
sum_all(*numbers)  # Unpack list

options = {'key': 'value'}
process(**options)  # Unpack dict
```

### LEGB Scope Rule

Python searches for variables in this order:
1. **L**ocal - current function
2. **E**nclosing - outer functions (closures)
3. **G**lobal - module level
4. **B**uilt-in - Python built-ins

```python
x = "global"

def outer():
    x = "enclosing"

    def inner():
        x = "local"  # Shadows outer scopes
        print(x)     # Prints "local"

    inner()
    print(x)  # Prints "enclosing"

outer()
print(x)  # Prints "global"
```

**Modifying outer scope:**
```python
# global - modify global variable
count = 0
def increment():
    global count
    count += 1

# nonlocal - modify enclosing variable
def outer():
    count = 0
    def inner():
        nonlocal count
        count += 1
    inner()
```

### Lambda Functions

**Syntax**: `lambda arguments: expression`

```python
# Single expression anonymous function
square = lambda x: x**2
add = lambda x, y: x + y

# Common with map, filter, sorted
squared = map(lambda x: x**2, numbers)
evens = filter(lambda x: x % 2 == 0, numbers)
sorted(words, key=lambda w: len(w))
```

**When to use**: Short, simple functions used once (especially as arguments)

**When NOT to use**: Complex logic (use regular function instead)

---

## Common Gotchas

### 1. Mutable Default Arguments

```python
# ❌ WRONG
def add_item(item, items=[]):
    items.append(item)  # Same list reused!
    return items

# ✅ CORRECT
def add_item(item, items=None):
    if items is None:
        items = []  # New list each time
    items.append(item)
    return items
```

**Why**: Default arguments evaluated once at function definition, not each call.

### 2. Shallow vs Deep Copy

```python
import copy

original = [[1, 2], [3, 4]]

# Assignment - same object
assigned = original

# Shallow copy - new outer, shared nested
shallow = original.copy()
shallow[0] is original[0]  # True - SHARED!

# Deep copy - all independent
deep = copy.deepcopy(original)
deep[0] is original[0]  # False - independent
```

### 3. Late Binding Closures

```python
# ❌ WRONG
funcs = [lambda: i for i in range(5)]
[f() for f in funcs]  # [4, 4, 4, 4, 4] - all 4!

# ✅ CORRECT - capture with default arg
funcs = [lambda x=i: x for i in range(5)]
[f() for f in funcs]  # [0, 1, 2, 3, 4]
```

### 4. String/Integer Caching

```python
# Strings that look like identifiers are interned
a = "hello"
b = "hello"
a is b  # True - same object

# Small integers (-5 to 256) are cached
x = 100
y = 100
x is y  # True - same object

# Large integers are NOT cached
x = 1000
y = 1000
x is y  # Usually False

# LESSON: Always use == for comparison, not is
```

---

## Built-in Functions

### Essential Functions to Know

**Iteration:**
```python
enumerate(iterable, start=0)  # (index, value) pairs
zip(*iterables)               # Parallel iteration
range(start, stop, step)      # Number sequence
reversed(sequence)            # Reverse iterator
```

**Aggregation:**
```python
sum(iterable, start=0)     # Sum all elements
min(iterable, key=None)    # Minimum value
max(iterable, key=None)    # Maximum value
any(iterable)              # True if any truthy
all(iterable)              # True if all truthy
```

**Transformation:**
```python
map(function, iterable)          # Apply function
filter(function, iterable)       # Keep if True
sorted(iterable, key, reverse)   # Return sorted list
list.sort(key, reverse)          # Sort in-place
```

### `sort()` vs `sorted()`

| Feature | `list.sort()` | `sorted()` |
|---------|--------------|------------|
| **Type** | Method | Function |
| **Modifies** | In-place | Returns new |
| **Return** | `None` | Sorted list |
| **Works on** | Lists only | Any iterable |

```python
# sort() - modifies in-place
numbers.sort()  # Returns None, modifies numbers

# sorted() - returns new list
result = sorted(numbers)  # numbers unchanged

# Both support key and reverse
sorted(words, key=len, reverse=True)
```

### `enumerate()` and `zip()`

```python
# enumerate - get index with value
for i, value in enumerate(items, start=1):
    print(f"{i}. {value}")

# zip - parallel iteration
names = ['Alice', 'Bob']
ages = [25, 30]
for name, age in zip(names, ages):
    print(f"{name} is {age}")

# Unzip with zip(*)
pairs = [(1, 'a'), (2, 'b')]
nums, letters = zip(*pairs)
```

### `any()` and `all()`

```python
any([False, False, True])   # True - at least one
all([True, True, True])     # True - all
any([])                     # False - empty
all([])                     # True - vacuous truth

# With comprehensions
has_even = any(x % 2 == 0 for x in numbers)
all_positive = all(x > 0 for x in numbers)
```

---

## OOP Basics

### `__init__` vs `__new__`

```python
class Example:
    def __new__(cls, *args, **kwargs):
        # Creates instance (rarely overridden)
        instance = super().__new__(cls)
        return instance

    def __init__(self, value):
        # Initializes instance (commonly overridden)
        self.value = value
```

**`__new__`**: Creates the object, returns instance
**`__init__`**: Initializes the object, returns None

**Use `__new__` for**: Singletons, subclassing immutable types

### Class vs Instance Attributes

```python
class Dog:
    species = "Canis familiaris"  # Class attribute (shared)

    def __init__(self, name):
        self.name = name  # Instance attribute (unique)

dog1 = Dog("Buddy")
dog2 = Dog("Max")

dog1.species  # Same for all
dog1.name     # Different for each
```

### `@staticmethod` vs `@classmethod`

```python
class Date:
    def __init__(self, year, month, day):
        self.year = year

    # Regular method - needs instance
    def format(self):
        return f"{self.year}-{self.month}-{self.day}"

    # Class method - gets class, can create instances
    @classmethod
    def from_string(cls, date_str):
        y, m, d = date_str.split('-')
        return cls(int(y), int(m), int(d))

    # Static method - just a function
    @staticmethod
    def is_valid(year, month, day):
        return 1 <= month <= 12
```

### `__str__` vs `__repr__`

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        # User-friendly, for print()
        return f"({self.x}, {self.y})"

    def __repr__(self):
        # Developer-friendly, for repr(), REPL
        # Ideal: eval(repr(obj)) == obj
        return f"Point({self.x}, {self.y})"
```

**Rule**: Always implement `__repr__`, optionally `__str__`

---

## Error Handling

### try/except/else/finally

```python
try:
    # Code that might raise exception
    result = risky_operation()
except SpecificError as e:
    # Handle specific exception
    handle_error(e)
except (TypeError, ValueError):
    # Handle multiple types
    pass
except Exception as e:
    # Catch all (use sparingly)
    log_error(e)
else:
    # Runs if NO exception
    process_success(result)
finally:
    # ALWAYS runs (cleanup)
    cleanup()
```

### Common Exceptions

**Type/Value:**
- `TypeError` - Wrong type
- `ValueError` - Right type, wrong value
- `AttributeError` - No such attribute
- `KeyError` - Dict key missing
- `IndexError` - Index out of range

**I/O:**
- `FileNotFoundError` - File doesn't exist
- `PermissionError` - No permission
- `IOError` - I/O operation failed

**Other:**
- `ZeroDivisionError` - Division by zero
- `ImportError` - Import failed
- `NameError` - Undefined variable
- `StopIteration` - Iterator exhausted

### Custom Exceptions

```python
class ValidationError(ValueError):
    """Custom exception for validation"""
    pass

def validate(value):
    if value < 0:
        raise ValidationError("Must be positive")
```

---

## Python Conventions

### PEP 8 Highlights

**Naming:**
```python
variable_name       # snake_case
function_name       # snake_case
CONSTANT_NAME       # UPPER_CASE
ClassName           # PascalCase
_private            # Leading underscore
__very_private      # Double underscore
```

**Spacing:**
- 4 spaces per indentation level (NOT tabs)
- 79 characters max line length
- 2 blank lines before class/function
- 1 blank line between methods
- Spaces around operators: `x = 1` not `x=1`

**Imports:**
- At top of file
- Standard library → third-party → local
- One import per line

### `if __name__ == "__main__":`

```python
def main_function():
    # Reusable code
    pass

if __name__ == "__main__":
    # Only runs when executed directly
    # Not when imported as module
    main_function()
```

**Use case**: Test code, CLI interfaces, demos

### EAFP vs LBYL

**LBYL (Look Before You Leap)** - Not Pythonic:
```python
if key in dictionary:
    value = dictionary[key]
```

**EAFP (Easier to Ask Forgiveness than Permission)** - Pythonic:
```python
try:
    value = dictionary[key]
except KeyError:
    value = None
```

**Even better** - use built-in:
```python
value = dictionary.get(key, default)
```

---

## Interview Cheat Sheet

### Common String Methods

```python
# Cleaning
s.strip()           # Remove whitespace
s.lstrip(), s.rstrip()

# Splitting/Joining
s.split(delimiter)  # Split into list
delimiter.join(list)  # Join list into string

# Searching
s.find(substring)   # Index or -1
s.index(substring)  # Index or ValueError
s.startswith(prefix)
s.endswith(suffix)
s.count(substring)

# Replacing
s.replace(old, new, count)

# Case
s.lower(), s.upper()
s.title(), s.capitalize()

# Checking
s.isdigit(), s.isalpha()
s.isalnum(), s.isspace()
```

### Common List Operations

```python
# Adding
lst.append(item)        # Add to end - O(1)
lst.insert(i, item)     # Add at index - O(n)
lst.extend(items)       # Add multiple - O(k)

# Removing
lst.pop()              # Remove last - O(1)
lst.pop(i)             # Remove at index - O(n)
lst.remove(value)      # Remove first occurrence - O(n)
lst.clear()            # Remove all

# Other
lst.reverse()          # Reverse in-place
lst.sort()             # Sort in-place
lst.count(value)       # Count occurrences
lst.index(value)       # Find index

# Slicing creates copy
new_list = lst[:]      # Shallow copy
```

### Common Dict Operations

```python
# Accessing
d[key]                 # KeyError if missing
d.get(key, default)    # Return default if missing
d.setdefault(key, default)  # Set if missing

# Adding/Updating
d[key] = value
d.update(other_dict)
d.update(key=value)

# Removing
d.pop(key, default)
d.popitem()           # Remove and return (key, value)
del d[key]
d.clear()

# Iteration
for key in d:
for key, value in d.items():
for key in d.keys():
for value in d.values():
```

### Collections Module Essentials

```python
from collections import defaultdict, Counter, deque

# defaultdict - dict with default factory
d = defaultdict(list)
d['key'].append('value')  # No KeyError

d = defaultdict(int)
d['count'] += 1  # Starts at 0

# Counter - count hashable objects
counts = Counter(['a', 'b', 'a', 'c', 'a'])
counts['a']  # 3
counts.most_common(2)  # [('a', 3), ('b', 1)]

# deque - double-ended queue
dq = deque([1, 2, 3])
dq.append(4)      # Add right - O(1)
dq.appendleft(0)  # Add left - O(1)
dq.pop()          # Remove right - O(1)
dq.popleft()      # Remove left - O(1)
```

### Time Complexity Reference

**List Operations:**
| Operation | Complexity |
|-----------|-----------|
| `lst[i]` | O(1) |
| `lst.append(x)` | O(1) |
| `lst.insert(i, x)` | O(n) |
| `lst.pop()` | O(1) |
| `lst.pop(i)` | O(n) |
| `x in lst` | O(n) |
| `lst.sort()` | O(n log n) |

**Dict Operations:**
| Operation | Average | Worst |
|-----------|---------|-------|
| `d[key]` | O(1) | O(n) |
| `d[key] = value` | O(1) | O(n) |
| `del d[key]` | O(1) | O(n) |
| `key in d` | O(1) | O(n) |

**Set Operations:**
| Operation | Average |
|-----------|---------|
| `x in s` | O(1) |
| `s.add(x)` | O(1) |
| `s.remove(x)` | O(1) |
| `s1 & s2` (intersection) | O(min(len(s1), len(s2))) |
| `s1 | s2` (union) | O(len(s1) + len(s2)) |

### Quick Interview Tips

**Before coding:**
1. Clarify requirements and constraints
2. Ask about edge cases
3. Discuss approach and time/space complexity
4. Get confirmation before coding

**While coding:**
1. Use clear variable names
2. Handle edge cases
3. Add brief comments for complex logic
4. Test with simple example

**After coding:**
1. Walk through with example
2. Discuss time/space complexity
3. Mention potential optimizations
4. Ask if they want you to test edge cases

**Common edge cases:**
- Empty input (`[]`, `""`, `None`)
- Single element
- Duplicates
- Very large/small numbers
- Negative numbers

### Pythonic Patterns

```python
# Swapping variables
a, b = b, a

# Multiple assignment
x, y, z = 1, 2, 3

# Conditional expression (ternary)
result = value_if_true if condition else value_if_false

# Membership testing
if x in [1, 2, 3]:

# Boolean operations
if any(condition for item in items):
if all(condition for item in items):

# Default dict value
value = d.get(key, default)

# Multiple comparisons
if 0 < x < 10:

# Enumerate for index + value
for i, value in enumerate(items):

# Zip for parallel iteration
for x, y in zip(list1, list2):
```

---

## Resources

**Documentation:**
- [Python Official Docs](https://docs.python.org/3/)
- [PEP 8 - Style Guide](https://pep8.org/)
- [Python Standard Library](https://docs.python.org/3/library/)

**Practice:**
- [LeetCode - Easy Problems](https://leetcode.com/problemset/all/?difficulty=EASY)
- [HackerRank - Python](https://www.hackerrank.com/domains/python)
- [Real Python Tutorials](https://realpython.com/)

**Books:**
- "Python Crash Course" by Eric Matthes
- "Fluent Python" by Luciano Ramalho
- "Effective Python" by Brett Slatkin

---

## Next Steps

1. Practice the notebook: `interview/python_basics.ipynb`
2. Review common patterns daily
3. Solve easy problems on LeetCode/HackerRank
4. Focus on time/space complexity analysis
5. Practice explaining your thought process out loud

Good luck with your interview preparation! 🚀
