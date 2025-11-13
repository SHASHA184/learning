# Object-Oriented Programming (OOP) - The Four Pillars

Comprehensive guide to the four fundamental principles of OOP with Python implementations, best practices, and real-world applications.

## Table of Contents

- [Overview](#overview)
- [The Four Pillars](#the-four-pillars)
  - [1. Encapsulation](#1-encapsulation)
  - [2. Inheritance](#2-inheritance)
  - [3. Polymorphism](#3-polymorphism)
  - [4. Abstraction](#4-abstraction)
- [Relationships Between Principles](#relationships-between-principles)
- [SOLID Principles Connection](#solid-principles-connection)
- [Common Pitfalls](#common-pitfalls)
- [Interview Preparation](#interview-preparation)
- [Quick Reference](#quick-reference)

---

## Overview

Object-Oriented Programming is a programming paradigm based on the concept of "objects" which contain data (attributes) and code (methods). The four pillars are fundamental principles that guide OOP design.

### Why OOP?

- **Modularity**: Code is organized into discrete objects
- **Reusability**: Code can be reused through inheritance and composition
- **Maintainability**: Encapsulation makes code easier to maintain
- **Scalability**: Well-designed OOP systems scale better
- **Real-world modeling**: Objects model real-world entities naturally

### Python's Approach to OOP

Python is a multi-paradigm language with strong OOP support:
- **Duck typing** over static typing
- **Properties** for Pythonic encapsulation
- **Multiple inheritance** with Method Resolution Order (MRO)
- **Magic methods** for operator overloading
- **ABC module** for abstract base classes

---

## The Four Pillars

### 1. Encapsulation

**Definition**: Bundling data (attributes) and methods that operate on that data within a single unit (class), while restricting direct access to some of the object's components.

#### Purpose
- **Data Hiding**: Protect object state from unauthorized access
- **Data Integrity**: Enforce validation rules
- **Flexibility**: Change internal implementation without affecting users
- **Reduce Coupling**: Hide implementation details

#### Python Implementation

```python
class BankAccount:
    def __init__(self, initial_balance):
        self._balance = initial_balance  # Protected attribute
        self.__account_id = self._generate_id()  # Private attribute

    @property
    def balance(self):
        """Getter - read-only access"""
        return self._balance

    def deposit(self, amount):
        """Controlled modification with validation"""
        if amount <= 0:
            raise ValueError("Amount must be positive")
        self._balance += amount

    def withdraw(self, amount):
        """Controlled modification with validation"""
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        self._balance -= amount
```

#### Access Modifiers in Python

| Convention | Meaning | Access |
|------------|---------|--------|
| `public` | Normal attribute | Accessible everywhere |
| `_protected` | Protected (convention) | Accessible but "internal use" |
| `__private` | Private (name mangling) | `_ClassName__private` |

**Note**: Python doesn't enforce true privacy - conventions rely on developer discipline.

#### Best Practices

✅ **DO:**
- Use `@property` for computed attributes
- Validate data in setters
- Use protected (`_`) for internal implementation
- Document public API clearly

❌ **DON'T:**
- Overuse private (`__`) attributes
- Create unnecessary getters/setters
- Violate encapsulation by accessing private attributes

#### Real-World Examples
- **Configuration objects**: Hide internal structure
- **Data models**: Validate and sanitize inputs
- **API clients**: Hide authentication details
- **Database connections**: Manage connection state

---

### 2. Inheritance

**Definition**: A mechanism where a new class (child/subclass/derived class) is created from an existing class (parent/superclass/base class), inheriting its attributes and methods.

#### Purpose
- **Code Reuse**: Avoid duplicating code
- **Hierarchical Classification**: Model "is-a" relationships
- **Extensibility**: Add or override functionality
- **Polymorphism**: Enable polymorphic behavior

#### Types of Inheritance

**Single Inheritance**
```python
class Animal:
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof!"
```

**Multiple Inheritance**
```python
class Flyable:
    def fly(self):
        return "Flying"

class Swimmable:
    def swim(self):
        return "Swimming"

class Duck(Animal, Flyable, Swimmable):
    pass
```

**Multilevel Inheritance**
```python
class Vehicle:
    pass

class Car(Vehicle):
    pass

class ElectricCar(Car):
    pass
```

#### Method Resolution Order (MRO)

Python uses **C3 Linearization** to determine method lookup order:

```python
class A: pass
class B(A): pass
class C(A): pass
class D(B, C): pass

print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)
```

**Rules:**
- Child classes before parents
- Parents in the order they appear in the definition
- Each class appears only once

#### Using `super()`

```python
class Parent:
    def __init__(self, name):
        self.name = name

class Child(Parent):
    def __init__(self, name, age):
        super().__init__(name)  # Call parent constructor
        self.age = age
```

**Benefits of `super()`:**
- Follows MRO correctly
- Works with multiple inheritance
- More maintainable than `Parent.__init__(self)`

#### Best Practices

✅ **DO:**
- Use inheritance for "is-a" relationships
- Keep hierarchies shallow (2-3 levels max)
- Use `super()` for parent class calls
- Consider composition over inheritance

❌ **DON'T:**
- Create deep inheritance hierarchies
- Use inheritance for code reuse alone (use composition)
- Violate Liskov Substitution Principle

#### Composition vs Inheritance

| Inheritance | Composition |
|-------------|-------------|
| "is-a" relationship | "has-a" relationship |
| Tight coupling | Loose coupling |
| Static relationship | Dynamic relationship |
| Less flexible | More flexible |

**When to prefer composition:**
```python
# Instead of inheritance
class Car(Engine, Wheels, Radio):  # ❌ Doesn't make sense
    pass

# Use composition
class Car:  # ✅ Car "has" these components
    def __init__(self):
        self.engine = Engine()
        self.wheels = Wheels()
        self.radio = Radio()
```

---

### 3. Polymorphism

**Definition**: The ability of different objects to respond to the same message (method call) in different ways. "Many forms" - same interface, different implementations.

#### Purpose
- **Flexibility**: Write code that works with multiple types
- **Extensibility**: Add new types without changing existing code
- **Simplicity**: Same interface for different behaviors
- **Abstraction**: Work with objects through common interface

#### Types in Python

**1. Method Overriding (Runtime Polymorphism)**
```python
class Shape:
    def area(self):
        raise NotImplementedError

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14159 * self.radius ** 2

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

# Polymorphic function
def print_area(shape: Shape):
    print(f"Area: {shape.area()}")  # Works with any Shape
```

**2. Duck Typing**
```python
# No inheritance needed!
class Duck:
    def quack(self):
        return "Quack!"

class Person:
    def quack(self):
        return "I'm imitating a duck!"

def make_it_quack(thing):
    # If it has quack(), we can use it
    return thing.quack()

make_it_quack(Duck())    # Works
make_it_quack(Person())  # Also works
```

**3. Operator Overloading**
```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        """Define v1 + v2"""
        return Vector(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar):
        """Define v * scalar"""
        return Vector(self.x * scalar, self.y * scalar)

    def __eq__(self, other):
        """Define v1 == v2"""
        return self.x == other.x and self.y == other.y

v1 = Vector(1, 2)
v2 = Vector(3, 4)
v3 = v1 + v2  # Uses __add__
v4 = v1 * 3   # Uses __mul__
```

#### Common Magic Methods

| Category | Methods | Purpose |
|----------|---------|---------|
| **String Representation** | `__str__`, `__repr__` | String conversion |
| **Arithmetic** | `__add__`, `__sub__`, `__mul__`, `__truediv__` | Operators |
| **Comparison** | `__eq__`, `__lt__`, `__le__`, `__gt__`, `__ge__` | Comparisons |
| **Container** | `__len__`, `__getitem__`, `__setitem__`, `__contains__` | Container protocol |
| **Callable** | `__call__` | Make object callable |
| **Context Manager** | `__enter__`, `__exit__` | `with` statement |
| **Attribute Access** | `__getattr__`, `__setattr__`, `__delattr__` | Attribute access |

#### Best Practices

✅ **DO:**
- Use duck typing for flexibility
- Implement magic methods consistently
- Define both `__eq__` and `__hash__` or neither
- Implement `__repr__` for debugging

❌ **DON'T:**
- Over-complicate with unnecessary magic methods
- Break expected behavior (e.g., `__add__` shouldn't delete items)
- Forget `functools.wraps` in decorators

#### EAFP vs LBYL

**EAFP (Easier to Ask for Forgiveness than Permission)** - Pythonic:
```python
try:
    return thing.quack()
except AttributeError:
    return "Can't quack"
```

**LBYL (Look Before You Leap)** - Less Pythonic:
```python
if hasattr(thing, 'quack'):
    return thing.quack()
else:
    return "Can't quack"
```

---

### 4. Abstraction

**Definition**: Hiding complex implementation details while exposing only essential features. Providing a simplified interface to complex systems.

#### Purpose
- **Reduce Complexity**: Hide unnecessary details
- **Define Contracts**: Specify what classes must implement
- **Enforce Implementation**: Ensure subclasses implement required methods
- **Separate Interface from Implementation**: "What" vs "how"

#### Python Implementation - ABC Module

```python
from abc import ABC, abstractmethod

class PaymentProcessor(ABC):
    """Abstract base class for payment processing"""

    @abstractmethod
    def process_payment(self, amount: float) -> bool:
        """Must be implemented by subclasses"""
        pass

    @abstractmethod
    def refund(self, transaction_id: str) -> bool:
        """Must be implemented by subclasses"""
        pass

    # Concrete method - shared by all subclasses
    def log_transaction(self, transaction_id: str):
        print(f"Transaction: {transaction_id}")

class StripePayment(PaymentProcessor):
    def process_payment(self, amount: float) -> bool:
        # Stripe-specific implementation
        return True

    def refund(self, transaction_id: str) -> bool:
        # Stripe-specific implementation
        return True

# Cannot instantiate ABC
# processor = PaymentProcessor()  # ❌ TypeError

# Can instantiate concrete implementation
processor = StripePayment()  # ✅ Works
```

#### Abstract Properties

```python
from abc import ABC, abstractmethod

class DatabaseConnection(ABC):
    @property
    @abstractmethod
    def connection_string(self) -> str:
        """Abstract property"""
        pass

    @abstractmethod
    def connect(self) -> bool:
        """Abstract method"""
        pass

class PostgreSQLConnection(DatabaseConnection):
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.host}:{self.port}/{self.db}"

    def connect(self) -> bool:
        # Implementation
        return True
```

#### Abstract vs Protocol (Python 3.8+)

**ABC (Explicit)**:
```python
from abc import ABC, abstractmethod

class Drawable(ABC):
    @abstractmethod
    def draw(self):
        pass

class Circle(Drawable):  # Must explicitly inherit
    def draw(self):
        pass
```

**Protocol (Structural)**:
```python
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None:
        ...

class Circle:  # No inheritance needed
    def draw(self) -> None:
        pass

# Duck typing with type checking
def render(shape: Drawable):
    shape.draw()
```

#### Best Practices

✅ **DO:**
- Use ABC for defining interfaces
- Keep abstract classes focused (SRP)
- Provide concrete helper methods when appropriate
- Document expected behavior

❌ **DON'T:**
- Make everything abstract
- Mix interface and implementation concerns
- Create "god" abstract classes

#### Real-World Use Cases
- **Plugin systems**: Define plugin interface
- **Framework development**: Define extension points
- **Data access layers**: Abstract different databases
- **UI components**: Define component interface

---

## Relationships Between Principles

The four pillars don't exist in isolation - they work together:

```
┌─────────────────────────────────────────────────┐
│                  ABSTRACTION                     │
│  (Define WHAT needs to be done)                 │
│                                                  │
│  ┌────────────────────────────────────────┐    │
│  │         ENCAPSULATION                   │    │
│  │  (Hide HOW it's done)                   │    │
│  │                                          │    │
│  │  ┌──────────────────────────────────┐  │    │
│  │  │      INHERITANCE                  │  │    │
│  │  │  (Reuse existing implementations) │  │    │
│  │  │                                    │  │    │
│  │  │   ┌─────────────────────────┐    │  │    │
│  │  │   │   POLYMORPHISM           │    │  │    │
│  │  │   │  (Different forms)       │    │  │    │
│  │  │   └─────────────────────────┘    │  │    │
│  │  └──────────────────────────────────┘  │    │
│  └────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

### How They Work Together

1. **Abstraction** defines the interface (what to do)
2. **Encapsulation** hides the implementation (how it's done)
3. **Inheritance** enables code reuse and hierarchies
4. **Polymorphism** allows different implementations of the same interface

### Example: Complete Integration

```python
# ABSTRACTION: Define interface
class NotificationService(ABC):
    @abstractmethod
    def send(self, message: str) -> bool:
        pass

# INHERITANCE: Create implementations
class EmailNotification(NotificationService):
    def __init__(self):
        # ENCAPSULATION: Hide implementation details
        self.__smtp_server = "smtp.gmail.com"
        self.__port = 587

    def send(self, message: str) -> bool:
        # Implementation details hidden
        return True

class SMSNotification(NotificationService):
    def send(self, message: str) -> bool:
        return True

# POLYMORPHISM: Same interface, different behavior
def notify_user(service: NotificationService, message: str):
    return service.send(message)  # Works with any service

notify_user(EmailNotification(), "Hello")  # Email
notify_user(SMSNotification(), "Hello")    # SMS
```

---

## SOLID Principles Connection

The four OOP pillars are the foundation for **SOLID principles** - five design principles that make software more maintainable and scalable.

> **📖 For comprehensive SOLID documentation with detailed examples, see [solid-principles.md](solid-principles.md)**

### Quick Overview of SOLID

**SOLID** is an acronym for:
- **S**ingle Responsibility Principle
- **O**pen/Closed Principle
- **L**iskov Substitution Principle
- **I**nterface Segregation Principle
- **D**ependency Inversion Principle

### How OOP Pillars Enable SOLID

The relationship between the four pillars and SOLID principles:

#### Single Responsibility Principle (SRP)
- **Enabled by**: Encapsulation
- **Principle**: A class should have only one reason to change
- **Connection**: Encapsulation naturally groups related functionality and data together, making it easier to identify and maintain single responsibilities
- **Example**: Separate `User` class from `UserRepository` and `EmailService` classes

```python
# Each class has a single, well-defined responsibility
class User:
    """Responsibility: Represent user data"""
    pass

class UserRepository:
    """Responsibility: Persist user data"""
    pass

class EmailService:
    """Responsibility: Send emails"""
    pass
```

#### Open/Closed Principle (OCP)
- **Enabled by**: Abstraction + Inheritance + Polymorphism
- **Principle**: Software entities should be open for extension but closed for modification
- **Connection**: Abstraction defines interfaces, inheritance creates new types, polymorphism allows interchangeable use
- **Example**: Add new payment methods without modifying existing code

```python
class PaymentMethod(ABC):
    @abstractmethod
    def process(self, amount):
        pass

class CreditCardPayment(PaymentMethod):
    def process(self, amount):
        # Implementation
        pass

# Can add new types without modifying existing code
class BitcoinPayment(PaymentMethod):
    def process(self, amount):
        # Implementation
        pass
```

#### Liskov Substitution Principle (LSP)
- **Enabled by**: Inheritance + Polymorphism
- **Principle**: Objects of a superclass should be replaceable with objects of a subclass without breaking the application
- **Connection**: Proper inheritance hierarchies and polymorphic behavior ensure substitutability
- **Example**: Any `Shape` subclass can replace the `Shape` base class

```python
def calculate_area(shape: Shape) -> float:
    return shape.area()

# Both work without modification
calculate_area(Circle(5))
calculate_area(Rectangle(4, 6))
```

#### Interface Segregation Principle (ISP)
- **Enabled by**: Abstraction
- **Principle**: Clients should not be forced to depend on interfaces they don't use
- **Connection**: Abstraction allows creating focused, specific interfaces rather than large, general ones
- **Example**: Split large interfaces into smaller, focused ones

```python
# Instead of one large interface
class Workable(ABC):
    @abstractmethod
    def work(self):
        pass

class Eatable(ABC):
    @abstractmethod
    def eat(self):
        pass

# Implement only what's needed
class Robot(Workable):  # Doesn't need Eatable
    pass

class Human(Workable, Eatable):  # Implements both
    pass
```

#### Dependency Inversion Principle (DIP)
- **Enabled by**: Abstraction
- **Principle**: High-level modules should not depend on low-level modules; both should depend on abstractions
- **Connection**: Abstraction provides the interfaces that both high-level and low-level modules can depend on
- **Example**: Depend on database abstraction, not concrete implementation

```python
class Database(ABC):
    @abstractmethod
    def save(self, data):
        pass

class OrderService:
    def __init__(self, db: Database):  # Depends on abstraction
        self.db = db

# Can inject any database implementation
service = OrderService(MySQLDatabase())
service = OrderService(PostgreSQLDatabase())
```

### Summary: OOP Pillars ↔ SOLID

| SOLID Principle | OOP Pillars Used | Benefit |
|----------------|------------------|---------|
| **SRP** | Encapsulation | Classes have focused, single responsibilities |
| **OCP** | Abstraction, Inheritance, Polymorphism | Extend without modifying existing code |
| **LSP** | Inheritance, Polymorphism | Subclasses work interchangeably with parents |
| **ISP** | Abstraction | Focused interfaces, no unused methods |
| **DIP** | Abstraction | Decouple high-level logic from implementation details |

> **💡 Key Insight**: The four OOP pillars are *how* you implement object-oriented design, while SOLID principles are *what* makes that design good. Master the pillars to apply SOLID effectively!

---

## Common Pitfalls

### Encapsulation Pitfalls

❌ **Over-encapsulation**
```python
class BadExample:
    def __init__(self):
        self.__value = 0

    def get_value(self):
        return self.__value

    def set_value(self, value):
        self.__value = value
```

✅ **Better approach**
```python
class GoodExample:
    def __init__(self):
        self._value = 0

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if value < 0:
            raise ValueError("Must be positive")
        self._value = value
```

### Inheritance Pitfalls

❌ **Deep inheritance hierarchies**
```python
class Animal: pass
class Mammal(Animal): pass
class Carnivore(Mammal): pass
class Feline(Carnivore): pass
class BigCat(Feline): pass
class Lion(BigCat): pass  # Too deep!
```

✅ **Prefer composition**
```python
class Lion:
    def __init__(self):
        self.diet = Carnivore()
        self.species = Feline()
```

### Polymorphism Pitfalls

❌ **Breaking LSP**
```python
class Bird:
    def fly(self):
        return "Flying"

class Penguin(Bird):
    def fly(self):
        raise Exception("Can't fly!")  # Breaks contract
```

✅ **Correct design**
```python
class Bird:
    pass

class FlyingBird(Bird):
    def fly(self):
        return "Flying"

class Penguin(Bird):  # Doesn't inherit fly()
    def swim(self):
        return "Swimming"
```

### Abstraction Pitfalls

❌ **Leaky abstraction**
```python
class Database(ABC):
    @abstractmethod
    def execute_sql(self, sql: str):  # Exposes SQL detail!
        pass
```

✅ **Proper abstraction**
```python
class Database(ABC):
    @abstractmethod
    def save(self, entity):  # Hides implementation
        pass

    @abstractmethod
    def find(self, id):
        pass
```

---

## Interview Preparation

### Common Questions

**Q: What are the four pillars of OOP?**
- Encapsulation, Inheritance, Polymorphism, Abstraction
- Be ready to explain each with examples

**Q: What's the difference between abstraction and encapsulation?**
- **Abstraction**: Hide complexity (what you can do)
- **Encapsulation**: Hide implementation (how it's done)
- Abstraction is design-level, encapsulation is implementation-level

**Q: Explain the diamond problem**
- Occurs in multiple inheritance
- Python solves with C3 linearization (MRO)
- Use `super()` to follow MRO correctly

**Q: When to use composition vs inheritance?**
- **Inheritance**: "is-a" relationship, polymorphism needed
- **Composition**: "has-a" relationship, more flexibility
- General rule: Favor composition over inheritance

**Q: What is duck typing?**
- "If it walks like a duck and quacks like a duck, it's a duck"
- Python focuses on behavior, not type
- No need for explicit interfaces

**Q: How does Python implement encapsulation differently than Java/C++?**
- No true private members
- Relies on conventions (`_` and `__`)
- Uses `@property` for getters/setters
- Philosophy: "We're all consenting adults here"

**Q: What are magic methods?**
- Special methods with double underscores
- Enable operator overloading and protocols
- Examples: `__init__`, `__str__`, `__add__`

**Q: Explain MRO (Method Resolution Order)**
- Determines method lookup order in inheritance
- Uses C3 linearization algorithm
- View with `ClassName.__mro__`
- Important for multiple inheritance

### Coding Interview Tips

1. **Start with requirements**: Clarify before coding
2. **Design first**: Sketch class hierarchy
3. **Apply SOLID**: Mention relevant principles
4. **Explain trade-offs**: Discuss alternatives
5. **Test edge cases**: Consider error conditions
6. **Refactor**: Show you can improve design

### Common Design Problems

- **Design a parking lot**: Inheritance, polymorphism
- **Design an elevator system**: State pattern, OOP
- **Design a library system**: Encapsulation, abstraction
- **Design a notification system**: Strategy pattern, polymorphism
- **Design a cache**: Encapsulation, interfaces

---

## Quick Reference

### Encapsulation

```python
class Example:
    def __init__(self):
        self.public = "accessible everywhere"
        self._protected = "internal use (convention)"
        self.__private = "name mangled"

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        if v >= 0:
            self._value = v
```

### Inheritance

```python
class Parent:
    def method(self):
        return "parent"

class Child(Parent):
    def method(self):
        return super().method() + " and child"

# MRO
print(Child.__mro__)
```

### Polymorphism

```python
# Duck typing
def process(thing):
    return thing.process()  # Works with any object with process()

# Operator overloading
class Vector:
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
```

### Abstraction

```python
from abc import ABC, abstractmethod

class Interface(ABC):
    @abstractmethod
    def required_method(self):
        pass

    def concrete_method(self):
        return "shared implementation"
```

### Access Levels Summary

| Syntax | Type | Accessibility |
|--------|------|--------------|
| `name` | Public | Everywhere |
| `_name` | Protected | Convention: internal only |
| `__name` | Private | Name mangled to `_Class__name` |

### Magic Methods Cheat Sheet

| Purpose | Methods |
|---------|---------|
| Construction | `__init__`, `__new__`, `__del__` |
| Representation | `__str__`, `__repr__`, `__format__` |
| Arithmetic | `__add__`, `__sub__`, `__mul__`, `__truediv__`, `__floordiv__`, `__mod__`, `__pow__` |
| Comparison | `__eq__`, `__ne__`, `__lt__`, `__le__`, `__gt__`, `__ge__` |
| Container | `__len__`, `__getitem__`, `__setitem__`, `__delitem__`, `__contains__`, `__iter__` |
| Callable | `__call__` |
| Context Manager | `__enter__`, `__exit__` |
| Attribute Access | `__getattr__`, `__setattr__`, `__delattr__`, `__getattribute__` |

---

## Resources

### Documentation
- [Python Classes Tutorial](https://docs.python.org/3/tutorial/classes.html)
- [Python Data Model](https://docs.python.org/3/reference/datamodel.html)
- [ABC Module](https://docs.python.org/3/library/abc.html)
- [PEP 8 - Style Guide](https://pep8.org/)

### Books
- "Fluent Python" by Luciano Ramalho
- "Python Cookbook" by David Beazley & Brian K. Jones
- "Effective Python" by Brett Slatkin
- "Design Patterns: Elements of Reusable Object-Oriented Software" (Gang of Four)

### Online Resources
- [Real Python - OOP](https://realpython.com/python3-object-oriented-programming/)
- [Refactoring Guru - Design Patterns](https://refactoring.guru/design-patterns/python)
- [Python Design Patterns](https://python-patterns.guide/)

### Practice
- [LeetCode - Object-Oriented Design](https://leetcode.com/problemset/all/?topicSlugs=object-oriented-design)
- Implement design patterns from `design_patterns/` directory
- Complete exercises in `interview/oop_principles.ipynb`

---

## Summary

The four pillars of OOP work together to create maintainable, scalable, and flexible software:

1. **Encapsulation** protects and controls access to data
2. **Inheritance** enables code reuse and hierarchical organization
3. **Polymorphism** allows flexible interfaces and duck typing
4. **Abstraction** hides complexity and defines contracts

Python's approach emphasizes **practicality** and **flexibility** with:
- Duck typing over strict typing
- Properties for Pythonic encapsulation
- Multiple inheritance with MRO
- ABC module for abstract interfaces

Master these principles to write better object-oriented Python code!
