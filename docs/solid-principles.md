# SOLID Principles - Comprehensive Guide

The SOLID principles are five design principles that help create maintainable, scalable, and robust object-oriented software.

## Table of Contents

- [Overview](#overview)
- [Single Responsibility Principle (SRP)](#single-responsibility-principle)
- [Open/Closed Principle (OCP)](#open-closed-principle)
- [Liskov Substitution Principle (LSP)](#liskov-substitution-principle)
- [Interface Segregation Principle (ISP)](#interface-segregation-principle)
- [Dependency Inversion Principle (DIP)](#dependency-inversion-principle)
- [SOLID Together](#solid-together)
- [Relationship to OOP](#relationship-to-oop)
- [Interview Q&A](#interview-qa)
- [Quick Reference](#quick-reference)

---

## Overview

**SOLID** is an acronym for five object-oriented design principles:

| Principle | Summary |
|-----------|---------|
| **S**ingle Responsibility | A class should have one, and only one, reason to change |
| **O**pen/Closed | Open for extension, closed for modification |
| **L**iskov Substitution | Subtypes must be substitutable for their base types |
| **I**nterface Segregation | Many specific interfaces better than one general |
| **D**ependency Inversion | Depend on abstractions, not concretions |

**Benefits:**
- Easier to maintain and modify
- More testable code
- Better code organization
- Reduced coupling
- Increased cohesion

**When to apply:** Use SOLID as guidelines, not absolute rules. Sometimes pragmatism trumps perfect adherence.

---

## Single Responsibility Principle

> "A class should have one, and only one, reason to change" - Robert C. Martin

### Definition

Each class should have only one job or responsibility. If a class has multiple responsibilities, they become coupled, and changes to one responsibility may affect others.

### Violation Signs

- Class name contains "And", "Or", "Manager", "Handler"
- Methods operate on different subsets of class attributes
- Changes for unrelated reasons affect the same class
- Class is difficult to name clearly

### Examples

#### ❌ BAD: Multiple Responsibilities

```python
class User:
    """Violates SRP - handles data, validation, persistence, and email"""

    def __init__(self, name, email):
        self.name = name
        self.email = email

    # Responsibility 1: Data validation
    def validate_email(self):
        return '@' in self.email

    # Responsibility 2: Database operations
    def save_to_database(self):
        # Direct database access
        import sqlite3
        conn = sqlite3.connect('users.db')
        conn.execute(f"INSERT INTO users VALUES ('{self.name}', '{self.email}')")
        conn.commit()

    # Responsibility 3: Email notifications
    def send_welcome_email(self):
        import smtplib
        # Email sending logic
        print(f"Sending welcome email to {self.email}")

    # Responsibility 4: Report generation
    def generate_report(self):
        return f"User Report: {self.name} - {self.email}"
```

**Problems:**
- Changes to database schema affect User class
- Changes to email template affect User class
- Changes to report format affect User class
- Difficult to test (requires database, email server)
- High coupling

#### ✅ GOOD: Single Responsibility

```python
# Responsibility 1: User data model
class User:
    """Single responsibility: Represent user data"""

    def __init__(self, name, email):
        self.name = name
        self.email = email

# Responsibility 2: Email validation
class EmailValidator:
    """Single responsibility: Validate emails"""

    @staticmethod
    def is_valid(email):
        return '@' in email and '.' in email

# Responsibility 3: Database operations
class UserRepository:
    """Single responsibility: Persist users"""

    def __init__(self, database):
        self.database = database

    def save(self, user):
        self.database.execute(
            "INSERT INTO users VALUES (?, ?)",
            (user.name, user.email)
        )

# Responsibility 4: Email notifications
class EmailService:
    """Single responsibility: Send emails"""

    def send_welcome_email(self, user):
        print(f"Sending welcome email to {user.email}")

# Responsibility 5: Report generation
class UserReportGenerator:
    """Single responsibility: Generate user reports"""

    def generate(self, user):
        return f"User Report: {user.name} - {user.email}"
```

**Benefits:**
- Each class has one reason to change
- Easy to test in isolation
- Easy to reuse components
- Clear, focused classes

### Real-World Applications

- **Django Models**: Models only represent data structure
- **Service Layer**: Business logic separated from controllers
- **Repository Pattern**: Data access separated from business logic

---

## Open/Closed Principle

> "Software entities should be open for extension, but closed for modification" - Bertrand Meyer

### Definition

You should be able to extend a class's behavior without modifying its existing code. Use abstraction and polymorphism to add new functionality.

### Violation Signs

- Frequent modifications to existing code for new features
- Long if/elif/else or switch statements for type checking
- Modifying tested code to add features

### Examples

#### ❌ BAD: Modification Required

```python
class PaymentProcessor:
    """Violates OCP - must modify class to add payment types"""

    def process_payment(self, payment_type, amount):
        if payment_type == 'credit_card':
            print(f"Processing ${amount} via credit card")
            # Credit card specific logic
        elif payment_type == 'paypal':
            print(f"Processing ${amount} via PayPal")
            # PayPal specific logic
        elif payment_type == 'bitcoin':  # Added later - MODIFICATION!
            print(f"Processing ${amount} via Bitcoin")
            # Bitcoin specific logic
        # Each new payment type requires MODIFYING this class
```

**Problems:**
- Must modify and retest existing code for new features
- Risk of breaking existing functionality
- Violates "closed for modification"

#### ✅ GOOD: Extension Without Modification

```python
from abc import ABC, abstractmethod

# Abstract base - defines contract
class PaymentMethod(ABC):
    """Open for extension - create new subclasses"""

    @abstractmethod
    def process(self, amount):
        pass

# Existing implementations
class CreditCardPayment(PaymentMethod):
    def process(self, amount):
        print(f"Processing ${amount} via credit card")

class PayPalPayment(PaymentMethod):
    def process(self, amount):
        print(f"Processing ${amount} via PayPal")

# NEW payment method - no modification to existing code!
class BitcoinPayment(PaymentMethod):
    def process(self, amount):
        print(f"Processing ${amount} via Bitcoin")

# Processor works with any PaymentMethod
class PaymentProcessor:
    """Closed for modification - works with all PaymentMethod types"""

    def process_payment(self, payment_method: PaymentMethod, amount):
        payment_method.process(amount)  # Polymorphism!

# Usage - extensible without modification
processor = PaymentProcessor()
processor.process_payment(CreditCardPayment(), 100)
processor.process_payment(PayPalPayment(), 200)
processor.process_payment(BitcoinPayment(), 300)  # New type, no changes!
```

**Benefits:**
- Add new functionality without touching existing code
- Existing code remains stable and tested
- Easy to add new types

### Design Patterns Supporting OCP

- **Strategy Pattern**: Interchangeable algorithms
- **Template Method**: Override specific steps
- **Decorator Pattern**: Add functionality dynamically
- **Factory Pattern**: Create new types without modification

---

## Liskov Substitution Principle

> "Objects of a superclass should be replaceable with objects of a subclass without breaking the application" - Barbara Liskov

### Definition

If S is a subtype of T, then objects of type T can be replaced with objects of type S without altering program correctness. Subtypes must honor the contract of their base types.

### Violation Signs

- Subclass throws exceptions that parent doesn't
- Subclass strengthens preconditions
- Subclass weakens postconditions
- Subclass returns different types
- Using `isinstance()` checks to handle subtypes differently

### Examples

#### ❌ BAD: Violates LSP

```python
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def set_width(self, width):
        self.width = width

    def set_height(self, height):
        self.height = height

    def area(self):
        return self.width * self.height

class Square(Rectangle):
    """Violates LSP - breaks Rectangle's contract"""

    def set_width(self, width):
        # Square must maintain width == height
        self.width = width
        self.height = width  # Unexpected side effect!

    def set_height(self, height):
        self.width = height
        self.height = height  # Unexpected side effect!

# Code expecting Rectangle behavior
def resize_rectangle(rect: Rectangle):
    rect.set_width(5)
    rect.set_height(4)
    assert rect.area() == 20  # Expects 5 * 4 = 20

# Works with Rectangle
rect = Rectangle(0, 0)
resize_rectangle(rect)  # ✓ Passes

# Breaks with Square
square = Square(0, 0)
resize_rectangle(square)  # ✗ AssertionError! area() = 16, not 20
```

**Problems:**
- Square cannot be substituted for Rectangle
- Breaks expected behavior
- Violates "is-a" relationship assumption

#### ✅ GOOD: Respects LSP

```python
from abc import ABC, abstractmethod

# Proper abstraction
class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def set_width(self, width):
        self.width = width

    def set_height(self, height):
        self.height = height

    def area(self):
        return self.width * self.height

class Square(Shape):
    def __init__(self, side):
        self.side = side

    def set_side(self, side):
        self.side = side

    def area(self):
        return self.side * self.side

# Code works with Shape abstraction
def print_area(shape: Shape):
    print(f"Area: {shape.area()}")

# Both work correctly
print_area(Rectangle(5, 4))  # Area: 20
print_area(Square(5))        # Area: 25
```

**Benefits:**
- Subtypes are truly substitutable
- No surprising behavior
- Correct abstraction hierarchy

### LSP Guidelines

**DO:**
- Honor parent class contracts
- Maintain expected behavior
- Keep or relax preconditions
- Keep or strengthen postconditions

**DON'T:**
- Throw new exceptions
- Change return types (except to subtypes)
- Modify expected state changes
- Require type checking in client code

---

## Interface Segregation Principle

> "Clients should not be forced to depend on interfaces they don't use" - Robert C. Martin

### Definition

Many specific interfaces are better than one general-purpose interface. Don't force classes to implement methods they don't need.

### Violation Signs

- Interface has many methods
- Implementations leave methods empty or throw NotImplementedError
- Classes implement interface but only use subset of methods
- Interface named "I...Manager" or "I...Handler"

### Examples

#### ❌ BAD: Fat Interface

```python
from abc import ABC, abstractmethod

class Worker(ABC):
    """Fat interface - forces all workers to implement all methods"""

    @abstractmethod
    def work(self):
        pass

    @abstractmethod
    def eat_lunch(self):
        pass

    @abstractmethod
    def get_salary(self):
        pass

    @abstractmethod
    def attend_meeting(self):
        pass

class HumanWorker(Worker):
    def work(self):
        print("Human working")

    def eat_lunch(self):
        print("Human eating lunch")

    def get_salary(self):
        return 50000

    def attend_meeting(self):
        print("Human attending meeting")

class RobotWorker(Worker):
    """Forced to implement methods it doesn't need"""

    def work(self):
        print("Robot working")

    def eat_lunch(self):
        # Robots don't eat!
        raise NotImplementedError("Robots don't eat")

    def get_salary(self):
        # Robots don't get salary!
        raise NotImplementedError("Robots don't get salary")

    def attend_meeting(self):
        # Robots don't attend meetings!
        raise NotImplementedError("Robots don't attend meetings")
```

**Problems:**
- RobotWorker forced to implement irrelevant methods
- Throws exceptions for methods it shouldn't have
- Interface too broad

#### ✅ GOOD: Segregated Interfaces

```python
from abc import ABC, abstractmethod

# Specific, focused interfaces
class Workable(ABC):
    @abstractmethod
    def work(self):
        pass

class Eatable(ABC):
    @abstractmethod
    def eat_lunch(self):
        pass

class Payable(ABC):
    @abstractmethod
    def get_salary(self):
        pass

class MeetingAttendee(ABC):
    @abstractmethod
    def attend_meeting(self):
        pass

# Implementations choose relevant interfaces
class HumanWorker(Workable, Eatable, Payable, MeetingAttendee):
    def work(self):
        print("Human working")

    def eat_lunch(self):
        print("Human eating lunch")

    def get_salary(self):
        return 50000

    def attend_meeting(self):
        print("Human attending meeting")

class RobotWorker(Workable):
    """Only implements what it needs"""

    def work(self):
        print("Robot working")

# Client code depends on specific interfaces
def manage_work(worker: Workable):
    worker.work()

def handle_lunch(eater: Eatable):
    eater.eat_lunch()

# Works correctly with both
human = HumanWorker()
robot = RobotWorker()

manage_work(human)  # ✓
manage_work(robot)  # ✓

handle_lunch(human)  # ✓
# handle_lunch(robot)  # Type error - robot isn't Eatable
```

**Benefits:**
- Classes implement only needed methods
- Clear, focused interfaces
- No dummy implementations
- Better composition

### Python-Specific: Duck Typing

Python's duck typing reduces need for explicit interfaces, but ISP still applies:

```python
# No need for formal interface
class Bird:
    def fly(self):
        print("Bird flying")

class Fish:
    def swim(self):
        print("Fish swimming")

# Functions depend on specific behavior
def make_it_fly(thing):
    thing.fly()  # Only needs fly()

def make_it_swim(thing):
    thing.swim()  # Only needs swim()
```

---

## Dependency Inversion Principle

> "Depend upon abstractions, not concretions" - Robert C. Martin

### Definition

Two parts:
1. High-level modules should not depend on low-level modules. Both should depend on abstractions.
2. Abstractions should not depend on details. Details should depend on abstractions.

### Violation Signs

- High-level code directly instantiates low-level classes
- Difficult to test (hard dependencies)
- Changes in low-level modules break high-level modules
- No use of interfaces/abstractions

### Examples

#### ❌ BAD: Direct Dependencies

```python
# Low-level module
class MySQLDatabase:
    def query(self, sql):
        print(f"Executing MySQL query: {sql}")
        return [{"id": 1, "name": "Alice"}]

# High-level module depends on concrete implementation
class UserService:
    """Violates DIP - depends on concrete MySQLDatabase"""

    def __init__(self):
        self.db = MySQLDatabase()  # Direct dependency!

    def get_users(self):
        return self.db.query("SELECT * FROM users")

# Problems:
# 1. Can't switch to PostgreSQL without modifying UserService
# 2. Hard to test (requires MySQL)
# 3. Tight coupling
```

**Problems:**
- Cannot easily switch database implementations
- Difficult to test in isolation
- High-level logic coupled to low-level details

#### ✅ GOOD: Depend on Abstractions

```python
from abc import ABC, abstractmethod

# Abstraction
class Database(ABC):
    """Abstract interface - both depend on this"""

    @abstractmethod
    def query(self, sql):
        pass

# Low-level modules implement abstraction
class MySQLDatabase(Database):
    def query(self, sql):
        print(f"Executing MySQL query: {sql}")
        return [{"id": 1, "name": "Alice"}]

class PostgreSQLDatabase(Database):
    def query(self, sql):
        print(f"Executing PostgreSQL query: {sql}")
        return [{"id": 1, "name": "Alice"}]

class MongoDatabase(Database):
    def query(self, sql):
        print(f"Executing Mongo query: {sql}")
        return [{"_id": 1, "name": "Alice"}]

# High-level module depends on abstraction
class UserService:
    """Depends on Database abstraction"""

    def __init__(self, database: Database):
        self.db = database  # Injected dependency!

    def get_users(self):
        return self.db.query("SELECT * FROM users")

# Dependency Injection - configure at runtime
mysql_service = UserService(MySQLDatabase())
postgres_service = UserService(PostgreSQLDatabase())
mongo_service = UserService(MongoDatabase())

# Easy to test with mock
class MockDatabase(Database):
    def query(self, sql):
        return [{"id": 999, "name": "Test User"}]

test_service = UserService(MockDatabase())
```

**Benefits:**
- Easy to swap implementations
- Testable with mocks
- Low coupling
- Flexible configuration

### Dependency Injection Methods

**1. Constructor Injection** (most common):
```python
class Service:
    def __init__(self, dependency: AbstractDependency):
        self.dependency = dependency
```

**2. Property Injection**:
```python
class Service:
    def set_dependency(self, dependency: AbstractDependency):
        self.dependency = dependency
```

**3. Method Injection**:
```python
class Service:
    def do_work(self, dependency: AbstractDependency):
        dependency.execute()
```

---

## SOLID Together

The principles work together to create maintainable systems:

```python
from abc import ABC, abstractmethod
from typing import List

# DIP: Depend on abstractions
class NotificationChannel(ABC):
    @abstractmethod
    def send(self, message: str) -> None:
        pass

# ISP: Small, focused interfaces
class EmailChannel(NotificationChannel):
    def send(self, message: str) -> None:
        print(f"Email: {message}")

class SMSChannel(NotificationChannel):
    def send(self, message: str) -> None:
        print(f"SMS: {message}")

# SRP: Single responsibility - send notifications
class NotificationService:
    def __init__(self, channel: NotificationChannel):
        self.channel = channel

    def notify(self, message: str) -> None:
        self.channel.send(message)

# OCP: Open for extension - add new channels without modification
class PushChannel(NotificationChannel):
    def send(self, message: str) -> None:
        print(f"Push: {message}")

# LSP: All channels are substitutable
def send_alert(service: NotificationService, message: str):
    service.notify(message)  # Works with any channel

# Usage - all principles in action
email_service = NotificationService(EmailChannel())
sms_service = NotificationService(SMSChannel())
push_service = NotificationService(PushChannel())

send_alert(email_service, "Hello via Email")
send_alert(sms_service, "Hello via SMS")
send_alert(push_service, "Hello via Push")
```

---

## Relationship to OOP

SOLID principles build on the four OOP pillars:

| SOLID Principle | Enabled By | How |
|----------------|------------|-----|
| **SRP** | Encapsulation | Hide implementation, expose focused interface |
| **OCP** | Inheritance + Polymorphism | Extend through subclasses |
| **LSP** | Inheritance + Polymorphism | Proper substitutability |
| **ISP** | Abstraction | Define specific contracts |
| **DIP** | Abstraction | Depend on interfaces, not implementations |

---

## Interview Q&A

### Q1: What are the SOLID principles?

**Answer**: SOLID is an acronym for five object-oriented design principles:
- **S**ingle Responsibility: Each class has one job
- **O**pen/Closed: Open for extension, closed for modification
- **L**iskov Substitution: Subtypes must be substitutable
- **I**nterface Segregation: Many specific interfaces over one general
- **D**ependency Inversion: Depend on abstractions, not concretions

### Q2: Give an example of SRP violation

**Answer**: A User class that handles data validation, database persistence, email notifications, and report generation violates SRP. Each responsibility should be a separate class: User (data model), EmailValidator, UserRepository, EmailService, ReportGenerator.

### Q3: How does OCP relate to design patterns?

**Answer**: Many patterns support OCP:
- Strategy Pattern: Add new algorithms without modifying context
- Template Method: Override specific steps without changing skeleton
- Decorator: Add functionality without modifying original class
- Factory: Add new product types without modifying factory interface

### Q4: What's the classic LSP violation example?

**Answer**: Square inheriting from Rectangle. If Rectangle allows independent width/height changes, Square breaks this contract because it must maintain width == height. Solution: Both inherit from Shape instead.

### Q5: How do you apply DIP in Python?

**Answer**: Use Abstract Base Classes (ABC) to define interfaces. High-level modules depend on ABC, low-level modules implement ABC. Use dependency injection to provide concrete implementations at runtime.

### Q6: When is it okay to violate SOLID?

**Answer**:
- Small, simple scripts where SOLID adds complexity
- Prototypes and throwaway code
- When performance is critical and abstraction has measurable cost
- When team size/timeline doesn't justify the overhead

**Rule**: Know the principles, understand trade-offs, be pragmatic.

---

## Quick Reference

### Violation Smells

| Principle | Smell |
|-----------|-------|
| **SRP** | God classes, "Manager"/"Handler" names |
| **OCP** | Long if/elif chains, frequent modifications |
| **LSP** | Type checking, isinstance() for subtypes |
| **ISP** | NotImplementedError, empty method bodies |
| **DIP** | Direct instantiation, hard to test |

### Refactoring Strategies

**SRP Violation → Extract Class**
```python
# Before: One class, multiple responsibilities
class UserManager:
    def validate(): ...
    def save(): ...
    def send_email(): ...

# After: Separate classes
class User: ...
class UserValidator: ...
class UserRepository: ...
class EmailService: ...
```

**OCP Violation → Strategy/Template Method**
```python
# Before: if/elif for types
if type == 'A':
    # logic A
elif type == 'B':
    # logic B

# After: Polymorphism
strategy.execute()
```

**LSP Violation → Proper Abstraction**
```python
# Before: Square extends Rectangle (breaks contract)
class Square(Rectangle): ...

# After: Both extend Shape
class Rectangle(Shape): ...
class Square(Shape): ...
```

**ISP Violation → Split Interface**
```python
# Before: Fat interface
class Worker(ABC):
    @abstractmethod
    def work(): ...
    @abstractmethod
    def eat(): ...

# After: Focused interfaces
class Workable(ABC): ...
class Eatable(ABC): ...
```

**DIP Violation → Dependency Injection**
```python
# Before: Direct dependency
class Service:
    def __init__(self):
        self.db = MySQLDatabase()

# After: Inject abstraction
class Service:
    def __init__(self, db: Database):
        self.db = db
```

---

## Resources

**Books:**
- "Clean Code" by Robert C. Martin
- "Agile Software Development: Principles, Patterns, and Practices" by Robert C. Martin
- "Design Patterns: Elements of Reusable Object-Oriented Software" (Gang of Four)

**Online:**
- [SOLID Principles Explained](https://en.wikipedia.org/wiki/SOLID)
- [Uncle Bob's Blog](http://blog.cleancoder.com/)
- [Refactoring Guru - SOLID](https://refactoring.guru/solid-principles)

**Practice:**
- Review existing code for violations
- Refactor to apply SOLID
- Study design patterns that support SOLID
- Complete exercises in `interview/solid_principles.ipynb`

---

## Summary

SOLID principles guide us toward:
- **Maintainable** code (easy to modify)
- **Testable** code (easy to test in isolation)
- **Flexible** code (easy to extend)
- **Robust** code (less likely to break)

**Key Takeaways:**
1. Each principle addresses specific design problems
2. Principles work together, not in isolation
3. Use as guidelines, not absolute rules
4. Balance principles with pragmatism
5. Refactor violations when they cause pain

Master SOLID to write better object-oriented code! 🚀
