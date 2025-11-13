# Design Patterns - Complete Learning Guide

A comprehensive guide to all 23 Gang of Four (GoF) design patterns with Python implementations and practical examples.

## Table of Contents

- [Overview](#overview)
- [Pattern Categories](#pattern-categories)
- [Creational Patterns](#creational-patterns)
- [Structural Patterns](#structural-patterns)
- [Behavioral Patterns](#behavioral-patterns)
- [Pattern Relationships](#pattern-relationships)
- [Choosing the Right Pattern](#choosing-the-right-pattern)

---

## Overview

Design patterns are reusable solutions to commonly occurring problems in software design. They represent best practices evolved over time by experienced object-oriented software developers.

### Why Learn Design Patterns?

- **Communication**: Common vocabulary for developers
- **Proven Solutions**: Battle-tested approaches to common problems
- **Code Quality**: Improved maintainability, flexibility, and scalability
- **Career Growth**: Essential knowledge for technical interviews and senior roles

### Pattern Structure

Each pattern is described with:
- **Intent**: What problem does it solve?
- **Problem**: When should you use it?
- **Solution**: How does it work?
- **Implementation**: Python code examples
- **Use Cases**: Real-world applications
- **Consequences**: Trade-offs and limitations

---

## Pattern Categories

### Creational Patterns (5)
Deal with object creation mechanisms, trying to create objects in a manner suitable to the situation.

### Structural Patterns (7)
Deal with object composition, creating relationships between objects to form larger structures.

### Behavioral Patterns (11)
Deal with communication between objects, how objects interact and distribute responsibility.

---

## Creational Patterns

### 1. Factory Method Pattern

**Intent**: Define an interface for creating objects, but let subclasses decide which class to instantiate.

**Problem**:
- You need to create objects without specifying exact classes
- Object creation logic is complex or needs to be centralized
- You want to delegate instantiation to subclasses

**Solution**: Use an abstract creator with a factory method that returns objects of an abstract product type. Concrete creators override this method to create specific products.

**Python Implementation**: `design_patterns/factory_pattern.ipynb`

**Real-World Examples**:
- GUI frameworks (create Windows/Mac specific components)
- Document generators (PDF, HTML, XML)
- Database connectors (MySQL, PostgreSQL, MongoDB)
- Plugin systems

**Benefits**:
- ✅ Loose coupling between creator and concrete products
- ✅ Single Responsibility Principle (creation logic in one place)
- ✅ Open/Closed Principle (add new products without changing existing code)

**Trade-offs**:
- ❌ Code can become complex with many subclasses
- ❌ Requires creating new subclass for each product type

**Related Patterns**: Abstract Factory, Template Method, Prototype

---

### 2. Abstract Factory Pattern

**Intent**: Provide an interface for creating families of related or dependent objects without specifying their concrete classes.

**Problem**:
- Your system needs to work with multiple families of related products
- You want to ensure products from the same family are used together
- You need to support multiple look-and-feel standards

**Solution**: Create an abstract factory interface with methods for creating each type of product. Concrete factories implement these methods to create product families.

**Python Implementation**: `design_patterns/abstract_factory_pattern.ipynb`

**Real-World Examples**:
- UI toolkits (Windows/Mac/Linux themes)
- Cross-platform applications
- Database abstraction layers
- Game engines (different renderer families)

**Benefits**:
- ✅ Ensures product family consistency
- ✅ Isolates concrete classes from client code
- ✅ Easy to swap entire product families

**Trade-offs**:
- ❌ Adding new product types requires changing all factories
- ❌ More complex than Factory Method

**Related Patterns**: Factory Method, Singleton, Prototype

---

### 3. Builder Pattern

**Intent**: Separate the construction of a complex object from its representation, allowing the same construction process to create different representations.

**Problem**:
- Object has many optional parameters (telescoping constructor problem)
- Object construction requires multiple steps
- You want immutable objects with many fields

**Solution**: Provide a builder class that constructs the object step by step. Optionally use a director to encapsulate construction logic.

**Python Implementation**: `design_patterns/builder_pattern.ipynb`

**Real-World Examples**:
- SQL query builders
- HTTP request builders
- Configuration objects
- Complex data structures (trees, graphs)
- Test data generation

**Benefits**:
- ✅ Construct objects step-by-step
- ✅ Reuse same construction code for different representations
- ✅ Isolate complex construction code
- ✅ Supports fluent interfaces (method chaining)

**Trade-offs**:
- ❌ More code complexity
- ❌ Requires creating new builder for each product type

**Python-Specific Note**: Python's keyword arguments and dataclasses can sometimes make builders unnecessary for simple cases.

**Related Patterns**: Abstract Factory, Composite, Fluent Interface

---

### 4. Singleton Pattern

**Intent**: Ensure a class has only one instance and provide a global point of access to it.

**Problem**:
- You need exactly one instance of a class (e.g., configuration, logger, connection pool)
- You need controlled access to a single object
- The single instance needs to be accessible from anywhere

**Solution**: Make constructor private and create a static method that creates/returns the single instance.

**Python Implementation**: `design_patterns/singleton_pattern.ipynb`

**Real-World Examples**:
- Configuration managers
- Logging systems
- Database connection pools
- Thread pools
- Caching systems

**Benefits**:
- ✅ Controlled access to single instance
- ✅ Reduced namespace pollution
- ✅ Permits refinement of operations and representation

**Trade-offs**:
- ❌ Violates Single Responsibility Principle
- ❌ Difficult to unit test
- ❌ Can hide dependencies
- ❌ Requires special treatment in multithreaded environments

**Python-Specific Considerations**:
- Module-level variables are already singletons
- Metaclasses can implement singleton behavior
- `__new__` method can control instantiation
- Thread-safe implementations need locks

**Related Patterns**: Abstract Factory, Builder, Prototype

---

### 5. Prototype Pattern

**Intent**: Specify the kinds of objects to create using a prototypical instance, and create new objects by copying this prototype.

**Problem**:
- Object creation is expensive (database query, network call)
- You want to avoid subclasses of object creator
- Object configurations need to be cloned

**Solution**: Create new objects by cloning existing prototype objects.

**Python Implementation**: `design_patterns/prototype_pattern.ipynb`

**Real-World Examples**:
- Game object spawning (clone enemy templates)
- Document templates
- Configuration presets
- Cell division in simulations

**Benefits**:
- ✅ Reduce subclassing
- ✅ Hide construction complexity
- ✅ Add/remove products at runtime
- ✅ More efficient than creating from scratch

**Trade-offs**:
- ❌ Cloning complex objects with circular references can be tricky
- ❌ Deep copy vs shallow copy considerations

**Python-Specific**: Use `copy.copy()` (shallow) or `copy.deepcopy()` (deep) from standard library.

**Related Patterns**: Abstract Factory, Composite, Decorator

---

## Structural Patterns

### 6. Adapter Pattern

**Intent**: Convert the interface of a class into another interface clients expect. Adapter lets classes work together that couldn't otherwise because of incompatible interfaces.

**Problem**:
- You need to use an existing class with incompatible interface
- You want to create reusable class that cooperates with unrelated classes
- Legacy system integration

**Solution**: Create an adapter class that wraps the incompatible object and translates calls to its interface.

**Python Implementation**: `design_patterns/adapter_pattern.ipynb`

**Real-World Examples**:
- Third-party library integration
- Legacy code integration
- Multiple data source adapters (REST API, database, file system)
- Payment gateway adapters

**Benefits**:
- ✅ Single Responsibility Principle
- ✅ Open/Closed Principle
- ✅ Reuse existing code

**Trade-offs**:
- ❌ Overall complexity increases (new interfaces/classes)

**Python-Specific**: Can use multiple inheritance or composition.

**Related Patterns**: Bridge, Decorator, Proxy

---

### 7. Bridge Pattern

**Intent**: Decouple an abstraction from its implementation so the two can vary independently.

**Problem**:
- You want to avoid permanent binding between abstraction and implementation
- Both abstractions and implementations should be extensible
- Changes in implementation shouldn't affect clients

**Solution**: Separate abstraction hierarchy from implementation hierarchy, connect via composition.

**Python Implementation**: `design_patterns/bridge_pattern.ipynb`

**Real-World Examples**:
- GUI frameworks (separate UI from platform-specific rendering)
- Database drivers (abstraction: query builder, implementation: DB-specific driver)
- Device drivers
- Remote controls and devices

**Benefits**:
- ✅ Separate interface from implementation
- ✅ Improved extensibility
- ✅ Hide implementation details from clients

**Trade-offs**:
- ❌ Increased complexity for highly cohesive classes

**Related Patterns**: Abstract Factory, Adapter, Strategy

---

### 8. Composite Pattern

**Intent**: Compose objects into tree structures to represent part-whole hierarchies. Composite lets clients treat individual objects and compositions uniformly.

**Problem**:
- You need to represent part-whole hierarchies
- You want clients to treat individual objects and compositions uniformly
- Tree structures (file systems, UI components, organizational charts)

**Solution**: Create a component interface used by both leaf and composite objects. Composites contain components (leaves or other composites).

**Python Implementation**: `design_patterns/composite_pattern.ipynb`

**Real-World Examples**:
- File systems (files and folders)
- UI component hierarchies (containers and widgets)
- Graphics systems (shapes grouped together)
- Organization structures
- Menu systems

**Benefits**:
- ✅ Defines class hierarchies of primitive and complex objects
- ✅ Makes client code simple
- ✅ Easy to add new component types

**Trade-offs**:
- ❌ Can make design overly general
- ❌ Difficult to restrict composite components

**Related Patterns**: Decorator, Iterator, Visitor, Chain of Responsibility

---

### 9. Decorator Pattern

**Intent**: Attach additional responsibilities to an object dynamically. Decorators provide a flexible alternative to subclassing for extending functionality.

**Problem**:
- You need to add responsibilities to objects dynamically
- You want to avoid class explosion from subclassing
- Extension by subclassing is impractical

**Solution**: Wrap object in decorator objects that add new behavior, keeping the same interface.

**Python Implementation**: `design_patterns/decorator_pattern.ipynb`

**Real-World Examples**:
- Stream I/O (BufferedReader wrapping FileReader)
- Middleware in web frameworks
- Logging, caching, authorization wrappers
- UI components with scrollbars, borders

**Benefits**:
- ✅ More flexible than inheritance
- ✅ Avoid feature-laden classes high in hierarchy
- ✅ Add/remove responsibilities at runtime

**Trade-offs**:
- ❌ Many small objects in system
- ❌ Decorators and components aren't identical (type checking issues)

**Python-Specific**: Don't confuse with Python's `@decorator` syntax (function decorators), though the concept is similar.

**Related Patterns**: Adapter, Composite, Strategy, Proxy

---

### 10. Facade Pattern

**Intent**: Provide a unified interface to a set of interfaces in a subsystem. Facade defines a higher-level interface that makes the subsystem easier to use.

**Problem**:
- Subsystem is complex with many classes
- You want to provide a simple interface to a complex system
- You want to layer your subsystems

**Solution**: Create a facade class that provides simple methods that delegate to subsystem classes.

**Python Implementation**: `design_patterns/facade_pattern.ipynb`

**Real-World Examples**:
- Compiler facade (lexical analysis, parsing, code generation)
- Video conversion libraries
- Database ORM layers
- Home automation systems
- Complex library APIs

**Benefits**:
- ✅ Shields clients from subsystem complexity
- ✅ Promotes weak coupling
- ✅ Doesn't prevent access to subsystem directly

**Trade-offs**:
- ❌ Facade can become a god object
- ❌ May hide important complexity from developers

**Related Patterns**: Abstract Factory, Mediator, Singleton

---

### 11. Flyweight Pattern

**Intent**: Use sharing to support large numbers of fine-grained objects efficiently.

**Problem**:
- Application uses large number of objects
- Storage costs are high due to sheer quantity
- Most object state can be made extrinsic

**Solution**: Share common state (intrinsic) between objects, pass varying state (extrinsic) as parameters.

**Python Implementation**: `design_patterns/flyweight_pattern.ipynb`

**Real-World Examples**:
- Text editors (character objects sharing font data)
- Game development (particle systems, tree rendering)
- String interning
- Connection pools

**Benefits**:
- ✅ Reduces memory usage
- ✅ Can handle huge number of objects

**Trade-offs**:
- ❌ Increased complexity
- ❌ Runtime costs for transferring/computing extrinsic state

**Python-Specific**: Python already uses flyweight for small integers and string interning.

**Related Patterns**: Composite, State, Strategy

---

### 12. Proxy Pattern

**Intent**: Provide a surrogate or placeholder for another object to control access to it.

**Problem**:
- You need to control access to an object
- You want to add functionality when accessing an object
- Lazy initialization, access control, logging, caching

**Solution**: Create a proxy with the same interface as the real object, controlling access to it.

**Types**:
- **Virtual Proxy**: Lazy initialization
- **Protection Proxy**: Access control
- **Remote Proxy**: Represents object in different address space
- **Caching Proxy**: Stores results

**Python Implementation**: `design_patterns/proxy_pattern.ipynb`

**Real-World Examples**:
- ORM lazy loading
- Remote service proxies
- Image loading (placeholder until loaded)
- Access control wrappers
- Logging/monitoring wrappers

**Benefits**:
- ✅ Control object access
- ✅ Lazy initialization
- ✅ Additional functionality without changing the object

**Trade-offs**:
- ❌ Increased complexity
- ❌ Delayed response from service

**Related Patterns**: Adapter, Decorator, Facade

---

## Behavioral Patterns

### 13. Chain of Responsibility Pattern

**Intent**: Avoid coupling the sender of a request to its receiver by giving more than one object a chance to handle the request. Chain the receiving objects and pass the request along the chain until an object handles it.

**Problem**:
- More than one object can handle a request
- You don't want to specify handler explicitly
- Set of handlers should be specified dynamically

**Solution**: Create a chain of handler objects. Each handler decides whether to process the request or pass it to the next handler.

**Python Implementation**: `design_patterns/chain_of_responsibility_pattern.ipynb`

**Real-World Examples**:
- Middleware in web frameworks
- Event bubbling in UI systems
- Logging frameworks (different log levels)
- Exception handling
- Request processing pipelines

**Benefits**:
- ✅ Reduced coupling
- ✅ Added flexibility in assigning responsibilities
- ✅ Can dynamically add/remove handlers

**Trade-offs**:
- ❌ Receipt not guaranteed
- ❌ Can be hard to debug

**Related Patterns**: Composite, Command, Mediator

---

### 14. Command Pattern

**Intent**: Encapsulate a request as an object, thereby letting you parameterize clients with different requests, queue or log requests, and support undoable operations.

**Problem**:
- You need to parameterize objects with operations
- You need to queue operations, schedule execution, or execute remotely
- You need to support undo/redo

**Solution**: Encapsulate request as an object with `execute()` method. This object can be passed, stored, and executed later.

**Python Implementation**: `design_patterns/command_pattern.ipynb`

**Real-World Examples**:
- GUI buttons and menu items
- Macro recording
- Transaction systems
- Job queues
- Undo/redo functionality
- Remote control systems

**Benefits**:
- ✅ Decouples sender from receiver
- ✅ Can assemble commands into composite commands
- ✅ Easy to add new commands
- ✅ Supports undo/redo

**Trade-offs**:
- ❌ Increased number of classes

**Related Patterns**: Composite, Memento, Prototype

---

### 15. Iterator Pattern

**Intent**: Provide a way to access elements of an aggregate object sequentially without exposing its underlying representation.

**Problem**:
- You need to traverse a collection without exposing its structure
- You want to support multiple traversals simultaneously
- You want a uniform interface for traversing different structures

**Solution**: Create an iterator object that knows how to traverse the collection.

**Python Implementation**: `design_patterns/iterator_pattern.ipynb`

**Real-World Examples**:
- Collection traversal
- Database result sets
- File system traversal
- Custom data structures (trees, graphs)

**Benefits**:
- ✅ Single Responsibility Principle (traversal logic separate)
- ✅ Open/Closed Principle (new iterators without changing collection)
- ✅ Can iterate over same collection in parallel

**Trade-offs**:
- ❌ Overkill for simple collections
- ❌ Can be less efficient than direct access

**Python-Specific**: Python has built-in iterator protocol (`__iter__` and `__next__`). Generators simplify iterator creation.

**Related Patterns**: Composite, Factory Method, Memento

---

### 16. Mediator Pattern

**Intent**: Define an object that encapsulates how a set of objects interact. Mediator promotes loose coupling by keeping objects from referring to each other explicitly.

**Problem**:
- Objects communicate in complex ways
- Reusing objects is difficult due to many dependencies
- Behavior distributed across multiple classes should be customizable

**Solution**: Create a mediator object that coordinates interactions between objects.

**Python Implementation**: `design_patterns/mediator_pattern.ipynb`

**Real-World Examples**:
- Chat rooms (users communicate through room)
- Air traffic control
- Dialog boxes (components coordinated)
- MVC controller
- Message brokers

**Benefits**:
- ✅ Reduces coupling between components
- ✅ Centralizes control
- ✅ Simplifies object protocols

**Trade-offs**:
- ❌ Mediator can become too complex (god object)

**Related Patterns**: Facade, Observer, Command

---

### 17. Memento Pattern

**Intent**: Without violating encapsulation, capture and externalize an object's internal state so that the object can be restored to this state later.

**Problem**:
- You need to save and restore object state
- Direct access to state would violate encapsulation
- You need undo/redo functionality

**Solution**: Create a memento object that stores the state. The originator creates and restores from mementos. A caretaker stores mementos.

**Python Implementation**: `design_patterns/memento_pattern.ipynb`

**Real-World Examples**:
- Undo/redo in editors
- Game save states
- Transaction rollback
- Database snapshots
- Version control

**Benefits**:
- ✅ Preserves encapsulation
- ✅ Simplifies originator class

**Trade-offs**:
- ❌ Can be expensive if state is large
- ❌ Caretakers must track memento lifecycle

**Related Patterns**: Command, Iterator

---

### 18. Observer Pattern

**Intent**: Define a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.

**Problem**:
- Changes in one object require changes in others
- You don't know how many objects need to be changed
- Object should notify others without assuming who they are

**Solution**: Subject maintains list of observers. When state changes, subject notifies all observers.

**Python Implementation**: `design_patterns/observer_pattern.ipynb`

**Real-World Examples**:
- Event handling systems
- MVC architecture (model notifies views)
- Pub/sub messaging
- Reactive programming
- Spreadsheet cells
- Social media notifications

**Benefits**:
- ✅ Loose coupling between subject and observers
- ✅ Open/Closed Principle
- ✅ Establish relationships at runtime

**Trade-offs**:
- ❌ Observers notified in random order
- ❌ Can cause memory leaks if not unsubscribed

**Python-Specific**: Can use `@property` setters to trigger notifications.

**Related Patterns**: Mediator, Singleton, Command

---

### 19. State Pattern

**Intent**: Allow an object to alter its behavior when its internal state changes. The object will appear to change its class.

**Problem**:
- Object behavior depends on its state
- Operations have large conditional statements on state
- State-specific behavior should be defined independently

**Solution**: Create separate classes for each state. Context delegates state-specific behavior to current state object.

**Python Implementation**: `design_patterns/state_pattern.ipynb`

**Real-World Examples**:
- TCP connection states
- Document workflow (draft, review, published)
- Game character states
- Media player states (playing, paused, stopped)
- Order processing states

**Benefits**:
- ✅ Eliminates large conditionals
- ✅ Each state has its own class (SRP)
- ✅ Easy to add new states

**Trade-offs**:
- ❌ Increases number of classes
- ❌ Overkill for simple state machines

**Related Patterns**: Flyweight, Strategy, Singleton

---

### 20. Strategy Pattern

**Intent**: Define a family of algorithms, encapsulate each one, and make them interchangeable. Strategy lets the algorithm vary independently from clients that use it.

**Problem**:
- You need different variants of an algorithm
- You want to avoid exposing algorithm implementation details
- Class has massive conditionals for different behaviors

**Solution**: Extract algorithms into separate strategy classes. Context uses a strategy object.

**Python Implementation**: `design_patterns/strategy_pattern.ipynb`

**Real-World Examples**:
- Sorting algorithms
- Payment methods
- Compression algorithms
- Validation strategies
- Route calculation
- Authentication methods

**Benefits**:
- ✅ Swap algorithms at runtime
- ✅ Isolate algorithm implementation
- ✅ Open/Closed Principle

**Trade-offs**:
- ❌ Clients must be aware of different strategies
- ❌ Increased number of objects

**Python-Specific**: Can use functions/lambdas as strategies (first-class functions).

**Related Patterns**: State, Template Method, Flyweight

---

### 21. Template Method Pattern

**Intent**: Define the skeleton of an algorithm in an operation, deferring some steps to subclasses. Template Method lets subclasses redefine certain steps of an algorithm without changing the algorithm's structure.

**Problem**:
- Two or more components have significant similarities but different details
- You want to avoid code duplication
- You want to control extension points

**Solution**: Define algorithm skeleton in base class. Subclasses override specific steps.

**Python Implementation**: `design_patterns/template_method_pattern.ipynb`

**Real-World Examples**:
- Framework extension points
- Data processing pipelines
- Testing frameworks (setUp/tearDown)
- Document generators
- Game AI

**Benefits**:
- ✅ Code reuse
- ✅ Control over extension points
- ✅ Inverse control structure

**Trade-offs**:
- ❌ Limits flexibility
- ❌ Can violate Liskov Substitution Principle

**Related Patterns**: Strategy, Factory Method

---

### 22. Visitor Pattern

**Intent**: Represent an operation to be performed on elements of an object structure. Visitor lets you define a new operation without changing the classes of the elements on which it operates.

**Problem**:
- You need to perform operations across a heterogeneous collection
- Operations should be separate from the objects
- Adding new operations should be easy

**Solution**: Define visitor interface with visit methods for each element type. Elements accept visitors.

**Python Implementation**: `design_patterns/visitor_pattern.ipynb`

**Real-World Examples**:
- Compiler AST traversal
- Tax calculation on shopping items
- Export to different formats
- Code analysis tools
- Serialization

**Benefits**:
- ✅ Open/Closed Principle (new operations without changing elements)
- ✅ Single Responsibility Principle (related operations grouped)
- ✅ Accumulate state while traversing

**Trade-offs**:
- ❌ Adding new element types requires changing all visitors
- ❌ May break encapsulation

**Related Patterns**: Composite, Interpreter, Iterator

---

### 23. Interpreter Pattern

**Intent**: Given a language, define a representation for its grammar along with an interpreter that uses the representation to interpret sentences in the language.

**Problem**:
- You need to interpret a simple language or expression
- Grammar is simple and stability is important
- Efficiency is not critical

**Solution**: Define grammar as class hierarchy. Each rule is a class with interpret method.

**Python Implementation**: `design_patterns/interpreter_pattern.ipynb`

**Real-World Examples**:
- SQL parsing
- Regular expressions
- Mathematical expressions
- Configuration file parsing
- Domain-specific languages

**Benefits**:
- ✅ Easy to change and extend grammar
- ✅ Easy to implement grammar

**Trade-offs**:
- ❌ Complex grammars are hard to maintain
- ❌ Not efficient for complex grammars

**Related Patterns**: Composite, Flyweight, Iterator, Visitor

---

## Pattern Relationships

### Pattern Combinations

**Common Combinations**:
- **Abstract Factory + Singleton**: Single instance of factory
- **Builder + Composite**: Build complex tree structures
- **Iterator + Visitor**: Traverse and operate on structures
- **Command + Memento**: Undo/redo operations
- **Observer + Mediator**: Event-driven architectures
- **Strategy + Factory**: Select algorithm at runtime
- **Decorator + Factory**: Create decorated objects

### Pattern Categories by Use Case

**Object Creation**: Factory Method, Abstract Factory, Builder, Prototype, Singleton

**Interface Adaptation**: Adapter, Bridge, Facade, Proxy

**Behavior Addition**: Decorator, Chain of Responsibility, Command

**Algorithm Selection**: Strategy, Template Method, State

**Complex Structures**: Composite, Flyweight

**Communication**: Observer, Mediator, Command

**Traversal**: Iterator, Visitor

**State Management**: Memento, State

---

## Choosing the Right Pattern

### Decision Tree

**Need to create objects?**
- One instance only? → **Singleton**
- Clone existing? → **Prototype**
- Complex construction? → **Builder**
- Delegate to subclasses? → **Factory Method**
- Families of objects? → **Abstract Factory**

**Need to structure objects?**
- Incompatible interfaces? → **Adapter**
- Simplify complex system? → **Facade**
- Tree structure? → **Composite**
- Add behavior dynamically? → **Decorator**
- Control access? → **Proxy**
- Separate abstraction/implementation? → **Bridge**
- Share many similar objects? → **Flyweight**

**Need to manage behavior?**
- Encapsulate algorithm? → **Strategy**
- State-dependent behavior? → **State**
- Algorithm skeleton in base class? → **Template Method**
- Process request through chain? → **Chain of Responsibility**
- Encapsulate requests? → **Command**
- Notify multiple objects? → **Observer**
- Coordinate object interactions? → **Mediator**
- Traverse collection? → **Iterator**
- Operations on object structure? → **Visitor**
- Save/restore state? → **Memento**
- Interpret language? → **Interpreter**

### Anti-Patterns to Avoid

- **Over-engineering**: Don't use patterns just to use them
- **Pattern Obsession**: Not every problem needs a pattern
- **Premature Optimization**: Start simple, refactor to patterns when needed
- **Forcing Patterns**: If it doesn't fit naturally, it's probably wrong

---

## Python-Specific Considerations

### When Python Features Replace Patterns

- **Singleton**: Module-level variables, metaclasses
- **Iterator**: Built-in protocols, generators
- **Strategy**: First-class functions, lambdas
- **Template Method**: Dependency injection, composition
- **Factory**: `__init__` flexibility, default arguments

### Python Advantages for Patterns

- **Duck Typing**: Less need for formal interfaces
- **First-Class Functions**: Simpler strategy implementations
- **Multiple Inheritance**: Flexible adapter/mixin patterns
- **Decorators**: Built-in decorator pattern support
- **Generators**: Simplified iterator pattern
- **Context Managers**: Resource management patterns

---

## Learning Path

### Week 1: Creational Patterns
1. ✅ Factory Method
2. ✅ Abstract Factory
3. ✅ Builder
4. Singleton
5. Prototype

### Week 2: Structural Patterns (Part 1)
6. Adapter
7. Bridge
8. Composite
9. Decorator

### Week 3: Structural Patterns (Part 2) + Behavioral (Part 1)
10. Facade
11. Flyweight
12. Proxy
13. Chain of Responsibility
14. Command

### Week 4: Behavioral Patterns (Part 2)
15. Iterator
16. Mediator
17. Memento
18. Observer
19. State

### Week 5: Behavioral Patterns (Part 3) + Integration
20. Strategy
21. Template Method
22. Visitor
23. Interpreter
24. **Integration Project**: Combine multiple patterns

---

## Additional Resources

### Books
- **Design Patterns: Elements of Reusable Object-Oriented Software** (Gang of Four)
- **Head First Design Patterns** (Freeman & Freeman)
- **Python Design Patterns** (Brandon Rhodes)
- **Fluent Python** (Luciano Ramalho)

### Online Resources
- [Refactoring Guru](https://refactoring.guru/design-patterns)
- [SourceMaking](https://sourcemaking.com/design_patterns)
- [Python Patterns Guide](https://python-patterns.guide/)

### Practice
- Refactor existing code using patterns
- Study open-source projects (Django, Flask, requests)
- Build small projects combining multiple patterns
- Code reviews focusing on design decisions

---

## Notes

- Each pattern has a dedicated Jupyter notebook in `design_patterns/` directory
- Notebooks include complete implementations, examples, and exercises
- Progress tracking available in `learning-plans/design-patterns.md`
- Real-world examples help understand practical applications
- Focus on understanding the problem each pattern solves, not memorizing implementations
