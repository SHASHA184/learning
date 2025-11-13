# Database Learning Module (PostgreSQL Focus)

## Overview
Understanding relational databases is crucial for backend development. This module focuses on PostgreSQL with MySQL coverage, emphasizing practical skills needed for Django development.

## Learning Objectives
- Write efficient SQL queries
- Design normalized database schemas
- Understand indexes and query optimization
- Handle transactions and concurrency
- Use PostgreSQL-specific features
- Integrate databases with Django ORM

## Prerequisites
- Basic SQL knowledge (SELECT, INSERT, UPDATE, DELETE)
- Understanding of data types

## Module Structure

### 1. SQL Fundamentals
**Time estimate**: 3-4 days

Topics:
- SQL syntax and data types
- CRUD operations
- WHERE, ORDER BY, LIMIT
- Aggregate functions (COUNT, SUM, AVG, etc.)
- GROUP BY and HAVING
- DISTINCT and NULL handling

Practice:
- `web-backend/databases/01_sql_basics/` - SQL exercise scripts
- `web-backend/databases/01_sql_basics/exercises.sql` - 50+ practice queries

Example exercises:
```sql
-- Basic queries
SELECT name, price FROM products WHERE price > 100 ORDER BY price DESC;

-- Aggregation
SELECT category, COUNT(*) as product_count, AVG(price) as avg_price
FROM products
GROUP BY category
HAVING COUNT(*) > 5;

-- Subqueries
SELECT name FROM products
WHERE price > (SELECT AVG(price) FROM products);
```

### 2. Joins and Relationships
**Time estimate**: 3-4 days

Topics:
- INNER JOIN
- LEFT/RIGHT/FULL OUTER JOIN
- CROSS JOIN
- Self-joins
- Multiple table joins
- Join optimization

Practice:
- `web-backend/databases/02_joins/` - Join exercises
- Real-world scenarios (e-commerce, social network)

Key concepts:
```sql
-- INNER JOIN: Get orders with customer details
SELECT o.id, o.order_date, c.name, c.email
FROM orders o
INNER JOIN customers c ON o.customer_id = c.id;

-- LEFT JOIN: Get all products, including those without orders
SELECT p.name, COUNT(oi.id) as times_ordered
FROM products p
LEFT JOIN order_items oi ON p.id = oi.product_id
GROUP BY p.id, p.name;

-- Multiple joins
SELECT
    o.id as order_id,
    c.name as customer_name,
    p.name as product_name,
    oi.quantity,
    oi.price
FROM orders o
INNER JOIN customers c ON o.customer_id = c.id
INNER JOIN order_items oi ON o.id = oi.order_id
INNER JOIN products p ON oi.product_id = p.id;
```

### 3. Database Design & Normalization
**Time estimate**: 3-4 days

Topics:
- Entity-Relationship diagrams
- Normal forms (1NF, 2NF, 3NF, BCNF)
- Primary and foreign keys
- Constraints (UNIQUE, CHECK, NOT NULL)
- Referential integrity
- When to denormalize

Practice:
- `web-backend/databases/03_design/` - Schema design exercises
- Design schemas for: blog, e-commerce, social network, booking system

Example schema:
```sql
-- E-commerce schema
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    price DECIMAL(10, 2) NOT NULL CHECK (price >= 0),
    stock_quantity INTEGER NOT NULL DEFAULT 0,
    category_id INTEGER REFERENCES categories(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    status VARCHAR(20) DEFAULT 'pending',
    total_amount DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_status CHECK (status IN ('pending', 'processing', 'shipped', 'delivered', 'cancelled'))
);

CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    price DECIMAL(10, 2) NOT NULL,
    UNIQUE(order_id, product_id)
);
```

### 4. Indexes & Query Optimization
**Time estimate**: 4-5 days

Topics:
- Index types (B-tree, Hash, GiST, GIN)
- When to create indexes
- Composite indexes
- Covering indexes
- EXPLAIN and EXPLAIN ANALYZE
- Query planning
- Index maintenance

Practice:
- `web-backend/databases/04_optimization/` - Optimization exercises
- Performance benchmarking scripts

Key concepts:
```sql
-- Create indexes
CREATE INDEX idx_products_category ON products(category_id);
CREATE INDEX idx_products_price ON products(price);
CREATE INDEX idx_products_category_price ON products(category_id, price);

-- Analyze query performance
EXPLAIN ANALYZE
SELECT p.name, p.price
FROM products p
INNER JOIN categories c ON p.category_id = c.id
WHERE c.name = 'Electronics' AND p.price < 1000;

-- Full-text search index (PostgreSQL)
CREATE INDEX idx_products_search ON products
USING GIN (to_tsvector('english', name || ' ' || description));

SELECT name FROM products
WHERE to_tsvector('english', name || ' ' || description)
    @@ to_tsquery('english', 'laptop & gaming');
```

### 5. Transactions & Concurrency
**Time estimate**: 3-4 days

Topics:
- ACID properties
- Transaction isolation levels
- BEGIN, COMMIT, ROLLBACK
- SAVEPOINT
- Deadlocks and how to handle them
- Optimistic vs pessimistic locking
- Row-level locking (FOR UPDATE)

Practice:
- `web-backend/databases/05_transactions/` - Transaction examples
- Concurrency problem simulations

Example:
```sql
-- Basic transaction
BEGIN;
    UPDATE accounts SET balance = balance - 100 WHERE id = 1;
    UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;

-- Transaction with error handling
BEGIN;
    INSERT INTO orders (customer_id, total_amount)
    VALUES (1, 150.00) RETURNING id;

    -- If this fails, entire transaction rolls back
    INSERT INTO order_items (order_id, product_id, quantity, price)
    VALUES (currval('orders_id_seq'), 10, 2, 75.00);
COMMIT;

-- Pessimistic locking
BEGIN;
    SELECT * FROM products WHERE id = 5 FOR UPDATE;
    -- Other transactions wait here
    UPDATE products SET stock_quantity = stock_quantity - 1 WHERE id = 5;
COMMIT;
```

### 6. PostgreSQL-Specific Features
**Time estimate**: 3-4 days

Topics:
- JSON/JSONB data types
- Array data types
- Window functions
- CTEs (Common Table Expressions)
- Recursive queries
- Full-text search
- Views and materialized views
- Triggers and stored procedures

Practice:
- `web-backend/databases/06_postgresql/` - PostgreSQL-specific features

Examples:
```sql
-- JSONB operations
CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50),
    metadata JSONB
);

INSERT INTO events (event_type, metadata)
VALUES ('user_login', '{"ip": "192.168.1.1", "device": "mobile"}');

SELECT * FROM events
WHERE metadata->>'device' = 'mobile';

-- Window functions
SELECT
    product_name,
    price,
    category,
    AVG(price) OVER (PARTITION BY category) as avg_category_price,
    RANK() OVER (PARTITION BY category ORDER BY price DESC) as price_rank
FROM products;

-- CTE (Common Table Expression)
WITH monthly_sales AS (
    SELECT
        DATE_TRUNC('month', created_at) as month,
        SUM(total_amount) as total
    FROM orders
    GROUP BY DATE_TRUNC('month', created_at)
)
SELECT
    month,
    total,
    LAG(total) OVER (ORDER BY month) as previous_month,
    total - LAG(total) OVER (ORDER BY month) as growth
FROM monthly_sales;

-- Recursive CTE (hierarchical data)
WITH RECURSIVE category_tree AS (
    SELECT id, name, parent_id, 0 as level
    FROM categories
    WHERE parent_id IS NULL

    UNION ALL

    SELECT c.id, c.name, c.parent_id, ct.level + 1
    FROM categories c
    INNER JOIN category_tree ct ON c.parent_id = ct.id
)
SELECT * FROM category_tree;
```

### 7. Django ORM Integration
**Time estimate**: 4-5 days

Topics:
- Mapping models to database tables
- QuerySet methods and query translation
- Raw SQL when needed
- select_related and prefetch_related
- Annotations and aggregations
- Database functions (F, Q, Case, When)
- Custom database functions

Practice:
- `django/02_models/orm_exercises/` - ORM translation exercises
- Compare raw SQL with Django ORM equivalents

Examples:
```python
# Django ORM equivalents

# Simple filter
Product.objects.filter(price__gt=100).order_by('-price')

# Join with select_related
Order.objects.select_related('customer').filter(
    customer__email='user@example.com'
)

# Aggregation
from django.db.models import Count, Avg
Product.objects.values('category').annotate(
    count=Count('id'),
    avg_price=Avg('price')
).filter(count__gt=5)

# Complex query with annotations
from django.db.models import F, Q, Case, When, Value
Product.objects.annotate(
    discount_price=Case(
        When(category__name='Electronics', then=F('price') * 0.9),
        When(category__name='Books', then=F('price') * 0.8),
        default=F('price')
    )
).filter(Q(stock_quantity__gt=0) & Q(discount_price__lt=100))

# Raw SQL when needed
Product.objects.raw(
    'SELECT * FROM products WHERE price > %s ORDER BY price',
    [100]
)
```

### 8. Database Administration Basics
**Time estimate**: 2-3 days

Topics:
- Database creation and connection
- User management and permissions
- Backup and restore
- Connection pooling
- Database configuration for development vs production
- Monitoring and logging
- psql command-line tool

Practice:
- `web-backend/databases/08_admin/` - Admin scripts

Commands:
```bash
# PostgreSQL commands
# Create database
createdb myproject_dev

# Connect to database
psql -d myproject_dev

# Backup
pg_dump myproject_dev > backup.sql

# Restore
psql myproject_dev < backup.sql

# Grant permissions
psql -d myproject_dev -c "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO myuser;"
```

## Practical Projects

### Project 1: Library Management System
**Time**: 1 week
**Schema**:
- Books, Authors, Publishers
- Members, Loans
- Complex queries for overdue books, popular books, etc.

### Project 2: Analytics Database
**Time**: 1 week
**Focus**:
- Time-series data
- Aggregation queries
- Window functions
- Materialized views for reporting

### Project 3: Multi-tenant SaaS Database
**Time**: 1.5 weeks
**Focus**:
- Schema design for multi-tenancy
- Row-level security
- Performance optimization
- Data isolation strategies

## Interview Preparation

### Common Database Questions
1. Explain the difference between INNER JOIN and LEFT JOIN
2. What are database indexes and when should you use them?
3. Explain ACID properties
4. What's the N+1 query problem?
5. How do you optimize a slow query?
6. Explain database normalization
7. What are transactions and why are they important?
8. Difference between optimistic and pessimistic locking?
9. How do you handle database migrations in production?
10. Explain the difference between DELETE and TRUNCATE

### SQL Coding Exercises
- Find the second highest salary
- Get employees earning more than their manager
- Find duplicate records
- Calculate running totals
- Implement pagination efficiently

## Tools & Setup

### Installation
```bash
# PostgreSQL installation (Ubuntu/Debian)
sudo apt update
sudo apt install postgresql postgresql-contrib

# Start PostgreSQL
sudo systemctl start postgresql

# MySQL installation (alternative)
sudo apt install mysql-server
```

### Tools
- **psql**: PostgreSQL command-line client
- **pgAdmin**: GUI for PostgreSQL
- **DBeaver**: Universal database tool
- **DataGrip**: JetBrains database IDE

### Python libraries
```bash
pip install psycopg2-binary  # PostgreSQL adapter
pip install django  # Includes ORM
pip install SQLAlchemy  # Alternative ORM
```

## Resources

### Documentation
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [MySQL Documentation](https://dev.mysql.com/doc/)
- [Django Database API](https://docs.djangoproject.com/en/stable/topics/db/)

### Books
- "PostgreSQL: Up and Running" by Regina Obe & Leo Hsu
- "SQL Performance Explained" by Markus Winand
- "Database Design for Mere Mortals" by Michael J. Hernandez

### Online Resources
- [PostgreSQL Exercises](https://pgexercises.com/)
- [LeetCode Database Problems](https://leetcode.com/problemset/database/)
- [SQL Zoo](https://sqlzoo.net/)

## Progress Tracking
- [ ] SQL fundamentals mastered
- [ ] Joins and relationships understood
- [ ] Database design principles learned
- [ ] Query optimization skills developed
- [ ] Transactions and concurrency understood
- [ ] PostgreSQL-specific features learned
- [ ] Django ORM proficiency achieved
- [ ] Database administration basics completed
- [ ] Library management project completed
- [ ] Analytics database project completed
- [ ] Multi-tenant project completed
