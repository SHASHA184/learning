# Django Learning Module

## Overview
Django is a high-level Python web framework that encourages rapid development and clean, pragmatic design. This module covers Django fundamentals needed for backend development roles.

## Learning Objectives
- Understand Django's MTV (Model-Template-View) architecture
- Master Django ORM for database operations
- Build RESTful APIs with Django REST Framework
- Debug effectively using Django shell and debugging tools
- Handle authentication and authorization
- Write tests for Django applications

## Prerequisites
- Python fundamentals
- Basic understanding of HTTP and web concepts
- SQL basics (see [databases.md](databases.md))

## Module Structure

### 1. Django Basics
**Time estimate**: 3-4 days

Topics:
- Django project structure
- Settings and configuration
- URL routing and views
- MTV pattern vs MVC
- Django admin interface

Practice:
- `django/01_basics/` - Basic Django app setup
- `django/01_basics/blog/` - Simple blog application

### 2. Django Models & ORM
**Time estimate**: 4-5 days

Topics:
- Model definition and field types
- Relationships (ForeignKey, ManyToMany, OneToOne)
- QuerySets and database queries
- Migrations
- Model managers and custom querysets
- Database indexing and optimization

Practice:
- `django/02_models/` - Model design exercises
- `django/02_models/ecommerce/` - E-commerce data model

Key concepts:
```python
# Model definition
class Product(models.Model):
    name = models.CharField(max_length=200)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=['category', 'price'])]
        ordering = ['-created_at']

    def __str__(self):
        return self.name

# QuerySet usage
products = Product.objects.filter(
    category__name='Electronics',
    price__lt=1000
).select_related('category').prefetch_related('reviews')
```

### 3. Django Shell & Debugging
**Time estimate**: 2-3 days

Topics:
- Django shell basics
- shell_plus from django-extensions
- IPython integration
- Debugging with pdb and django-debug-toolbar
- Logging configuration
- Query analysis with django-debug-toolbar

Practice:
- `django/03_debugging/` - Debugging exercises
- Common debugging scenarios

Commands:
```bash
# Django shell
python manage.py shell

# Shell plus (with django-extensions)
python manage.py shell_plus

# Run specific queries for testing
python manage.py shell < test_queries.py
```

### 4. Django REST Framework (DRF)
**Time estimate**: 5-6 days

Topics:
- Serializers (ModelSerializer, custom serializers)
- Views and ViewSets
- Routers and URL configuration
- Authentication (Token, JWT, Session)
- Permissions and throttling
- Pagination
- Filtering and search

Practice:
- `django/04_rest_api/` - REST API examples
- `django/04_rest_api/task_manager/` - Task management API

Key concepts:
```python
# Serializer
class ProductSerializer(serializers.ModelSerializer):
    category_name = serializers.CharField(source='category.name', read_only=True)

    class Meta:
        model = Product
        fields = ['id', 'name', 'price', 'category', 'category_name']

    def validate_price(self, value):
        if value < 0:
            raise serializers.ValidationError("Price cannot be negative")
        return value

# ViewSet
class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.select_related('category')
    serializer_class = ProductSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['name', 'category__name']
    ordering_fields = ['price', 'created_at']
```

### 5. Authentication & Authorization
**Time estimate**: 3-4 days

Topics:
- User model and authentication backends
- Session vs Token authentication
- Django permissions system
- Custom permissions
- OAuth2 integration
- JWT tokens

Practice:
- `django/05_auth/` - Authentication examples
- Multi-tenant application example

### 6. Testing
**Time estimate**: 3-4 days

Topics:
- Django TestCase and test client
- Fixtures and factories
- Mocking external dependencies
- Testing API endpoints
- Coverage analysis
- Continuous integration setup

Practice:
- `django/06_testing/` - Test examples
- Test-driven development exercises

### 7. Performance & Optimization
**Time estimate**: 3-4 days

Topics:
- Query optimization (select_related, prefetch_related)
- Database indexes
- Caching strategies (Redis, Memcached)
- N+1 query problem
- Connection pooling
- Async views (Django 4.1+)

Practice:
- `django/07_performance/` - Optimization exercises
- Performance benchmarking scripts

### 8. Celery & Background Tasks
**Time estimate**: 3-4 days

Topics:
- Celery basics and configuration
- Task definition and execution
- Periodic tasks with Celery Beat
- Task monitoring with Flower
- Error handling and retries
- Task chaining and groups

Practice:
- `django/08_celery/` - Celery integration
- Email sending, report generation tasks

Example:
```python
# tasks.py
from celery import shared_task
from django.core.mail import send_mail

@shared_task(bind=True, max_retries=3)
def send_notification_email(self, user_id, message):
    try:
        user = User.objects.get(id=user_id)
        send_mail(
            'Notification',
            message,
            'noreply@example.com',
            [user.email],
            fail_silently=False,
        )
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60)
```

## Practical Projects

### Project 1: Blog API
**Time**: 1 week
**Features**:
- User registration and authentication
- CRUD operations for posts and comments
- Tag system
- Search and filtering
- Pagination
- Rate limiting

### Project 2: Task Management System
**Time**: 1.5 weeks
**Features**:
- Multi-user support
- Project and task organization
- Task assignment and status tracking
- File attachments
- Email notifications (Celery)
- REST API with full documentation

### Project 3: E-commerce Backend
**Time**: 2 weeks
**Features**:
- Product catalog with categories
- Shopping cart
- Order processing
- Payment integration (Stripe)
- Inventory management
- Admin dashboard
- Background tasks for order processing

## Interview Preparation

### Common Django Questions
1. Explain Django's MTV architecture
2. What's the difference between select_related and prefetch_related?
3. How do you handle database migrations?
4. Explain Django's request-response cycle
5. What are Django signals? When would you use them?
6. How do you optimize slow queries in Django?
7. Explain CSRF protection in Django
8. What's the difference between Model.save() and QuerySet.update()?
9. How do you implement caching in Django?
10. Explain Django middleware and write a custom middleware

### Debugging Scenarios
- Finding N+1 query problems
- Debugging slow API endpoints
- Investigating memory leaks
- Analyzing failed background tasks
- Database deadlock resolution

## Resources

### Official Documentation
- [Django Documentation](https://docs.djangoproject.com/)
- [Django REST Framework](https://www.django-rest-framework.org/)
- [Celery Documentation](https://docs.celeryproject.org/)

### Books
- "Two Scoops of Django" by Daniel and Audrey Roy Greenfeld
- "Django for APIs" by William S. Vincent
- "Django for Professionals" by William S. Vincent

### Tools
- **django-extensions**: Enhanced management commands
- **django-debug-toolbar**: Development debugging
- **django-silk**: Request profiling
- **factory-boy**: Test fixtures
- **pytest-django**: Better testing

## Next Steps
1. Complete Django basics (Projects 1-2)
2. Build a REST API project
3. Integrate with PostgreSQL (see [databases.md](databases.md))
4. Containerize with Docker (see [docker.md](docker.md))
5. Build full-stack project combining all skills

## Progress Tracking
- [ ] Django basics completed
- [ ] Models & ORM mastered
- [ ] Django shell proficiency
- [ ] REST API built
- [ ] Authentication implemented
- [ ] Tests written
- [ ] Performance optimized
- [ ] Celery integrated
- [ ] Blog API project completed
- [ ] Task Manager project completed
- [ ] E-commerce project completed
