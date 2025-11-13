# Task Manager API - Integration Project

## Overview
This is a comprehensive integration project that combines Django REST Framework, PostgreSQL, Docker, and Celery. It demonstrates all the key skills required for the Namecheap Junior Python role.

## Project Goals
Build a production-ready task management API with the following features:
- User authentication and authorization
- CRUD operations for projects and tasks
- Task assignment and status tracking
- Background email notifications (Celery)
- File attachments
- API documentation
- Full test coverage
- Containerized with Docker

## Technology Stack
- **Backend**: Django 4.2+ with Django REST Framework
- **Database**: PostgreSQL 15
- **Cache/Broker**: Redis 7
- **Task Queue**: Celery with Celery Beat
- **Containerization**: Docker & Docker Compose
- **Web Server**: Gunicorn + Nginx (production)
- **Testing**: pytest-django
- **API Docs**: drf-spectacular (OpenAPI/Swagger)

## Project Structure
```
task_manager/
├── docker-compose.yml
├── docker-compose.override.yml  # Dev config
├── Dockerfile
├── .dockerignore
├── .env.example
├── .gitignore
├── requirements.txt
├── pytest.ini
├── manage.py
├── config/
│   ├── __init__.py
│   ├── settings/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── development.py
│   │   └── production.py
│   ├── urls.py
│   ├── wsgi.py
│   ├── asgi.py
│   └── celery.py
├── apps/
│   ├── users/
│   │   ├── models.py
│   │   ├── serializers.py
│   │   ├── views.py
│   │   ├── urls.py
│   │   └── tests/
│   ├── projects/
│   │   ├── models.py
│   │   ├── serializers.py
│   │   ├── views.py
│   │   ├── permissions.py
│   │   ├── urls.py
│   │   └── tests/
│   └── tasks/
│       ├── models.py
│       ├── serializers.py
│       ├── views.py
│       ├── tasks.py  # Celery tasks
│       ├── urls.py
│       └── tests/
├── static/
├── media/
└── docs/
    ├── setup.md
    ├── api.md
    └── deployment.md
```

## Features & Implementation

### 1. User Management
**Models**:
- Custom User model extending AbstractUser
- User profiles with avatar
- Email verification

**Endpoints**:
```
POST   /api/auth/register/          - User registration
POST   /api/auth/login/             - Login (JWT)
POST   /api/auth/logout/            - Logout
POST   /api/auth/refresh/           - Refresh token
GET    /api/auth/profile/           - Get current user
PATCH  /api/auth/profile/           - Update profile
POST   /api/auth/change-password/   - Change password
```

**Implementation**:
```python
# users/models.py
from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    email = models.EmailField(unique=True)
    avatar = models.ImageField(upload_to='avatars/', null=True, blank=True)
    is_verified = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']

    def __str__(self):
        return self.email

# users/serializers.py
from rest_framework import serializers
from django.contrib.auth import get_user_model

User = get_user_model()

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'avatar', 'created_at']
        read_only_fields = ['id', 'created_at']

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=8)
    password_confirm = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'password', 'password_confirm']

    def validate(self, data):
        if data['password'] != data['password_confirm']:
            raise serializers.ValidationError("Passwords don't match")
        return data

    def create(self, validated_data):
        validated_data.pop('password_confirm')
        user = User.objects.create_user(**validated_data)
        return user
```

### 2. Projects
**Models**:
- Project with name, description, owner
- Members (many-to-many with User)
- Created/updated timestamps

**Endpoints**:
```
GET    /api/projects/              - List user's projects
POST   /api/projects/              - Create project
GET    /api/projects/{id}/         - Get project details
PUT    /api/projects/{id}/         - Update project
DELETE /api/projects/{id}/         - Delete project
POST   /api/projects/{id}/members/ - Add member
DELETE /api/projects/{id}/members/{user_id}/ - Remove member
```

**Implementation**:
```python
# projects/models.py
from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()

class Project(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='owned_projects')
    members = models.ManyToManyField(User, related_name='projects')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['owner', 'created_at']),
        ]

    def __str__(self):
        return self.name

# projects/serializers.py
from rest_framework import serializers
from .models import Project
from apps.users.serializers import UserSerializer

class ProjectSerializer(serializers.ModelSerializer):
    owner = UserSerializer(read_only=True)
    members = UserSerializer(many=True, read_only=True)
    tasks_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = Project
        fields = ['id', 'name', 'description', 'owner', 'members',
                  'tasks_count', 'created_at', 'updated_at']
        read_only_fields = ['id', 'created_at', 'updated_at']

# projects/views.py
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.db.models import Count
from .models import Project
from .serializers import ProjectSerializer
from .permissions import IsOwnerOrMember

class ProjectViewSet(viewsets.ModelViewSet):
    serializer_class = ProjectSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Project.objects.filter(
            models.Q(owner=self.request.user) | models.Q(members=self.request.user)
        ).annotate(
            tasks_count=Count('tasks')
        ).select_related('owner').prefetch_related('members').distinct()

    def perform_create(self, serializer):
        project = serializer.save(owner=self.request.user)
        project.members.add(self.request.user)

    @action(detail=True, methods=['post'])
    def add_member(self, request, pk=None):
        project = self.get_object()
        user_id = request.data.get('user_id')

        try:
            user = User.objects.get(id=user_id)
            project.members.add(user)
            return Response({'status': 'member added'})
        except User.DoesNotExist:
            return Response({'error': 'User not found'},
                          status=status.HTTP_404_NOT_FOUND)
```

### 3. Tasks
**Models**:
- Task with title, description, status, priority
- Assigned to user
- Due date
- Attachments

**Endpoints**:
```
GET    /api/tasks/                    - List tasks (with filters)
POST   /api/tasks/                    - Create task
GET    /api/tasks/{id}/               - Get task details
PUT    /api/tasks/{id}/               - Update task
DELETE /api/tasks/{id}/               - Delete task
PATCH  /api/tasks/{id}/assign/        - Assign task
PATCH  /api/tasks/{id}/status/        - Update status
POST   /api/tasks/{id}/attachments/   - Add attachment
GET    /api/tasks/statistics/         - Get task statistics
```

**Implementation**:
```python
# tasks/models.py
from django.db import models
from apps.projects.models import Project
from django.contrib.auth import get_user_model

User = get_user_model()

class Task(models.Model):
    STATUS_CHOICES = [
        ('todo', 'To Do'),
        ('in_progress', 'In Progress'),
        ('review', 'In Review'),
        ('done', 'Done'),
    ]

    PRIORITY_CHOICES = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
    ]

    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='tasks')
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='todo')
    priority = models.CharField(max_length=20, choices=PRIORITY_CHOICES, default='medium')
    assigned_to = models.ForeignKey(User, on_delete=models.SET_NULL,
                                   null=True, blank=True, related_name='assigned_tasks')
    due_date = models.DateTimeField(null=True, blank=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='created_tasks')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['project', 'status']),
            models.Index(fields=['assigned_to', 'status']),
            models.Index(fields=['due_date']),
        ]

    def __str__(self):
        return self.title

class TaskAttachment(models.Model):
    task = models.ForeignKey(Task, on_delete=models.CASCADE, related_name='attachments')
    file = models.FileField(upload_to='task_attachments/')
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.task.title} - {self.file.name}"
```

### 4. Celery Integration
**Background Tasks**:
- Send email notifications on task assignment
- Send reminders for due tasks
- Generate reports
- Clean up old attachments

**Implementation**:
```python
# config/celery.py
import os
from celery import Celery
from celery.schedules import crontab

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.development')

app = Celery('task_manager')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

# Periodic tasks
app.conf.beat_schedule = {
    'send-due-task-reminders': {
        'task': 'apps.tasks.tasks.send_due_task_reminders',
        'schedule': crontab(hour=9, minute=0),  # Daily at 9 AM
    },
    'cleanup-old-attachments': {
        'task': 'apps.tasks.tasks.cleanup_old_attachments',
        'schedule': crontab(hour=2, minute=0, day_of_week=0),  # Weekly Sunday 2 AM
    },
}

# tasks/tasks.py
from celery import shared_task
from django.core.mail import send_mail
from django.utils import timezone
from datetime import timedelta
from .models import Task

@shared_task(bind=True, max_retries=3)
def send_task_assignment_email(self, task_id, user_email):
    """Send email when task is assigned"""
    try:
        task = Task.objects.get(id=task_id)
        send_mail(
            subject=f'Task Assigned: {task.title}',
            message=f'You have been assigned to task: {task.title}\n\n'
                   f'Project: {task.project.name}\n'
                   f'Due Date: {task.due_date}\n',
            from_email='noreply@taskmanager.com',
            recipient_list=[user_email],
            fail_silently=False,
        )
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60)

@shared_task
def send_due_task_reminders():
    """Send reminders for tasks due in next 24 hours"""
    tomorrow = timezone.now() + timedelta(days=1)
    due_tasks = Task.objects.filter(
        due_date__lte=tomorrow,
        due_date__gte=timezone.now(),
        status__in=['todo', 'in_progress']
    ).select_related('assigned_to', 'project')

    for task in due_tasks:
        if task.assigned_to:
            send_mail(
                subject=f'Reminder: Task "{task.title}" is due soon',
                message=f'Your task "{task.title}" in project "{task.project.name}" '
                       f'is due on {task.due_date}',
                from_email='noreply@taskmanager.com',
                recipient_list=[task.assigned_to.email],
                fail_silently=True,
            )

    return f'Sent reminders for {due_tasks.count()} tasks'
```

### 5. Docker Configuration
**docker-compose.yml**:
```yaml
version: '3.8'

services:
  db:
    image: postgres:15-alpine
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      POSTGRES_DB: task_manager
      POSTGRES_USER: taskuser
      POSTGRES_PASSWORD: ${DB_PASSWORD:-changeme}
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U taskuser -d task_manager"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  web:
    build: .
    command: gunicorn config.wsgi:application --bind 0.0.0.0:8000
    volumes:
      - .:/app
      - static_volume:/app/staticfiles
      - media_volume:/app/media
    ports:
      - "8000:8000"
    environment:
      DJANGO_SETTINGS_MODULE: config.settings.production
      DATABASE_URL: postgresql://taskuser:${DB_PASSWORD:-changeme}@db:5432/task_manager
      REDIS_URL: redis://redis:6379/0
      SECRET_KEY: ${SECRET_KEY}
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started

  celery:
    build: .
    command: celery -A config worker -l info
    volumes:
      - .:/app
    environment:
      DJANGO_SETTINGS_MODULE: config.settings.production
      DATABASE_URL: postgresql://taskuser:${DB_PASSWORD:-changeme}@db:5432/task_manager
      REDIS_URL: redis://redis:6379/0
    depends_on:
      - db
      - redis

  celery-beat:
    build: .
    command: celery -A config beat -l info
    volumes:
      - .:/app
    environment:
      DJANGO_SETTINGS_MODULE: config.settings.production
      DATABASE_URL: postgresql://taskuser:${DB_PASSWORD:-changeme}@db:5432/task_manager
      REDIS_URL: redis://redis:6379/0
    depends_on:
      - db
      - redis

volumes:
  postgres_data:
  redis_data:
  static_volume:
  media_volume:
```

**Dockerfile**:
```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    postgresql-client \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN python manage.py collectstatic --noinput

EXPOSE 8000

CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8000"]
```

## Setup Instructions

### Prerequisites
- Docker and Docker Compose installed
- Python 3.11+ (for local development without Docker)

### Quick Start with Docker
```bash
# Clone repository
git clone <repo-url>
cd task_manager

# Create .env file
cp .env.example .env
# Edit .env with your settings

# Build and start containers
docker-compose up --build

# Run migrations
docker-compose exec web python manage.py migrate

# Create superuser
docker-compose exec web python manage.py createsuperuser

# Access application
# API: http://localhost:8000/api/
# Admin: http://localhost:8000/admin/
# API Docs: http://localhost:8000/api/schema/swagger-ui/
```

### Local Development (without Docker)
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup PostgreSQL and Redis locally
# Create database: task_manager

# Run migrations
python manage.py migrate

# Start development server
python manage.py runserver

# In separate terminals:
celery -A config worker -l info
celery -A config beat -l info
```

## Testing
```bash
# Run all tests
docker-compose exec web pytest

# Run with coverage
docker-compose exec web pytest --cov=apps

# Run specific test file
docker-compose exec web pytest apps/tasks/tests/test_views.py

# Run with verbose output
docker-compose exec web pytest -v
```

## API Documentation
Access interactive API documentation at:
- Swagger UI: http://localhost:8000/api/schema/swagger-ui/
- ReDoc: http://localhost:8000/api/schema/redoc/

## Learning Outcomes
By completing this project, you will have demonstrated:
1. ✅ Django proficiency with complex models
2. ✅ Django REST Framework for API development
3. ✅ PostgreSQL database design and queries
4. ✅ Celery for background tasks
5. ✅ Docker containerization
6. ✅ Docker Compose for orchestration
7. ✅ Authentication and permissions
8. ✅ Testing with pytest
9. ✅ API documentation
10. ✅ Production-ready configuration

## Next Steps
1. Deploy to AWS/GCP using Docker
2. Add CI/CD pipeline with GitHub Actions
3. Implement real-time updates with WebSockets
4. Add monitoring with Prometheus/Grafana
5. Implement caching strategies with Redis
6. Add full-text search with PostgreSQL
7. Create frontend with React/Vue

## Resources
- [Project tutorial](../docs/tutorial.md)
- [API reference](../docs/api.md)
- [Deployment guide](../docs/deployment.md)
