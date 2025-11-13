# Namecheap Junior Python Developer Preparation Plan

**Target Role**: Junior Software Engineer (Python) in Pre-Production Team
**Company**: Namecheap
**Location**: Remote (Ukraine)
**Duration**: 8-10 weeks (intensive) or 12-16 weeks (balanced)
**Start Date**: [Your start date]

## Role Analysis

### Critical Requirements (Must Have)
1. ✅ **Django proficiency** - Framework, shell, debugging
2. ✅ **PostgreSQL/MySQL** - Relational database experience
3. ✅ **Linux/Unix** - Command line and OS knowledge
4. ✅ **Networking** - TCP/IP, HTTP protocols

### High-Value Skills (Strong Plus)
5. 🔶 **Celery** - Distributed task queue
6. 🔶 **Docker** - Containerized applications
7. 🔶 **Logging** - Kibana, OpenSearch
8. 🔶 **CI/CD** - Automated deployment principles
9. 🔶 **Cloud Services** - AWS, GCP

### Your Current Position
**Strengths**:
- ✅ Python fundamentals
- ✅ Design patterns knowledge
- ✅ Concurrency understanding
- ✅ Problem-solving skills

**Gaps to Address**:
- ❌ Django experience
- ❌ Database (PostgreSQL) practical skills
- ❌ Docker containerization
- ❌ Production debugging experience
- ❌ Linux system administration

## Learning Path

### Phase 1: Foundation Building (Weeks 1-3)

#### Week 1: Django Fundamentals
**Time commitment**: 15-20 hours/week

**Days 1-3: Django Basics**
- [ ] Set up Django development environment
- [ ] Study Django project structure
- [ ] Complete official Django tutorial (Polls app)
- [ ] Understand MTV architecture
- [ ] Practice URL routing and views
- [ ] Set up Django admin interface

**Resources**:
- [Django Official Tutorial](https://docs.djangoproject.com/en/stable/intro/tutorial01/)
- `docs/django.md` - Django Basics section

**Days 4-5: Django Models & ORM**
- [ ] Create models with various field types
- [ ] Practice relationships (ForeignKey, ManyToMany)
- [ ] Write QuerySets and filters
- [ ] Create and apply migrations
- [ ] Practice select_related and prefetch_related

**Practice Project**: Build a simple blog with posts, authors, categories
- Models: User, Post, Category, Comment
- Admin interface for content management
- Basic views for listing and detail pages

**Days 6-7: Django Shell & Debugging**
- [ ] Install django-extensions and django-debug-toolbar
- [ ] Practice shell_plus for interactive queries
- [ ] Learn to analyze SQL queries
- [ ] Debug views with pdb
- [ ] Set up logging

**Resources**:
- `docs/django.md` - Models & ORM, Debugging sections

#### Week 2: PostgreSQL & Databases
**Time commitment**: 15-20 hours/week

**Days 1-2: SQL Fundamentals**
- [ ] Install PostgreSQL locally
- [ ] Practice basic SQL (SELECT, INSERT, UPDATE, DELETE)
- [ ] Learn JOINs (INNER, LEFT, RIGHT)
- [ ] Study aggregate functions
- [ ] Practice GROUP BY and HAVING

**Resources**:
- `docs/databases.md` - SQL Fundamentals section
- [PostgreSQL Exercises](https://pgexercises.com/)

**Days 3-4: Database Design & Optimization**
- [ ] Design normalized schemas
- [ ] Create indexes
- [ ] Analyze query performance with EXPLAIN
- [ ] Practice transactions
- [ ] Study database constraints

**Practice**: Design and implement e-commerce database schema

**Days 5-7: Django + PostgreSQL Integration**
- [ ] Configure Django to use PostgreSQL
- [ ] Migrate blog project to PostgreSQL
- [ ] Optimize queries with select_related/prefetch_related
- [ ] Use database indexes in Django models
- [ ] Practice raw SQL queries in Django

**Resources**:
- `docs/databases.md` - Django ORM Integration section

#### Week 3: Django REST Framework
**Time commitment**: 15-20 hours/week

**Days 1-3: DRF Basics**
- [ ] Install Django REST Framework
- [ ] Create serializers for models
- [ ] Build APIViews and ViewSets
- [ ] Configure URL routing with routers
- [ ] Test APIs with Postman/curl

**Days 4-5: Authentication & Permissions**
- [ ] Implement token authentication
- [ ] Create custom permissions
- [ ] Add throttling and rate limiting
- [ ] Practice different authentication methods

**Days 6-7: Advanced DRF Features**
- [ ] Implement pagination
- [ ] Add filtering and search
- [ ] Create nested serializers
- [ ] Handle file uploads
- [ ] Generate API documentation

**Practice Project**: Blog REST API
- Convert blog to REST API
- Full CRUD operations
- User authentication
- Filtering and search
- API documentation with drf-spectacular

**Resources**:
- [DRF Official Tutorial](https://www.django-rest-framework.org/tutorial/quickstart/)
- `docs/django.md` - Django REST Framework section

### Phase 2: DevOps & Tools (Weeks 4-6)

#### Week 4: Docker & Containerization
**Time commitment**: 12-15 hours/week

**Days 1-2: Docker Basics**
- [ ] Install Docker and Docker Compose
- [ ] Learn basic Docker commands
- [ ] Pull and run containers
- [ ] Understand images vs containers
- [ ] Practice volume mounting

**Days 3-4: Dockerfile Creation**
- [ ] Write Dockerfile for Python apps
- [ ] Create multi-stage builds
- [ ] Use .dockerignore
- [ ] Build and tag images
- [ ] Push to Docker Hub

**Days 5-7: Docker Compose**
- [ ] Create docker-compose.yml
- [ ] Set up multi-container apps
- [ ] Configure networks and volumes
- [ ] Use environment variables
- [ ] Practice development workflow

**Practice Project**: Containerize Blog API
- Django app in container
- PostgreSQL in container
- Redis for caching
- Nginx as reverse proxy
- Docker Compose orchestration

**Resources**:
- `docs/docker.md` - Complete guide
- [Docker Official Tutorial](https://docs.docker.com/get-started/)

#### Week 5: Linux & Networking
**Time commitment**: 12-15 hours/week

**Days 1-3: Linux Command Line**
- [ ] Master basic commands (ls, cd, grep, find, etc.)
- [ ] Practice file permissions and ownership
- [ ] Learn process management (ps, top, kill)
- [ ] Study shell scripting basics
- [ ] Practice log file analysis

**Days 4-5: Networking Fundamentals**
- [ ] Study TCP/IP protocol stack
- [ ] Understand HTTP protocol
- [ ] Learn DNS basics
- [ ] Practice with curl and wget
- [ ] Use netstat, ping, traceroute

**Days 6-7: Network Debugging**
- [ ] Practice with tcpdump
- [ ] Debug API calls with curl
- [ ] Analyze network traffic
- [ ] Test SSL/TLS connections
- [ ] Monitor open ports

**Resources**:
- `docs/networking.md` - Complete guide
- Linux command practice online

#### Week 6: Celery & Background Tasks
**Time commitment**: 12-15 hours/week

**Days 1-3: Celery Basics**
- [ ] Understand task queue concepts
- [ ] Install Celery and Redis
- [ ] Create basic tasks
- [ ] Configure Celery with Django
- [ ] Monitor tasks with Flower

**Days 4-5: Advanced Celery**
- [ ] Implement task retries and error handling
- [ ] Use Celery Beat for periodic tasks
- [ ] Configure task routing
- [ ] Practice task chaining
- [ ] Optimize task performance

**Days 6-7: Production Patterns**
- [ ] Implement email sending tasks
- [ ] Create report generation tasks
- [ ] Handle file processing asynchronously
- [ ] Monitor task failures
- [ ] Set up task logging

**Practice**: Add Celery to Blog API
- Async email notifications
- Scheduled content publishing
- Report generation
- Image processing

**Resources**:
- [Celery Documentation](https://docs.celeryproject.org/)
- `docs/django.md` - Celery section

### Phase 3: Integration & Projects (Weeks 7-10)

#### Weeks 7-8: Task Manager Integration Project
**Time commitment**: 20-25 hours/week

Complete the comprehensive integration project that demonstrates all skills:

**Week 7: Core Implementation**
- [ ] Set up project structure
- [ ] Implement user authentication
- [ ] Create project and task models
- [ ] Build REST API with DRF
- [ ] Write comprehensive tests

**Week 8: Advanced Features & Deployment**
- [ ] Integrate Celery for notifications
- [ ] Add file attachments
- [ ] Containerize with Docker
- [ ] Set up docker-compose
- [ ] Deploy locally and test thoroughly

**Resources**:
- `django/10_integration_project/README.md` - Complete project guide

#### Weeks 9-10: Additional Projects & Polish
**Time commitment**: 15-20 hours/week

**Week 9: E-commerce Backend (Optional)**
Build a more complex system:
- [ ] Product catalog with categories
- [ ] Shopping cart functionality
- [ ] Order processing
- [ ] Inventory management
- [ ] Background order processing with Celery

**Week 10: Interview Preparation**
- [ ] Review all Django concepts
- [ ] Practice database optimization
- [ ] Review Docker commands
- [ ] Practice system design questions
- [ ] Mock interviews
- [ ] Prepare portfolio presentation

**Resources**:
- `learning-plans/interview-prep.md` - Updated with Django/backend questions

## Week-by-Week Breakdown

### Intensive Schedule (8 weeks, ~20 hours/week)

| Week | Focus | Hours | Key Deliverables |
|------|-------|-------|------------------|
| 1 | Django Fundamentals | 20 | Blog app with admin |
| 2 | PostgreSQL & Databases | 20 | E-commerce schema, optimized queries |
| 3 | Django REST Framework | 20 | Blog REST API with auth |
| 4 | Docker & Containerization | 15 | Containerized Blog API |
| 5 | Linux & Networking | 15 | Network debugging exercises |
| 6 | Celery & Background Tasks | 15 | Blog API with async tasks |
| 7 | Integration Project Part 1 | 25 | Task Manager core features |
| 8 | Integration Project Part 2 | 25 | Complete Task Manager with Docker |

**Total**: ~155 hours

### Balanced Schedule (12 weeks, ~12-15 hours/week)

| Week | Focus | Hours | Key Deliverables |
|------|-------|-------|------------------|
| 1-2 | Django Fundamentals | 15/wk | Blog app with admin |
| 3-4 | PostgreSQL & Databases | 15/wk | Database design & optimization |
| 5-6 | Django REST Framework | 15/wk | Blog REST API |
| 7-8 | Docker & Linux | 12/wk | Containerized applications |
| 9-10 | Celery & Networking | 12/wk | Async task processing |
| 11-12 | Integration Project | 20/wk | Complete Task Manager |

**Total**: ~180 hours

## Daily Practice Routine

### Morning (1-2 hours)
- Theory and documentation reading
- Video tutorials (if applicable)
- Note-taking

### Evening (2-3 hours)
- Hands-on coding practice
- Project work
- Problem-solving exercises

### Weekend (4-6 hours each day)
- Larger project work
- Review and consolidation
- Practice interviews

## Progress Tracking

### Week 1 Checklist
- [ ] Django development environment set up
- [ ] Completed Django official tutorial
- [ ] Built blog application
- [ ] Comfortable with Django admin
- [ ] Understand Django ORM basics

### Week 2 Checklist
- [ ] PostgreSQL installed and configured
- [ ] Comfortable with SQL queries
- [ ] Designed normalized database schema
- [ ] Django connected to PostgreSQL
- [ ] Can optimize database queries

### Week 3 Checklist
- [ ] DRF installed and configured
- [ ] Built complete REST API
- [ ] Implemented authentication
- [ ] API documentation created
- [ ] Tested with Postman/curl

### Week 4 Checklist
- [ ] Docker installed and working
- [ ] Written Dockerfiles
- [ ] Created docker-compose configurations
- [ ] Containerized Django application
- [ ] Understand container networking

### Week 5 Checklist
- [ ] Comfortable with Linux command line
- [ ] Understand TCP/IP and HTTP
- [ ] Can debug network issues
- [ ] Familiar with networking tools
- [ ] Can analyze HTTP requests/responses

### Week 6 Checklist
- [ ] Celery configured with Django
- [ ] Created background tasks
- [ ] Implemented periodic tasks
- [ ] Understand task queues
- [ ] Can monitor and debug tasks

### Weeks 7-8 Checklist
- [ ] Task Manager project completed
- [ ] All features implemented
- [ ] Tests written and passing
- [ ] Dockerized and deployed
- [ ] Documentation complete

### Weeks 9-10 Checklist
- [ ] Additional project completed (optional)
- [ ] All concepts reviewed
- [ ] Mock interviews practiced
- [ ] Portfolio ready
- [ ] Confident in technical skills

## Interview Preparation Checklist

### Technical Skills
- [ ] Can build Django REST API from scratch
- [ ] Comfortable with PostgreSQL queries and optimization
- [ ] Can containerize applications with Docker
- [ ] Understand Celery task queue
- [ ] Can debug production issues
- [ ] Familiar with Linux command line

### Soft Skills
- [ ] Can explain technical concepts clearly
- [ ] Prepared to discuss past projects
- [ ] Ready to ask thoughtful questions
- [ ] Comfortable with pair programming
- [ ] Can handle feedback positively

### Portfolio Projects
- [ ] **Blog REST API**: Simple Django REST API with authentication
- [ ] **Task Manager**: Full-featured app with Django, PostgreSQL, Docker, Celery
- [ ] **E-commerce Backend** (optional): Complex system demonstrating advanced skills

### Common Interview Topics
- [ ] Django ORM and query optimization
- [ ] Database design and normalization
- [ ] REST API best practices
- [ ] Docker containerization
- [ ] Debugging production issues
- [ ] System design basics

## Resources Summary

### Documentation
- [Django Documentation](https://docs.djangoproject.com/)
- [Django REST Framework](https://www.django-rest-framework.org/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [Celery Documentation](https://docs.celeryproject.org/)

### Local Guides
- `docs/django.md` - Complete Django guide
- `docs/databases.md` - PostgreSQL and database guide
- `docs/docker.md` - Docker containerization guide
- `docs/networking.md` - Networking fundamentals
- `learning-plans/interview-prep.md` - Interview questions

### Projects
- `django/10_integration_project/` - Task Manager integration project
- `web-backend/docker/` - Docker examples
- `web-backend/databases/` - Database exercises

### Online Practice
- [Django Tutorial](https://docs.djangoproject.com/en/stable/intro/tutorial01/)
- [PostgreSQL Exercises](https://pgexercises.com/)
- [LeetCode Database Problems](https://leetcode.com/problemset/database/)
- [Docker Labs](https://labs.play-with-docker.com/)

## Success Metrics

### By Week 4
- Can build complete Django REST API independently
- Comfortable with PostgreSQL queries and schema design
- Can containerize Django applications

### By Week 6
- Can implement background tasks with Celery
- Proficient in Linux command line
- Understand networking concepts

### By Week 8-10
- Have portfolio-ready projects
- Confident in all required skills
- Ready for technical interviews

## Tips for Success

### Learning Tips
1. **Build, don't just read**: Code along with tutorials
2. **Practice daily**: Consistency beats cramming
3. **Debug everything**: Learn from errors
4. **Read documentation**: Official docs are best resources
5. **Ask questions**: Use Stack Overflow, Django forums

### Project Tips
1. **Start simple**: Build features incrementally
2. **Write tests**: TDD helps catch issues early
3. **Use Git**: Commit frequently with clear messages
4. **Document**: README and code comments
5. **Deploy**: Even locally, practice full stack

### Interview Tips
1. **Practice out loud**: Explain your thinking
2. **Use STAR method**: Situation, Task, Action, Result
3. **Ask clarifying questions**: Don't assume requirements
4. **Admit gaps**: "I don't know but I'd approach it by..."
5. **Show enthusiasm**: Demonstrate genuine interest

## Next Steps

1. **Set start date**: [Fill in your start date]
2. **Choose schedule**: Intensive (8 weeks) or Balanced (12 weeks)
3. **Set up environment**: Install Python, PostgreSQL, Docker
4. **Begin Week 1**: Start with Django fundamentals
5. **Track progress**: Use this document as checklist
6. **Stay motivated**: Remember your goal!

## Emergency Fast-Track (4 weeks)

If you need to prepare quickly:

### Week 1: Django + PostgreSQL (30 hours)
- Days 1-3: Django crash course
- Days 4-5: PostgreSQL basics
- Days 6-7: Django REST API

### Week 2: Docker + Celery (25 hours)
- Days 1-3: Docker fundamentals
- Days 4-5: Containerize Django app
- Days 6-7: Celery basics

### Week 3: Integration Project (30 hours)
- Build simplified Task Manager
- Focus on core features only
- Docker setup

### Week 4: Polish & Interview Prep (20 hours)
- Complete project
- Practice interview questions
- Prepare portfolio

**Note**: Fast-track is intense and covers only essentials. Balanced schedule recommended for better retention.

---

**Good luck with your preparation!** 🚀

Remember: The Namecheap role emphasizes learning and growth. Show enthusiasm, proactiveness, and willingness to learn. Your Python fundamentals are strong - now build the Django/backend experience to match!
