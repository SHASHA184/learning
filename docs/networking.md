# Networking Fundamentals Learning Module

## Overview
Understanding networking protocols is essential for backend development. This module covers TCP/IP, HTTP, and related concepts needed for roles like the Namecheap Junior Python position.

## Learning Objectives
- Understand the OSI and TCP/IP models
- Master HTTP protocol and RESTful APIs
- Debug network issues
- Understand DNS, SSL/TLS
- Work with network tools
- Apply networking concepts in Django applications

## Prerequisites
- Basic programming knowledge
- Command-line familiarity

## Module Structure

### 1. Network Fundamentals
**Time estimate**: 2-3 days

Topics:
- OSI Model (7 layers)
- TCP/IP Model (4 layers)
- IP addressing and subnetting
- Ports and sockets
- Client-server architecture

Practice:
- `web-backend/networking/01_fundamentals/` - Basic concepts and diagrams

Key concepts:
```
OSI Model:
7. Application  - HTTP, FTP, SMTP
6. Presentation - SSL/TLS, encryption
5. Session      - Session management
4. Transport    - TCP, UDP
3. Network      - IP, routing
2. Data Link    - Ethernet, MAC
1. Physical     - Cables, signals

TCP/IP Model:
4. Application  - HTTP, DNS, SSH
3. Transport    - TCP, UDP
2. Internet     - IP, ICMP
1. Link         - Ethernet, WiFi
```

IP addressing:
```
IPv4: 192.168.1.1 (32-bit)
IPv6: 2001:0db8:85a3::8a2e:0370:7334 (128-bit)

Common IP ranges:
- 127.0.0.1: localhost
- 192.168.x.x: Private network
- 10.x.x.x: Private network
- 0.0.0.0: All interfaces

Common ports:
- 80: HTTP
- 443: HTTPS
- 22: SSH
- 3306: MySQL
- 5432: PostgreSQL
- 6379: Redis
- 8000: Django dev server
```

### 2. TCP/IP Protocol
**Time estimate**: 3-4 days

Topics:
- TCP vs UDP
- Three-way handshake
- TCP flow control and congestion control
- TCP states
- Socket programming basics
- Connection management

Practice:
- `web-backend/networking/02_tcp_ip/` - TCP examples

TCP three-way handshake:
```
Client                  Server
  |                        |
  |-------- SYN -------->  |  (Client initiates)
  |                        |
  |<----- SYN-ACK ------   |  (Server acknowledges)
  |                        |
  |-------- ACK -------->  |  (Connection established)
  |                        |
```

Python socket programming:
```python
# TCP Server
import socket

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('0.0.0.0', 8080))
server.listen(5)

print("Server listening on port 8080")

while True:
    client, address = server.accept()
    print(f"Connection from {address}")

    data = client.recv(1024)
    print(f"Received: {data.decode()}")

    client.send(b"Hello from server")
    client.close()

# TCP Client
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('localhost', 8080))
client.send(b"Hello server")
response = client.recv(1024)
print(f"Response: {response.decode()}")
client.close()
```

### 3. HTTP Protocol
**Time estimate**: 4-5 days

Topics:
- HTTP request/response structure
- HTTP methods (GET, POST, PUT, PATCH, DELETE)
- Status codes
- Headers (common headers)
- Cookies and sessions
- HTTP/1.1 vs HTTP/2 vs HTTP/3
- Keep-alive and persistent connections

Practice:
- `web-backend/networking/03_http/` - HTTP examples

HTTP request structure:
```
POST /api/users HTTP/1.1
Host: example.com
Content-Type: application/json
Authorization: Bearer token123
Content-Length: 45

{"name": "John", "email": "john@example.com"}
```

HTTP response structure:
```
HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: 82
Set-Cookie: sessionid=abc123; HttpOnly

{"id": 1, "name": "John", "email": "john@example.com"}
```

HTTP methods:
```
GET     - Retrieve resource (idempotent, safe)
POST    - Create resource
PUT     - Replace resource (idempotent)
PATCH   - Partial update (not idempotent)
DELETE  - Delete resource (idempotent)
HEAD    - Same as GET but no body
OPTIONS - Get supported methods
```

Common status codes:
```
2xx Success:
200 OK                  - Request succeeded
201 Created             - Resource created
204 No Content          - Success, no response body

3xx Redirection:
301 Moved Permanently   - Resource moved
302 Found               - Temporary redirect
304 Not Modified        - Cached version valid

4xx Client Error:
400 Bad Request         - Invalid request
401 Unauthorized        - Authentication required
403 Forbidden           - No permission
404 Not Found           - Resource not found
405 Method Not Allowed  - Wrong HTTP method
409 Conflict            - Resource conflict
422 Unprocessable       - Validation error
429 Too Many Requests   - Rate limited

5xx Server Error:
500 Internal Server     - Server error
502 Bad Gateway         - Invalid upstream response
503 Service Unavailable - Server overloaded
504 Gateway Timeout     - Upstream timeout
```

Important headers:
```
Request Headers:
- Host: example.com
- User-Agent: Mozilla/5.0...
- Accept: application/json
- Accept-Encoding: gzip, deflate
- Authorization: Bearer token
- Content-Type: application/json
- Cookie: sessionid=abc123

Response Headers:
- Content-Type: application/json
- Content-Length: 1234
- Set-Cookie: sessionid=abc123
- Cache-Control: max-age=3600
- ETag: "33a64df551425fcc55e4d42a148795d9f25f89d4"
- Location: /new-url (with redirects)
- Access-Control-Allow-Origin: * (CORS)
```

Python HTTP examples:
```python
import requests

# GET request
response = requests.get('https://api.example.com/users')
print(response.status_code)
print(response.json())
print(response.headers)

# POST request
data = {'name': 'John', 'email': 'john@example.com'}
headers = {'Authorization': 'Bearer token123'}
response = requests.post(
    'https://api.example.com/users',
    json=data,
    headers=headers
)

# Custom headers and timeout
response = requests.get(
    'https://api.example.com/users',
    headers={'User-Agent': 'MyApp/1.0'},
    timeout=5
)
```

### 4. RESTful API Design
**Time estimate**: 3-4 days

Topics:
- REST principles
- Resource naming conventions
- URL structure
- HTTP method semantics
- HATEOAS
- API versioning
- Error handling

Practice:
- `web-backend/networking/04_rest_api/` - REST API design examples

REST principles:
```
1. Client-Server separation
2. Stateless communication
3. Cacheable responses
4. Uniform interface
5. Layered system
6. Code on demand (optional)
```

Resource naming:
```
Good examples:
GET    /api/users              - List users
GET    /api/users/123          - Get user 123
POST   /api/users              - Create user
PUT    /api/users/123          - Update user 123
PATCH  /api/users/123          - Partial update
DELETE /api/users/123          - Delete user 123
GET    /api/users/123/posts    - Get posts by user 123

Avoid:
/getUsers
/createUser
/user/delete/123
```

Query parameters:
```
Pagination:
/api/users?page=2&limit=20
/api/users?offset=20&limit=20

Filtering:
/api/products?category=electronics&price_min=100&price_max=500

Sorting:
/api/users?sort=created_at&order=desc

Search:
/api/products?search=laptop

Fields selection:
/api/users?fields=id,name,email
```

### 5. DNS (Domain Name System)
**Time estimate**: 2-3 days

Topics:
- DNS hierarchy
- DNS record types (A, AAAA, CNAME, MX, TXT)
- DNS resolution process
- DNS caching
- Troubleshooting DNS issues

Practice:
- `web-backend/networking/05_dns/` - DNS exercises

DNS record types:
```
A       - IPv4 address
AAAA    - IPv6 address
CNAME   - Canonical name (alias)
MX      - Mail exchange
TXT     - Text records
NS      - Name server
SOA     - Start of authority
```

DNS tools:
```bash
# Query DNS
nslookup example.com
dig example.com
host example.com

# Detailed query
dig example.com +trace
dig example.com ANY

# Check specific record
dig example.com A
dig example.com MX
dig example.com TXT

# Reverse DNS lookup
dig -x 93.184.216.34
```

### 6. SSL/TLS and HTTPS
**Time estimate**: 3-4 days

Topics:
- Encryption basics (symmetric vs asymmetric)
- SSL/TLS handshake
- Certificates and Certificate Authorities
- HTTPS vs HTTP
- Common SSL/TLS issues
- Let's Encrypt

Practice:
- `web-backend/networking/06_ssl_tls/` - SSL/TLS examples

TLS handshake:
```
Client                          Server
  |                                |
  |---- ClientHello ------------> |
  |                                |
  |<--- ServerHello -------------- |
  |<--- Certificate -------------- |
  |<--- ServerHelloDone ---------- |
  |                                |
  |---- ClientKeyExchange ------> |
  |---- ChangeCipherSpec -------> |
  |---- Finished ---------------> |
  |                                |
  |<--- ChangeCipherSpec --------- |
  |<--- Finished ----------------- |
  |                                |
  |<==== Encrypted data =========>|
```

SSL/TLS debugging:
```bash
# Check SSL certificate
openssl s_client -connect example.com:443

# View certificate details
openssl s_client -connect example.com:443 -showcerts

# Test specific TLS version
openssl s_client -connect example.com:443 -tls1_2

# Check certificate expiry
echo | openssl s_client -connect example.com:443 2>/dev/null | openssl x509 -noout -dates
```

### 7. Network Tools & Debugging
**Time estimate**: 3-4 days

Topics:
- ping, traceroute
- netstat, ss
- tcpdump, wireshark
- curl, wget
- nc (netcat)
- telnet
- Browser developer tools

Practice:
- `web-backend/networking/07_tools/` - Tool usage examples

Essential commands:
```bash
# Test connectivity
ping example.com
ping -c 4 example.com  # Send 4 packets

# Trace route
traceroute example.com
mtr example.com  # Real-time traceroute

# Check open ports
netstat -tuln
ss -tuln
lsof -i :8000

# Port scanning
nmap example.com
nc -zv example.com 80

# HTTP requests
curl https://api.example.com/users
curl -X POST https://api.example.com/users \
  -H "Content-Type: application/json" \
  -d '{"name": "John"}'

# Verbose output
curl -v https://example.com
curl -I https://example.com  # Headers only

# Save response
curl -o output.html https://example.com

# Follow redirects
curl -L https://example.com

# Test with different methods
curl -X GET https://api.example.com/users
curl -X POST https://api.example.com/users -d '{"name":"John"}'
curl -X PUT https://api.example.com/users/1 -d '{"name":"Jane"}'
curl -X DELETE https://api.example.com/users/1

# Packet capture
tcpdump -i eth0 port 80
tcpdump -i eth0 -w capture.pcap
tcpdump -r capture.pcap

# Check DNS
nslookup example.com
dig example.com
host example.com
```

### 8. Network Security Basics
**Time estimate**: 2-3 days

Topics:
- Common vulnerabilities (XSS, CSRF, SQL injection)
- CORS (Cross-Origin Resource Sharing)
- Authentication methods
- Rate limiting
- Firewalls and security groups

Practice:
- `web-backend/networking/08_security/` - Security examples

CORS:
```python
# Django CORS configuration
CORS_ALLOWED_ORIGINS = [
    "https://example.com",
    "http://localhost:3000",
]

CORS_ALLOW_METHODS = [
    'DELETE',
    'GET',
    'OPTIONS',
    'PATCH',
    'POST',
    'PUT',
]

CORS_ALLOW_HEADERS = [
    'accept',
    'authorization',
    'content-type',
]
```

Security headers:
```python
# Django security settings
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'
```

### 9. Django Networking Integration
**Time estimate**: 3-4 days

Topics:
- Django request/response cycle
- Middleware and request processing
- WSGI and ASGI
- Reverse proxy (Nginx)
- Load balancing
- WebSockets

Practice:
- `django/10_networking/` - Django networking examples

Django request cycle:
```
Browser → Nginx (reverse proxy) → Gunicorn (WSGI) → Django

1. Nginx receives HTTP request
2. Forwards to Gunicorn on localhost:8000
3. Gunicorn creates WSGI environ
4. Django processes through middleware
5. URL routing finds view
6. View processes request
7. Response travels back through middleware
8. Gunicorn sends to Nginx
9. Nginx sends to browser
```

Django middleware example:
```python
class RequestLoggingMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Log request
        print(f"{request.method} {request.path}")
        print(f"Headers: {request.headers}")

        response = self.get_response(request)

        # Log response
        print(f"Status: {response.status_code}")

        return response
```

## Practical Projects

### Project 1: HTTP Server from Scratch
**Time**: 1 week
**Goal**: Build a basic HTTP server using Python sockets
**Features**:
- Handle GET/POST requests
- Parse headers
- Serve static files
- Return proper status codes

### Project 2: API Client Library
**Time**: 1 week
**Goal**: Build a client library for a REST API
**Features**:
- HTTP methods implementation
- Authentication
- Error handling
- Retry logic

### Project 3: Network Monitoring Tool
**Time**: 1.5 weeks
**Goal**: Monitor network services
**Features**:
- Check service availability
- Measure response times
- Alert on failures
- Dashboard with statistics

## Interview Preparation

### Common Networking Questions
1. Explain the difference between TCP and UDP
2. What happens when you type a URL in the browser?
3. What is the three-way handshake?
4. Explain HTTP status codes (200, 404, 500, etc.)
5. What's the difference between HTTP and HTTPS?
6. What are HTTP headers and name some important ones?
7. Explain REST principles
8. What is CORS and why is it needed?
9. How does DNS work?
10. What is a reverse proxy?

### Practical Scenarios
- Debug a failing API request
- Design a REST API for a blog
- Troubleshoot slow network requests
- Explain how to secure an API
- Diagnose connection timeout issues

## Tools & Resources

### Command-line Tools
```bash
# Ubuntu/Debian installation
sudo apt install curl wget netcat-openbsd dnsutils \
  net-tools iputils-ping traceroute nmap tcpdump
```

### Python Libraries
```python
# HTTP clients
import requests
import httpx  # Async HTTP client
import urllib3

# Low-level networking
import socket
import asyncio
import aiohttp  # Async HTTP
```

### Resources
- [MDN HTTP Documentation](https://developer.mozilla.org/en-US/docs/Web/HTTP)
- [RFC 2616 - HTTP/1.1](https://www.ietf.org/rfc/rfc2616.txt)
- [REST API Tutorial](https://www.restapitutorial.com/)
- [High Performance Browser Networking](https://hpbn.co/)

## Integration with Other Modules

### With Django
- Understanding Django's request/response cycle
- Implementing RESTful APIs
- Debugging network issues in Django apps

### With Docker
- Container networking
- Port mapping
- Service communication

### With Deployment
- Nginx configuration
- Load balancing
- SSL/TLS setup

## Progress Tracking
- [ ] Network fundamentals understood
- [ ] TCP/IP protocol mastered
- [ ] HTTP protocol learned
- [ ] REST API design principles understood
- [ ] DNS concepts learned
- [ ] SSL/TLS basics understood
- [ ] Network debugging tools mastered
- [ ] Security basics learned
- [ ] Django networking integrated
- [ ] HTTP server project completed
- [ ] API client library completed
- [ ] Network monitoring tool completed

## Next Steps
After completing this module:
1. Apply networking knowledge to Django REST APIs
2. Set up Nginx as reverse proxy
3. Implement SSL/TLS with Let's Encrypt
4. Build production-ready API with proper error handling
5. Learn advanced topics: HTTP/2, WebSockets, gRPC
