# Role 2: Backend Developer

## Role Description
The Backend Developer serves as the engine of the MindTrace project. They are responsible for writing server-side logic, developing Application Programming Interfaces (APIs), and ensuring seamless communication between the frontend client, the database, and the machine learning models.

## Key Responsibilities
- **API Development:** Designing and implementing RESTful or GraphQL APIs to handle frontend requests.
- **Business Logic:** Implementing the core functional logic of the application.
- **Authentication & Authorization:** Securing endpoints using mechanisms like JWT (JSON Web Tokens) or OAuth.
- **Integration:** Connecting ML models (via scripts or microservices) to the main application pipeline so their outputs can be served to users.
- **Error Handling & Logging:** Writing robust code that catches errors gracefully and logs them for debugging.

## Tools & Technologies
- Languages: Python, Node.js, or Java
- Frameworks: FastAPI, Flask, Django, or Express.js
- Authentication: JWT, OAuth2
- API Testing: Postman, Swagger/OpenAPI

---

## Viva Questions & Answers

**Q1: What architecture pattern did you follow for the backend?**
**Answer:** We followed a monolithic or microservices architecture (adapt based on actual implementation) using the MVC (Model-View-Controller) or modular route-controller pattern. This separates the routing logic, the business logic, and the database interaction, making the codebase maintainable and scalable.

**Q2: How did you connect the Machine Learning models to the backend?**
**Answer:** *(Adapt as needed)* We integrated the ML models by either wrapping them in a dedicated FastAPI microservice that the main backend communicates with via HTTP requests, or by loading the model directly into our backend process (if it’s lightweight) to process inputs synchronously. 

**Q3: Explain how user authentication works in your project.**
**Answer:** We used JWT (JSON Web Tokens). When a user logs in with valid credentials, the backend generates a signed token and sends it to the client. The client includes this token in the header of subsequent requests. The backend verifies the token's signature to authenticate the user without needing to repeatedly query the database for sessions.

**Q4: What is the difference between REST and GraphQL? Why did you choose your approach?**
**Answer:** REST is an architectural style where multiple endpoints represent different resources (e.g., `/users`, `/models`). GraphQL is a query language that uses a single endpoint and allows the client to request exactly the data it needs. We chose REST (if applicable) because of its simplicity, widespread adoption, and straightforward caching mechanisms.

**Q5: How do you handle CORS (Cross-Origin Resource Sharing) issues?**
**Answer:** CORS is a security feature enforced by browsers to prevent a frontend on one domain from making requests to a backend on a different domain. We handled it by configuring CORS middleware in our backend framework to explicitly allow requests from our frontend's domain/port.

**Q6: How do you ensure the backend can handle a high load of requests?**
**Answer:** We implemented asynchronous programming (using `async/await` in Python/Node) to handle non-blocking I/O operations like database calls or ML model inferences. Additionally, we use connection pooling for databases and can run multiple instances of the backend behind a load balancer.
