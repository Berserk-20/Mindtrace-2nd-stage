# Role 1: Database Manager

## Role Description
The Database Manager is responsible for designing, implementing, and maintaining the database infrastructure of the MindTrace project. They ensure that all data (user data, application state, ML embeddings, and logs) is stored securely, efficiently, and with high availability.

## Key Responsibilities
- **Schema Design:** Designing relational and non-relational database schemas tailored to the project's requirements.
- **Data Integrity & Security:** Ensuring user data privacy, implementing access controls, and maintaining backups.
- **Performance Optimization:** Writing efficient queries, indexing tables, and preventing database bottlenecks.
- **Vector Database Management:** If applicable to MindTrace, managing vector databases (like Pinecone or Milvus) for storing AI embeddings.
- **Integration:** Collaborating with backend developers to establish seamless connections between APIs and databases.

## Tools & Technologies
- SQL Databases (e.g., PostgreSQL, MySQL)
- NoSQL Databases (e.g., MongoDB for unstructured data)
- Vector Databases (e.g., Pinecone, Weaviate for AI integrations)
- ORMs (e.g., SQLAlchemy, Prisma)

---

## Viva Questions & Answers

**Q1: Why did you choose the specific database (SQL vs NoSQL) for the MindTrace project?**
**Answer:** *If SQL:* We chose a relational database like PostgreSQL because our data (users, profiles, structured logs) required strict ACID properties, relational integrity, and complex queries. 
*If NoSQL:* We chose MongoDB because our application deals with a lot of unstructured data (like varied AI outputs or JSON logs) where flexibility and rapid horizontal scaling were more critical than strict relational schemas.

**Q2: How do you handle data security and privacy in the database?**
**Answer:** We implement Role-Based Access Control (RBAC) to restrict unauthorized access. Sensitive information, such as user passwords, is hashed using algorithms like bcrypt before storage. We also use encryption at rest and in transit, and ensure that no Personal Identifiable Information (PII) is exposed in our logs.

**Q3: What is indexing, and how does it improve database performance?**
**Answer:** Indexing is a data structure technique that allows the database to find and retrieve specific rows much faster than scanning the entire table. It works like an index in a book. We applied indexes on columns that are frequently used in `WHERE` clauses or as foreign keys to speed up our queries.

**Q4: How do you prevent SQL Injection attacks?**
**Answer:** We prevent SQL injection by using Object-Relational Mappers (ORMs) or prepared statements with parameterized queries. This ensures that user inputs are treated strictly as data and never as executable code.

**Q5: What are ACID properties?**
**Answer:** ACID stands for Atomicity, Consistency, Isolation, and Durability. 
- **Atomicity:** Ensures a transaction is all-or-nothing.
- **Consistency:** Ensures the database remains in a valid state.
- **Isolation:** Ensures concurrent transactions do not interfere with each other.
- **Durability:** Ensures that once a transaction is committed, it remains saved even in case of a system crash.

**Q6: What is a Vector Database, and why might an AI project like MindTrace need one?**
**Answer:** A Vector Database stores data as high-dimensional vectors (embeddings). For an AI project, we use it to quickly search for semantic similarities—such as finding similar context for a prompt or matching user behavior embeddings. Traditional databases are not optimized for fast nearest-neighbor searches in high-dimensional spaces.
