You are simulating real users searching for technical content on Hacker News. Your job is to generate realistic search queries that users would type BEFORE they know the exact article they're looking for.

Given this Hacker News story title: "{title}"

Generate {num_queries} diverse search queries that represent different user intents and skill levels.

CRITICAL RULES:
1. Users are searching for SOLUTIONS TO PROBLEMS, not article titles
2. Users don't know the exact technology/tool yet - they're exploring
3. Real users make typos, use informal language, and phrase things awkwardly
4. Queries should be 2-6 words (users are lazy)
5. Mix of skill levels: beginners ask "how to", experts ask specific technical questions

QUERY TYPES TO INCLUDE:

**Problem-seeking (30%):** User has a problem, doesn't know the solution
- "database slow under load"
- "app crashes with many users"
- "postgres running out of memory"

**Solution-comparison (20%):** User is evaluating options
- "mongodb vs postgres scaling"
- "best database for high traffic"
- "redis vs memcached performance"

**Learning (20%):** User wants to understand a concept
- "what is database sharding"
- "how does connection pooling work"
- "explain nosql databases"

**Debugging (15%):** User hit a specific error
- "mongodb connection timeout"
- "postgres deadlock error"
- "database backup failing"

**Implementation (15%):** User knows what they want, needs guidance
- "setup mongodb replica set"
- "postgres connection pooling config"
- "migrate mysql to postgres"

ANTI-PATTERNS (DO NOT GENERATE):
❌ Queries that are just the title rephrased: "mongodb new sharding architecture"
❌ Overly formal: "comprehensive guide to database scaling strategies"
❌ Too specific to the article: "mongodb 7.0 sharding changes"
❌ Marketing speak: "best practices for enterprise-grade database solutions"
❌ Queries with leading/trailing whitespace or multiple consecutive spaces
❌ Empty or whitespace-only queries

REALISM FACTORS:
- 10% of queries should have minor typos ("postgress", "mongo db")
- 20% should be very short (2-3 words)
- 30% should be questions ("how to scale database", "why postgres slow")
- Include some vague/ambiguous queries ("database issues", "scaling problems")

DIVERSITY REQUIREMENTS:
- Vary vocabulary (don't repeat "scaling" in all queries)
- Different skill levels (junior to senior engineer)
- Different stages of problem-solving (researching → implementing → debugging)
- Different technologies (even if story is about MongoDB, some users might search for Postgres)

OUTPUT FORMAT:
Return ONLY a valid JSON array of query strings. No explanation, no markdown, no extra text.
Each query must be properly trimmed with no leading/trailing whitespace and no multiple consecutive spaces.

Example output format:
["database slow after 10k users", "mongodb vs postgres for saas", "what is sharding", "postgres connection timeout fix", "setup database cluster"]

Now generate {num_queries} queries for: "{title}"

