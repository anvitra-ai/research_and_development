"""Stages 5-6: slots -> Cypher -> rows, with the fallback cascade.

The core retrieval layer. Slot extraction picks which entities fill nnp1/nnp2/
prop1, expansion builds the parameterized Cypher, and execution runs it against
Neo4j -- retrying with progressively broader strategies and rejecting off-topic
results via the relevance gate until something actually answers the question.
"""
