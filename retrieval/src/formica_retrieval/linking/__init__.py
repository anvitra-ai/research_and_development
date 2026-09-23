"""Stage 1/3/4: link query spans to knowledge-graph nodes.

Gazetteer, alias table, NER and embedding similarity decide which node a phrase
in the question refers to. Everything downstream is only as good as this: a
wrong node here produces a confidently wrong answer.
"""
