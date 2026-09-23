"""Measuring the pipeline: grading, metrics and result records.

Two independent axes, deliberately kept apart. retrieval_eval is deterministic
and asks "was the answer in the retrieved rows?"; llm_judge asks "did the
summary answer the question?". Confusing the two hides whether a failure is in
retrieval or in generation.
"""
