"""
Evaluation utilities for Movie Knowledge Assistant.

Metrics:
- Hit Rate
- Mean Reciprocal Rank (MRR)
"""

from typing import List, Dict

import pandas as pd
from tqdm import tqdm


def hit_rate(
    retrieved: List[Dict],
    relevant_movies: List[str]
) -> int:
    """
    Returns 1 if at least one relevant movie
    is retrieved, otherwise 0.
    """

    retrieved_titles = {
        movie["title"].strip().lower()
        for movie in retrieved
    }

    relevant_titles = {
        title.strip().lower()
        for title in relevant_movies
    }

    return int(
        len(retrieved_titles & relevant_titles) > 0
    )


def reciprocal_rank(
    retrieved: List[Dict],
    relevant_movies: List[str]
) -> float:
    """
    Computes Reciprocal Rank (RR).

    RR = 1 / rank of the first relevant movie.
    """

    relevant_titles = {
        title.strip().lower()
        for title in relevant_movies
    }

    for rank, movie in enumerate(retrieved, start=1):

        if movie["title"].strip().lower() in relevant_titles:
            return 1 / rank

    return 0.0


def evaluate_retrieval(
    ground_truth: List[Dict],
    search_function,
    top_k: int = 5
):

    hit_scores = []
    rr_scores = []
    rows = []

    for item in tqdm(
        ground_truth,
        desc="Evaluating Retrieval"
    ):

        question = item["question"]

        relevant = item["relevant_movies"]

        retrieved = search_function(
            question,
            limit=top_k
        )

        hit = hit_rate(
            retrieved,
            relevant
        )

        rr = reciprocal_rank(
            retrieved,
            relevant
        )

        hit_scores.append(hit)
        rr_scores.append(rr)

        rows.append({

            "question": question,

            "expected": ", ".join(relevant),

            "retrieved": ", ".join(
                movie["title"]
                for movie in retrieved
            ),

            "hit": hit,

            "reciprocal_rank": rr

        })

    report = pd.DataFrame(rows)

    metrics = {

        "Hit Rate": sum(hit_scores) / len(hit_scores),

        "MRR": sum(rr_scores) / len(rr_scores)

    }

    return report, metrics