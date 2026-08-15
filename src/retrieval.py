"""
Movie Knowledge Assistant - Retrieval Module

Retrieval methods:
1. Dense Vector Search using Qdrant
2. BM25 Keyword Search
3. Hybrid Search using Reciprocal Rank Fusion (RRF)

Public functions:
- search_movies()          -> Dense Vector Search baseline
- hybrid_search_movies()  -> Hybrid Search

The two public functions are intentionally separate so
retrieval evaluation can compare them fairly.
"""

from pathlib import Path
import re
import os

import numpy as np
import pandas as pd

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# Configuration

COLLECTION_NAME = "movies"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

QDRANT_PATH = PROJECT_ROOT / "qdrant_db"

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "movie_dataset_clean.parquet"
)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Number of candidates retrieved before RRF
DENSE_LIMIT = 50
BM25_LIMIT = 50

# RRF constant
RRF_K = 60


# Qdrant Client

QDRANT_URL = os.getenv("QDRANT_URL")

if QDRANT_URL:
    client = QdrantClient(
        url=QDRANT_URL
    )
else:
    client = QdrantClient(
        path=str(QDRANT_PATH)
    )

# Embedding Model

model = SentenceTransformer(
    EMBEDDING_MODEL
)


# Load Dataset for BM25

df = pd.read_parquet(
    DATASET_PATH
)

df = df.reset_index(drop=True)

# Text Tokenization


def tokenize(text):
    """
    Normalize text and split into useful tokens.

    Example:

        "Science-Fiction Movie!"
        ->
        ["science", "fiction", "movie"]
    """

    text = str(text).lower()

    # Keep letters and numbers.
    # Replace punctuation with spaces.
    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text
    )

    return text.split()


# BM25 Document Preparation


def build_bm25_documents():
    """
    Build text documents for BM25 search.

    Searchable fields:
    - title
    - overview
    - genres
    - directors
    - cast
    - keywords
    """

    documents = []

    for _, row in df.iterrows():

        title = str(
            row.get("title", "")
        )

        overview = str(
            row.get("overview", "")
        )

        genres = str(
            row.get("genres", "")
        )

        directors = str(
            row.get("directors", "")
        )

        cast = str(
            row.get("cast", "")
        )

        keywords = str(
            row.get("keywords", "")
        )

        document = " ".join(
            [
                title,
                overview,
                genres,
                directors,
                cast,
                keywords,
            ]
        )

        documents.append(
            document
        )

    return documents


bm25_documents = build_bm25_documents()


# Tokenize BM25 Documents

tokenized_documents = [
    tokenize(document)
    for document in bm25_documents
]


# Build BM25 Index

bm25 = BM25Okapi(
    tokenized_documents
)


# Helper: Format Movie

def format_movie(
    payload,
    score=None
):
    """
    Convert a movie payload into a consistent dictionary.

    Dense Search, BM25, and Hybrid Search all return
    the same movie structure.
    """

    movie = {
        "title": payload.get(
            "title",
            ""
        ),

        "overview": payload.get(
            "overview",
            ""
        ),

        "genres": payload.get(
            "genres",
            ""
        ),

        "directors": payload.get(
            "directors",
            ""
        ),

        "cast": payload.get(
            "cast",
            ""
        ),

        "keywords": payload.get(
            "keywords",
            ""
        ),

        "rating": payload.get(
            "rating",
            payload.get(
                "vote_average",
                ""
            )
        ),

        "release_date": payload.get(
            "release_date",
            ""
        ),

        "runtime": payload.get(
            "runtime",
            ""
        ),
    }

    if score is not None:
        movie["score"] = float(
            score
        )

    return movie


# Dense Vector Search

def dense_search(
    query: str,
    limit: int = DENSE_LIMIT
):
    """
    Search Qdrant using semantic similarity.

    Returns raw Qdrant search results.
    """

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=limit,
        with_payload=True
    )

    return results.points


# Public Dense Search

def search_movies(
    query: str,
    limit: int = 5
):
    """
    Dense Vector Search baseline.
    """

    results = dense_search(
        query=query,
        limit=limit
    )

    movies = []

    for result in results:

        payload = result.payload or {}

        movie = format_movie(
            payload=payload,
            score=result.score
        )

        movies.append(
            movie
        )

    return movies


# BM25 Keyword Search


def keyword_search(
    query: str,
    limit: int = BM25_LIMIT
):
    """
    Search movies using BM25 keyword matching.

    Returns raw BM25 results.
    """

    query_tokens = tokenize(
        query
    )

    scores = bm25.get_scores(
        query_tokens
    )

    top_indices = np.argsort(
        scores
    )[::-1][:limit]

    results = []

    for index in top_indices:

        results.append(
            {
                # IMPORTANT:
                # This index corresponds to the same
                # row ordering used to create Qdrant.
                "index": int(index),

                "score": float(
                    scores[index]
                ),

                "payload": df.iloc[
                    index
                ].to_dict()
            }
        )

    return results



def reciprocal_rank_fusion(
    dense_results,
    keyword_results,
    limit=5,
    k=RRF_K
):
    """
    Combine Dense and BM25 results using
    Reciprocal Rank Fusion (RRF).

    RRF formula:

        RRF(d) = sum(1 / (k + rank))

    Movies appearing in both retrieval methods
    receive a higher combined score.

    IMPORTANT:
    Dense uses Qdrant point IDs.

    BM25 uses DataFrame indices.

    These are aligned because the Qdrant collection
    was created using the same row ordering as the
    BM25 dataset.
    """

    fused_scores = {}

    movie_data = {}

    # Dense Results

    for rank, result in enumerate(
        dense_results,
        start=1
    ):

        payload = result.payload or {}

        # Qdrant point ID corresponds to
        # the DataFrame row index.
        movie_id = str(
            result.id
        )

        fused_scores[movie_id] = (
            fused_scores.get(
                movie_id,
                0.0
            )
            + 1 / (
                k + rank
            )
        )

        movie_data[movie_id] = payload


    # BM25 Results

    for rank, result in enumerate(
        keyword_results,
        start=1
    ):

        payload = result["payload"]

        movie_id = str(
            result["index"]
        )

        fused_scores[movie_id] = (
            fused_scores.get(
                movie_id,
                0.0
            )
            + 1 / (
                k + rank
            )
        )

        movie_data[movie_id] = payload


    # Sort by RRF Score
    ranked_movies = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # Build Final Results

    movies = []

    for movie_id, score in ranked_movies[:limit]:

        payload = movie_data[movie_id]

        movie = format_movie(
            payload=payload,
            score=score
        )

        movies.append(
            movie
        )

    return movies


# Hybrid Search


def hybrid_search(
    query: str,
    limit: int = 5
):
    """
    Hybrid retrieval pipeline:

        Query
          |
          +-------------------+
          |                   |
          v                   v
       Dense                BM25
       Search              Search
          |                   |
          +---------+---------+
                    |
                    v
                  RRF
                    |
                    v
               Top K Movies
    """

    dense_results = dense_search(
        query=query,
        limit=DENSE_LIMIT
    )

    keyword_results = keyword_search(
        query=query,
        limit=BM25_LIMIT
    )

    movies = reciprocal_rank_fusion(
        dense_results=dense_results,
        keyword_results=keyword_results,
        limit=limit
    )

    return movies


# Public Hybrid Search


def hybrid_search_movies(
    query: str,
    limit: int = 5
):
    """
    Public Hybrid Search function.

    Used for comparing Hybrid Search against
    the Dense Vector Search baseline.
    """

    return hybrid_search(
        query=query,
        limit=limit
    )


# Close Qdrant


def close_qdrant():
    """
    Close the local Qdrant client.
    """

    client.close()


# Local Test


if __name__ == "__main__":

    query = (
        "A science fiction movie "
        "about space travel"
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "DENSE VECTOR SEARCH"
    )

    print(
        "=" * 60
    )

    dense_movies = search_movies(
        query=query,
        limit=5
    )

    for rank, movie in enumerate(
        dense_movies,
        start=1
    ):

        print(
            f"{rank}. "
            f"{movie['title']} "
            f"(score={movie['score']:.4f})"
        )

    print(
        "\n" + "=" * 60
    )

    print(
        "HYBRID SEARCH"
    )

    print(
        "=" * 60
    )

    hybrid_movies = hybrid_search_movies(
        query=query,
        limit=5
    )

    for rank, movie in enumerate(
        hybrid_movies,
        start=1
    ):

        print(
            f"{rank}. "
            f"{movie['title']} "
            f"(RRF={movie['score']:.4f})"
        )

    close_qdrant()