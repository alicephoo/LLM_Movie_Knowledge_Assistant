from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# Load reranking model
reranker = CrossEncoder(
    MODEL_NAME,
    device="cpu"
)


def movie_to_text(movie):
    """
    Convert a movie dictionary into text for the cross-encoder.
    """

    return " ".join([
        f"Title: {movie.get('title', '')}",
        f"Overview: {movie.get('overview', '')}",
        f"Genres: {movie.get('genres', '')}",
        f"Directors: {movie.get('directors', '')}",
        f"Cast: {movie.get('cast', '')}",
        f"Keywords: {movie.get('keywords', '')}",
    ])


def rerank_movies(query, candidates, limit=5):
    """
    Rerank retrieved movie candidates using a cross-encoder.

    Args:
        query: User's search query.
        candidates: Movies returned by hybrid/vector search.
        limit: Number of final movies to return.

    Returns:
        Top reranked movie candidates.
    """

    if not candidates:
        return []

    # Create query-document pairs
    pairs = [
        [query, movie_to_text(movie)]
        for movie in candidates
    ]

    # Calculate relevance scores
    scores = reranker.predict(pairs)

    # Combine movies with their reranking scores
    ranked_results = []

    for movie, score in zip(candidates, scores):

        movie = movie.copy()

        movie["rerank_score"] = float(score)

        ranked_results.append(movie)

    # Sort by reranking score
    ranked_results.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    # Return top results
    return ranked_results[:limit]