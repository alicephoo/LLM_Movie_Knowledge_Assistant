

import os
import atexit

from dotenv import load_dotenv
from openai import OpenAI
import time


from src.retrieval import hybrid_search_movies, client as qdrant_client
from src.monitoring import (
    log_rag_request
)

# Configuration


os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set.")

openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)

LLM_MODEL = "gpt-5.4-nano"

atexit.register(qdrant_client.close)

# Build Context

def build_context(movies):

    context = ""

    for movie in movies:

        context += f"""
Movie Title:
{movie['title']}

Release Year:
{movie['release_date']}

Rating:
{movie['rating']}

Director:
{movie['directors']}

Genres:
{movie['genres']}

Cast:
{movie['cast']}

Overview:
{movie['overview']}

-------------------------
"""

    return context


# Generate Answer

def generate_rag_response(question: str):

    start_time = time.perf_counter()

    movies = hybrid_search_movies(
        question,
        limit=5
    )

    context = build_context(movies)

    prompt = f"""
You are a movie recommendation assistant.

Use ONLY the retrieved movies.

When recommending movies:

- Recommend the best matching movie(s).
- Explain why each recommendation matches.
- Include:
  - Title
  - Genre
  - Director
  - Rating
  - Short explanation

If multiple retrieved movies fit, recommend more than one.

If the retrieved context does not contain enough information,
say so rather than making up facts.

Movie Database:

{context}

User Question:

{question}

Provide:

- Movie recommendation
- Short explanation why it matches
- Important details (director, genre, rating)
"""

    response = openai_client.chat.completions.create(
        model=LLM_MODEL,

        messages=[
            {
                "role": "system",
                "content": "You are an expert movie assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.3
    )

    answer = response.choices[0].message.content

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    log_rag_request(
        question=question,
        retrieved_movies=movies,
        answer=answer,
        latency_ms=latency_ms,
        model=LLM_MODEL
    )

    return {
        "question": question,
        "retrieved_movies": movies,
        "context": context,
        "answer": answer
    }

def answer_movie_question(question: str):

    result = generate_rag_response(question)

    return result["answer"]

# Test

if __name__ == "__main__":

    question = (
        "Which Christopher Nolan movie "
        "should I watch if I liked Interstellar?"
    )

    answer = answer_movie_question(question)

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(answer)