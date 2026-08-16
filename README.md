# 🎬 Movie Knowledge Assistant

An end-to-end Retrieval-Augmented Generation (RAG) application that helps users discover movies using natural-language questions.

The application combines **vector search, BM25 keyword search, hybrid retrieval, document re-ranking, OpenAI generation, user feedback, PostgreSQL monitoring, Grafana dashboards, Qdrant, and Streamlit**.

---

## 📌 Project Overview

Finding the right movie can be difficult when users describe what they want using natural language rather than exact movie titles.

For example:

> "Which movie should I watch if I liked Interstellar?"

or:

> "Recommend a science fiction movie about space exploration."

A traditional keyword search may fail when the user's wording differs from the movie metadata.

This project addresses this problem by building a RAG-based Movie Knowledge Assistant that retrieves relevant movies from a movie knowledge base and uses an LLM to generate grounded recommendations.

The application retrieves movie information from a Qdrant vector database and combines semantic and keyword-based retrieval to improve search quality.

---

# 🎯 Objectives

The main objectives of this project are:

- Build an end-to-end RAG application.
- Create a searchable movie knowledge base.
- Implement semantic vector search.
- Implement BM25 keyword search.
- Evaluate different retrieval approaches.
- Implement hybrid search.
- Experiment with document re-ranking.
- Evaluate the final LLM-generated responses.
- Provide an interactive Streamlit interface.
- Collect user feedback.
- Monitor application performance using PostgreSQL and Grafana.
- Containerize the application using Docker Compose.

---

# 🏗️ Architecture

```text
                         ┌─────────────────────┐
                         │      User           │
                         │  Natural Language   │
                         │      Question       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     Streamlit UI    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     RAG Pipeline    │
                         │      src/rag.py     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │       Hybrid Retrieval       │
                    │      src/retrieval.py        │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
           ┌─────────────────┐          ┌─────────────────┐
           │  Vector Search  │          │   BM25 Search   │
           │     Qdrant      │          │ Keyword Search  │
           └────────┬────────┘          └────────┬────────┘
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │ Hybrid Ranking      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Retrieved Movies   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Prompt + Context  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    OpenAI LLM       │
                         │   gpt-5.4-nano      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Movie Recommendation│
                         └──────────┬──────────┘
                                    │
                      ┌─────────────┴─────────────┐
                      │                           │
                      ▼                           ▼
              ┌───────────────┐          ┌────────────────┐
              │ User Feedback │          │ PostgreSQL Logs│
              └───────┬───────┘          └───────┬────────┘
                      │                           │
                      └─────────────┬─────────────┘
                                    ▼
                            ┌───────────────┐
                            │    Grafana    │
                            │   Dashboard   │
                            └───────────────┘





```

-----

## 📊 Dataset

The project uses a merged **IMDb/TMDB movie dataset** from Kaggle and cleaned these data to get better version.

Dataset link - https://www.kaggle.com/datasets/ggtejas/tmdb-imdb-merged-movies-dataset


The dataset contains movie information including:

* Movie title
* Overview
* Genres
* Cast
* Directors
* Keywords
* Release date
* Rating
* Runtime

The data was cleaned and filtered before being used by the RAG system.

---

## 🔎 Data Processing

The movie dataset was cleaned before ingestion.

Relevant movie fields are combined into a searchable document containing:

```text
Title
Overview
Genres
Directors
Cast
Keywords
Release Date
Rating
Runtime
```

These documents are then converted into vector embeddings.

---

## 🧠 Embeddings

The project uses:

```text
BAAI/bge-small-en-v1.5
```

The embedding dimension is:

```text
384
```

The generated embeddings are stored in:

```text
data/movie_embeddings.npy
```

---

## 🗄️ Vector Database

The project uses **Qdrant** as the vector database.

Each movie is stored as a vector together with metadata including:

```text
title
overview
genres
directors
cast
keywords
release_date
rating
runtime
```

The Qdrant collection is:

```text
movies
```

---

# 🔍 Retrieval

The application implements multiple retrieval approaches.

## 1. Vector Search

The user query is converted into an embedding and searched against movie embeddings stored in Qdrant.

This provides semantic matching.

For example:

```text
mind-bending science fiction movie
```

can retrieve movies related to the concept even when the exact words do not appear in the movie metadata.

---

## 2. BM25 Search

BM25 provides keyword-based retrieval.

This is useful when queries contain important exact terms such as:

```text
Christopher Nolan
space
Batman
science fiction
```

---

## 3. Hybrid Search

Hybrid search combines:

* Vector/semantic search
* BM25/keyword search

This allows the system to benefit from both semantic similarity and exact keyword matching.

---

# 📈 Retrieval Evaluation

Retrieval performance was evaluated using:

* Hit Rate
* Mean Reciprocal Rank (MRR)

The following methods were compared:

| Method            |  Hit Rate |       MRR |
| ----------------- | --------: | --------: |
| Vector Search     |     0.167 |     0.070 |
| **Hybrid Search** | **0.200** | **0.102** |

### Result

Hybrid Search performed better than Vector Search on both metrics.

Therefore, **Hybrid Search was selected as the retrieval strategy used by the final application**.

---

# 🔄 Document Re-ranking

Document re-ranking was also implemented and evaluated.

The following approaches were compared:

| Method              | Hit Rate@5 |     MRR@5 |
| ------------------- | ---------: | --------: |
| **Hybrid Search**   |  **0.200** | **0.102** |
| Hybrid + Re-ranking |      0.167 |     0.059 |

The evaluation showed that re-ranking did not improve retrieval performance for this dataset.

Therefore, the final application uses:

```text
Hybrid Search
        ↓
No re-ranking
```

The re-ranking implementation and evaluation are retained in the project for experimentation and reproducibility.

---

# 🤖 RAG Generation

After retrieving relevant movies, the system builds a context containing information about the retrieved movies.

The retrieved context is provided to the OpenAI model.

The application instructs the LLM to:

* Use only the retrieved movies.
* Recommend the best matching movie or movies.
* Explain why each movie matches the user's request.
* Include relevant movie details.
* Avoid inventing information when the retrieved context is insufficient.

The current LLM is:

```text
gpt-5.4-nano
```

---

# 🧪 LLM Evaluation

Generated responses were evaluated using three criteria:

* Relevance
* Correctness
* Completeness

The evaluation results were:

| Metric       | Average Score |
| ------------ | ------------: |
| Relevance    |  **4.43 / 5** |
| Correctness  |  **3.13 / 5** |
| Completeness |  **3.73 / 5** |
| **Overall**  |  **3.77 / 5** |

The results indicate that the system generally produces relevant movie recommendations, while correctness and completeness remain areas for further improvement.

---

# 🖥️ User Interface

The application uses **Streamlit** to provide an interactive web interface.

Users can enter natural-language movie questions such as:

```text
Which movie should I watch if I liked Interstellar?
```

```text
Recommend Christopher Nolan movies.
```

```text
Find science fiction movies about space.
```

The application retrieves relevant movies and generates a recommendation based on the retrieved context.


### Application Screenshot

##Basic recommendation

#Q1
<img width="3342" height="1856" alt="image" src="https://github.com/user-attachments/assets/78eb1eac-963f-439e-a421-97b092192987" />

#Q2
<img width="3342" height="1856" alt="image" src="https://github.com/user-attachments/assets/c485d3da-177b-452a-9353-4b658f583630" />

#Q3
<img width="3342" height="1856" alt="image" src="https://github.com/user-attachments/assets/1bd571df-c2dd-40e5-8a4a-5a100552579a" />

##Similar movie recommendations

#Q1
<img width="3342" height="1856" alt="image" src="https://github.com/user-attachments/assets/ff2eb19c-cbaf-4179-850e-17daa201de93" />

#Q2
<img width="3342" height="1856" alt="image" src="https://github.com/user-attachments/assets/3d619834-2d0b-4dcf-88d4-11a873528844" />

##Out-of-context / hallucination tests
#Q1
<img width="3342" height="1856" alt="image" src="https://github.com/user-attachments/assets/009d6fea-f2db-448d-954a-1a9858014992" />

#Q2
<img width="3342" height="1856" alt="image" src="https://github.com/user-attachments/assets/1c4b5d0c-edfd-4946-8854-e01b1cccf04d" />

---

# 👍 User Feedback

The application collects user feedback after generating an answer.

Users can select:

```text
👍 Yes
👎 No
```

Feedback is stored in PostgreSQL.

The feedback table is:

```text
user_feedback
```

It contains:

```text
id
question
answer
feedback
created_at
```

This allows the application to measure whether users find the generated recommendations useful.

---

# 📊 Monitoring

Application monitoring is implemented using:

* PostgreSQL
* Grafana

RAG requests are stored in the:

```text
rag_logs
```

table.

The logs contain:

* User question
* Retrieved movies
* Generated answer
* Latency
* LLM model
* Errors
* Timestamp

User feedback is stored separately in:

```text
user_feedback
```

---

## Grafana Dashboard

The monitoring dashboard tracks application usage and performance.

Recommended dashboard panels include:

1. Total RAG Requests
2. RAG Errors
3. Average Latency
4. Requests Over Time
5. User Feedback
6. Positive vs Negative Feedback

### Monitoring Screenshot

```text
screenshots/grafana-dashboard.png
```
<img width="1671" height="956" alt="Screenshot 2026-08-16 at 8 10 30 AM" src="https://github.com/user-attachments/assets/c87ad2d4-9a93-4fd4-b813-cd43ff22fa7e" />

<img width="1671" height="956" alt="Screenshot 2026-08-16 at 8 10 54 AM" src="https://github.com/user-attachments/assets/d32b3052-2be1-4c8b-aba5-d863e1407784" />

---

# 🐳 Docker

The project is containerized using Docker Compose.

The main application services include:

```text
movie-rag
qdrant
```

Qdrant provides the vector database while the Movie RAG container runs the Streamlit application.

Inside Docker, the application connects to Qdrant using:

```text
QDRANT_URL=http://qdrant:6333
```

PostgreSQL is used for application logging and user feedback.

---

# 📁 Project Structure

```text
Movie Knowledge Assistant/
│
├── app.py
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── .env
│
├── src/
│   ├── rag.py
│   ├── retrieval.py
│   ├── reranking.py
│   ├── ingest.py
│   ├── evaluation.py
│   ├── monitoring.py
│   └── test_monitoring.py
│
├── data/
│   ├── movie_dataset_clean.parquet
│   ├── movie_embeddings.npy
│   ├── retrieval_ground_truth.json
│   ├── retrieval_method_comparison.csv
│   ├── retrieval_detailed_comparison.csv
│   ├── retrieval_report.csv
│   ├── retrieval_results.csv
│   ├── vector_retrieval_report.csv
│   ├── hybrid_retrieval_report.csv
│   ├── reranking_report.csv
│   ├── reranking_detailed_results.csv
│   ├── llm_evaluation.csv
│   ├── llm_summary.csv
│   └── rag_evaluation_results.csv
│
├── notebook/
│   ├── 01_cleaned_data.ipynb
│   ├── 02_embedding_generation.ipynb
│   ├── 03_vector_database.ipynb
│   ├── 04_retrieval_evaluation.ipynb
│   ├── 07_llm_evaluation.ipynb
│   └── 06_reranking.ipynb
│
├── monitoring/
│   └── dashboard.py
│
└── screenshots/
    ├── streamlit-app.png
    └── grafana-dashboard.png
```

---

# ⚙️ Installation

## Requirements

The project uses:

* Python 3.11
* Docker
* Docker Compose
* PostgreSQL
* Qdrant
* OpenAI API

---

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Movie-Knowledge-Assistant
```

---

## 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key

POSTGRES_HOST=host.docker.internal
POSTGRES_PORT=5432
POSTGRES_DB=movie_rag
POSTGRES_USER=your_postgres_user
POSTGRES_PASSWORD=your_postgres_password
```

---

# 📦 Python Dependencies

The project dependencies are pinned to improve reproducibility.

Important versions include:

| Package               | Version |
| --------------------- | ------: |
| Python                |    3.11 |
| torch                 |   2.2.2 |
| numpy                 |  1.26.4 |
| transformers          |  4.41.2 |
| sentence-transformers |   3.0.1 |
| qdrant-client         |  1.18.0 |
| openai                |  2.49.0 |
| streamlit             |  1.60.0 |
| pandas                |   2.2.2 |
| pyarrow               |  16.1.0 |
| psycopg2-binary       |  2.9.12 |
| rank-bm25             |   0.2.2 |
| python-dotenv         |   1.2.2 |
| tqdm                  |  4.70.0 |

Install the dependencies with:

```bash
pip install -r requirements.txt
```

---

# 🚀 Running the Application

## 1. Start Docker Compose

```bash
docker compose up -d
```

Check the running services:

```bash
docker compose ps
```

---

## 2. Open Streamlit

Open the application in your browser:

```text
http://localhost:8501
```

---

# 🗄️ Qdrant Ingestion

If the Qdrant collection needs to be recreated, run the ingestion pipeline:

```bash
python src/ingest.py
```

The ingestion pipeline:

1. Loads the cleaned movie dataset.
2. Loads the generated embeddings.
3. Creates the `movies` collection.
4. Creates Qdrant points.
5. Uploads movie vectors and metadata.
6. Verifies the collection.

The ingestion process uses the existing embedding file:

```text
data/movie_embeddings.npy
```

so embeddings do not need to be regenerated unless the dataset changes.

---

# 🧪 Evaluation Files

Evaluation results are stored in the `data/` directory.

## Retrieval Evaluation

```text
retrieval_ground_truth.json
retrieval_method_comparison.csv
retrieval_detailed_comparison.csv
retrieval_report.csv
retrieval_results.csv
vector_retrieval_report.csv
hybrid_retrieval_report.csv
```

## Re-ranking Evaluation

```text
reranking_report.csv
reranking_detailed_results.csv
```

## LLM Evaluation

```text
llm_evaluation.csv
llm_summary.csv
rag_evaluation_results.csv
```

---

# 🧭 Example Usage

## Example 1

**Question:**

```text
Which Christopher Nolan movie should I watch if I liked Interstellar?
```

The system retrieves relevant movies using hybrid search and generates a recommendation based only on the retrieved movie context.

---

## Example 2

**Question:**

```text
Recommend a mind-bending science fiction movie.
```

The system uses semantic and keyword retrieval to identify relevant movies before generating the answer.

---

## Example 3

**Question:**

```text
Find a science fiction movie about space exploration.
```

The retrieval pipeline searches the movie knowledge base and provides relevant recommendations.

---

# 📊 Project Evaluation Summary

| Component           | Implementation                                      |
| ------------------- | --------------------------------------------------- |
| Problem Description | Movie recommendation using natural-language queries |
| Knowledge Base      | IMDb/TMDB movie dataset                             |
| Vector Database     | Qdrant                                              |
| Embeddings          | BAAI/bge-small-en-v1.5                              |
| Vector Search       | Yes                                                 |
| BM25 Search         | Yes                                                 |
| Hybrid Search       | Yes                                                 |
| Re-ranking          | Implemented and evaluated                           |
| LLM                 | OpenAI `gpt-5.4-nano`                               |
| LLM Evaluation      | Yes                                                 |
| Interface           | Streamlit                                           |
| User Feedback       | PostgreSQL                                          |
| Monitoring          | PostgreSQL + Grafana                                |
| Containerization    | Docker Compose                                      |
| Dependency Pinning  | Yes                                                 |

---

# 🏆 DataTalksClub LLM Zoomcamp Project Criteria

This project addresses the major project evaluation criteria.

## Problem Description

The project solves the problem of finding relevant movies from natural-language descriptions and generating grounded movie recommendations.

## Retrieval Flow

The application retrieves movie context from Qdrant using hybrid retrieval and uses an LLM to generate grounded recommendations.

```text
User Query
    ↓
Query Embedding + BM25
    ↓
Hybrid Retrieval
    ↓
Relevant Movies
    ↓
Context Construction
    ↓
OpenAI LLM
    ↓
Movie Recommendation
```

## Retrieval Evaluation

Multiple retrieval methods were evaluated:

* Vector Search
* Hybrid Search
* Hybrid Search + Re-ranking

Hybrid Search achieved the best retrieval performance.

## LLM Evaluation

Generated responses were evaluated using:

* Relevance
* Correctness
* Completeness

The overall average score was:

```text
3.77 / 5
```

## Interface

A Streamlit UI provides an interactive interface for submitting natural-language movie queries.

## Ingestion

A Python ingestion pipeline loads movie data and pre-generated embeddings into Qdrant.

## Monitoring

PostgreSQL stores RAG logs and user feedback, while Grafana provides monitoring dashboards.

## Containerization

Docker Compose is used to run the application and Qdrant.

## Reproducibility

Dependencies are pinned and Docker provides a reproducible application environment.

---

# 🔬 Best Practices

The project implements several RAG techniques.

## Hybrid Search

Hybrid Search combines:

```text
Semantic Vector Search
        +
BM25 Keyword Search
```

This improves retrieval by combining semantic understanding with exact keyword matching.

## Document Re-ranking

A document re-ranking pipeline was implemented and evaluated.

However, evaluation showed that re-ranking did not outperform the hybrid baseline for this dataset.

Therefore, the production application uses:

```text
Hybrid Search
```

without re-ranking.

The re-ranking implementation remains available for experimentation and reproducibility.

## Query Rewriting

Query rewriting was evaluated as a potential improvement but was **not included in the final implementation**.

---

# 📈 Future Improvements

Potential future improvements include:

* Query rewriting
* Improved re-ranking models
* Better grounding and factual consistency
* Larger evaluation datasets
* More detailed user feedback
* Cloud deployment
* Automated ingestion using Airflow, Kestra, or another orchestration tool
* Additional monitoring metrics
* Personalized movie recommendations
* Improved evaluation datasets and ground-truth coverage

---

# 🎯 Conclusion

The Movie Knowledge Assistant demonstrates a complete end-to-end RAG pipeline for movie recommendation.

The final system combines:

```text
IMDb/TMDB Dataset
        ↓
Data Cleaning
        ↓
BGE Embeddings
        ↓
Qdrant
        ↓
Hybrid Search
        ↓
Context Construction
        ↓
OpenAI LLM
        ↓
Streamlit Application
        ↓
PostgreSQL Logging + Feedback
        ↓
Grafana Monitoring
```

The retrieval experiments showed that **Hybrid Search outperformed Vector Search**, while the re-ranking experiment showed that re-ranking did not improve the results for this dataset.

The final application therefore uses **Hybrid Search without re-ranking**, providing a practical balance between retrieval quality and system complexity.

