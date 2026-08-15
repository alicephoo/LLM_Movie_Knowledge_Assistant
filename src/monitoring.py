import os
import json
from datetime import datetime

import psycopg2
from dotenv import load_dotenv


load_dotenv()

# PostgreSQL Configuration

DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "movie_rag")
DB_USER = os.getenv("POSTGRES_USER", "phoothwincho")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")


# Database Connection

def get_connection():

    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )


# Create Logs Table

def create_logs_table():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rag_logs (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            question TEXT NOT NULL,
            retrieved_movies JSONB,
            answer TEXT,
            latency_ms FLOAT,
            model TEXT,
            error TEXT
        );
    """)

    connection.commit()

    cursor.close()
    connection.close()

# Create User Feedback Table


def create_feedback_table():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_feedback (
            id SERIAL PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT,
            feedback VARCHAR(20) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    connection.commit()

    cursor.close()
    connection.close()


# Save User Feedback

def save_user_feedback(
    question,
    answer,
    feedback
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO user_feedback (
            question,
            answer,
            feedback
        )
        VALUES (%s, %s, %s)
        """,
        (
            question,
            answer,
            feedback
        )
    )

    connection.commit()

    cursor.close()
    connection.close()

# Log RAG Request

def log_rag_request(
    question,
    retrieved_movies,
    answer,
    latency_ms,
    model,
    error=None
):

    connection = get_connection()

    cursor = connection.cursor()

    movie_data = [
        {
            "title": movie.get("title"),
            "rating": movie.get("rating"),
            "release_date": movie.get("release_date")
        }
        for movie in retrieved_movies
    ]

    cursor.execute(
        """
        INSERT INTO rag_logs (
            question,
            retrieved_movies,
            answer,
            latency_ms,
            model,
            error
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            question,
            json.dumps(movie_data),
            answer,
            latency_ms,
            model,
            error
        )
    )

    connection.commit()

    cursor.close()
    connection.close()

# Initialize Monitoring Tables

def initialize_monitoring():

    create_logs_table()
    create_feedback_table()