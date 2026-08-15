"""
Movie Knowledge Assistant - Qdrant Ingestion Pipeline

This script:
1. Loads cleaned movie dataset
2. Loads generated embeddings
3. Creates Qdrant collection
4. Uploads movie vectors with metadata
"""

import os
import pandas as pd
import numpy as np

from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)


# Configuration

from pathlib import Path

COLLECTION_NAME = "movies"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "movie_dataset_clean.parquet"

EMBEDDING_PATH = PROJECT_ROOT / "data" / "movie_embeddings.npy"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_PATH = PROJECT_ROOT / "qdrant_db"


# Load Data

def load_data():

    print("Loading movie dataset...")

    df = pd.read_parquet(
    DATA_PATH
)

    print(
        f"Movies loaded: {len(df)}"
    )


    print("Loading embeddings...")

    embeddings = np.load(
    str(EMBEDDING_PATH)
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )


    assert len(df) == len(embeddings), (
        "Dataset and embeddings size mismatch"
    )


    return df, embeddings


# Create Qdrant Collection

def create_collection(
    client,
    vector_size
):

    if client.collection_exists(
        COLLECTION_NAME
    ):

        print(
            "Deleting existing collection..."
        )

        client.delete_collection(
            COLLECTION_NAME
        )


    client.create_collection(

        collection_name=COLLECTION_NAME,

        vectors_config=VectorParams(

            size=vector_size,

            distance=Distance.COSINE
        )
    )


    print(
        f"Collection '{COLLECTION_NAME}' created"
    )



# Create Qdrant Points

def create_points(
    df,
    embeddings
):

    print(
        "Preparing vectors..."
    )

    points = []


    for idx, row in tqdm(
        df.iterrows(),
        total=len(df)
    ):

        payload = {

            "title": row["title"],

            "overview": row["overview"],

            "genres": row["genres"],

            "directors": row["directors"],

            "cast": row["cast"],

            "keywords": row["keywords"],

            "release_date":
                row["release_date"]
            ,

            "rating": float(
                row["vote_average"]
            ),

            "runtime": int(
                row["runtime"]
            )
        }


        point = PointStruct(

            id=int(idx),

            vector=embeddings[idx].tolist(),

            payload=payload

        )


        points.append(point)


    return points


# Upload to Qdrant


def upload_points(
    client,
    points
):

    print(
        "Uploading vectors..."
    )


    batch_size = 100


    for start in tqdm(
        range(
            0,
            len(points),
            batch_size
        )
    ):

        client.upsert(

            collection_name=COLLECTION_NAME,

            points=points[
                start:start + batch_size
            ]

        )


    print(
        f"Uploaded {len(points)} movies"
    )


# Main Pipeline

def main():

    # Load data

    df, embeddings = load_data()


    # Connect Qdrant

    print(
        "Connecting to Qdrant..."
    )

    if QDRANT_URL:
        client = QdrantClient(
            url=QDRANT_URL
        )
    else:
        client = QdrantClient(
            path=QDRANT_PATH
        )


    # Create collection

    create_collection(
        client,
        embeddings.shape[1]
    )


    # Create points

    points = create_points(
        df,
        embeddings
    )


    # Upload

    upload_points(
        client,
        points
    )


    # Verify

    collection_info = client.get_collection(
        COLLECTION_NAME
    )


    print(
        collection_info
    )


if __name__ == "__main__":

    main()