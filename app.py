import openai
from openai.embeddings_utils import (
    distances_from_embeddings,
    indices_of_nearest_neighbors_from_distances,
)
import os
from dotenv import load_dotenv
import os
import argparse
import pandas as pd
import numpy as np
from tenacity import retry, wait_random_exponential, stop_after_attempt
import pickle
import tiktoken
from nomic import atlas


# Establish a cache of embedding's to avoid re-computing
# Cash is a dictionary of tuples (text, model) to embedding, saved as a pickle file

embedding_cache_path = "movie_embeddings.pkl"

# load the cache if it exists, and save a copy to disk
try:
    embedding_cache = pd.read_pickle(embedding_cache_path)
except FileNotFoundError:
    embedding_cache = {}
with open(embedding_cache_path, "wb") as embedding_cache_file:
    pickle.dump(embedding_cache, embedding_cache_file)


def main():
    dataset_path = "./movie_plots.csv"
    df = pd.read_csv(dataset_path)
    movies = (
        df[df["Origin/Ethnicity"] == "American"]
        .sort_values("Release Year", ascending=False)
        .head(5000)
    )
    movie_plots = movies["Plot"].values

    # enc = tiktoken.encoding_for_model("text-embedding-ada-002")

    # total_tokens = sum([len(enc.encode(plot)) for plot in movie_plots])
    # cost = total_tokens * (0.0001 / 1000)
    # print(f"cost: {cost:.2f}")

    plot_embedding = [
        embedding_from_string(plot, model="text-embedding-ada-002")
        for plot in movie_plots
    ]

    data = movies[["Title", "Genre"]].to_dict("records")

    # project = atlas.map_embeddings(
    #     embeddings=np.array(plot_embedding),
    #     data=data,
    # )

    print_recommendations_from_strings(movie_plots, 3)


def print_recommendations_from_strings(
    strings,
    index_of_source_string,
    k_nearest_neighbors=3,
    model="text-embedding-ada-002",
):
    # get all of the embeddings
    embeddings = [embedding_from_string(string) for string in strings]
    # get embedding for our specific query string
    query_embedding = embeddings[index_of_source_string]
    # get distances between our embedding and all other embeddings
    distances = distances_from_embeddings(query_embedding, embeddings)
    # get indices of nearest neighbors
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(
        distances
    )

    query_string = strings[index_of_source_string]
    match_count = 0

    for i in indices_of_nearest_neighbors:
        if query_string == strings[i]:
            continue
        if match_count >= k_nearest_neighbors:
            break
        match_count += 1

        print(f"Found {match_count} closest match: ")
        print(f"Distance of: {distances[i]} ")
        print(strings[i])


# Define a function to retrieve an beddings from the cache of present, otherwise request via OpenAI API
def embedding_from_string(
    string, model="text-embedding-ada-002", embedding_cache=embedding_cache
):
    if (string, model) not in embedding_cache.keys():
        embedding_cache[(string, model)] = get_embedding(string, model)
        print(f"GOT EMBEDDING FROM OPENAI FOR {string[:20]}")
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, model)]


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=text, model=model)["data"][0]["embedding"]


if __name__ == "__main__":
    load_dotenv(".env")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    main()
