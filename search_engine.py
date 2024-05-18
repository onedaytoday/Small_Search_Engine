import os

import numpy
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, List
from transformers import AutoTokenizer, AutoModel
import heapq
from sklearn.metrics.pairwise import cosine_similarity

print('Loading model...')
tokenizer = AutoTokenizer.from_pretrained('thenlper/gte-small')
model = AutoModel.from_pretrained('thenlper/gte-small')
database: Dict[str, np.ndarray] = {}  # In-memory database to store document embeddings


def embed_text(text: str) -> np.ndarray:
    """
        Generate embeddings for the given text using BERT.

        Parameters:
        text (str): The input text string to be embedded. This text is processed by the BERT 
                    tokenizer and model to generate a vector representation.

        Returns:
        np.ndarray: A numpy array representing the averaged token embeddings of the input text, 
                    effectively condensing the information of the entire input text into a single 
                    embedding vector.
    """

    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    outputs = model(**inputs).last_hidden_state

    return outputs.mean(dim=1).squeeze().detach().numpy()


def index_documents(path: str):
    """
        Index all text files in the given directory and store their embeddings in a global database.

        Parameters:
        path (str): The directory path that contains text files to be indexed. The function expects 
                    this directory to contain .txt files.
    """

    for filename in tqdm(os.listdir(path)):
        if filename.endswith('.txt'):
            file_path = os.path.join(path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                file_text = file.read()
                embedding = embed_text(file_text)
                database[filename] = embedding
    print(f"Indexed {len(database)} documents.")


def search_documents(query: str) -> List[Tuple[str, float]]:
    """
        Search the indexed documents for the top 5 documents that are most similar to the query 
        based on cosine similarity.

        Parameters:
        query (str): The search query as a string. This is the text input by a user that the system 
                    will use to find similar documents in the indexed database.

        Returns:
        List[Tuple[str, float]]: A list of tuples where each tuple contains a document's filename 
                                and its similarity score relative to the query. The list is sorted 
                                by similarity score in descending order, with only the top 5 
                                results returned.
    """
    similarities: Dict[str, float] = {}
    query_embedding = embed_text(query)

    for name, embedding in database.items():
        similarity = cosine_similarity(numpy.expand_dims(embedding, axis=0), numpy.expand_dims(query_embedding, axis=0))
        similarities.update({name: similarity[0][0]})

    similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)

    return similarities[:5]


def main():
    """Console interface for the search engine."""
    while True:
        cmd = input("Enter command (search <query>, index <path>, or exit): ")
        if cmd.startswith("search "):
            _, query = cmd.split(" ", 1)
            results = search_documents(query)
            for doc, score in results:
                print(f"{doc}: {score:.4f}")
        elif cmd.startswith("index "):
            _, path = cmd.split(" ", 1)
            index_documents(path)
        elif cmd == "exit":
            print("Exiting the search engine.")
            break
        else:
            print("Unknown command. Please use 'search <query>', 'index <path>', or 'exit'.")


if __name__ == "__main__":
    main()
