# Document Search Engine

This project implements a simple document search engine using BERT embeddings to index and search text documents. The system allows you to index documents stored in a directory and search for the most similar documents to a given query based on cosine similarity.

## Features

- **Embed Text**: Uses a BERT-based model to generate embeddings for text input.
- **Index Documents**: Indexes text documents in a specified directory by generating and storing their embeddings.
- **Search Documents**: Searches the indexed documents for the most similar documents to a given query.

## Requirements

- Python 3.6 or later
- `transformers` library
- `torch` library
- `tqdm` library
- `scikit-learn` library
- `numpy` library

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/document-search-engine.git
    cd document-search-engine
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install transformers torch tqdm scikit-learn numpy
    ```

## Usage

1. **Indexing Documents**:
    - Place your text files (`.txt`) in a directory.
    - Run the script and use the `index` command to index the documents in the directory:
        ```sh
        python search_engine.py
        ```
      Example command to index documents in a directory named `documents`:
        ```
        Enter command (search <query>, index <path>, or exit): index documents
        ```

2. **Searching Documents**:
    - After indexing the documents, you can search for similar documents by using the `search` command:
      Example command to search for documents related to the query "machine learning":
        ```
        Enter command (search <query>, index <path>, or exit): search machine learning
        ```

3. **Exiting the Search Engine**:
    - To exit the search engine, use the `exit` command:
        ```
        Enter command (search <query>, index <path>, or exit): exit
        ```

## Code Explanation

### Embedding Text

The `embed_text` function generates embeddings for the given text using a BERT-based model. The embeddings are averaged to create a single vector representation of the input text.

### Indexing Documents

The `index_documents` function processes all `.txt` files in a specified directory, generates embeddings for each file, and stores these embeddings in an in-memory database.

### Searching Documents

The `search_documents` function computes the cosine similarity between the query embedding and the embeddings of the indexed documents. It returns the top 5 documents that are most similar to the query.

### Main Function

The `main` function provides a console interface to interact with the search engine. It supports the following commands:
- `search <query>`: Search for documents related to the given query.
- `index <path>`: Index the text files in the specified directory.
- `exit`: Exit the search engine.
