# VSM Information Retrieval

This project implements a Vector Space Model (VSM) for information retrieval. The aim is to retrieve relevant documents based on user queries using the TF-IDF weighting scheme.

## Project Overview

The project consists of the following components:

1. Preprocessing: The input documents are preprocessed to remove special characters, tokenize the text, perform stemming, and remove stop words. The preprocessed documents are then stored in the "Dataset" folder.

2. Document Vectorization: The documents are vectorized using the TF-IDF vectorizer from scikit-learn. The resulting document vectors are normalized and saved as JSON files.

3. Query Processing: User queries are processed in a similar manner as document vectorization. The query vector is calculated using the TF-IDF vectorizer with the preprocessed documents' vocabulary.

4. Similarity Calculation: The cosine similarity between the query vector and document vectors is calculated to rank the documents based on relevance.

5. Graphical User Interface (GUI): The project includes a simple GUI built with tkinter to provide a user-friendly interface for querying and displaying search results.

## Dependencies

- Python 3.7 or higher
- NLTK (Natural Language Toolkit)
- scikit-learn
- tkinter (for GUI)
