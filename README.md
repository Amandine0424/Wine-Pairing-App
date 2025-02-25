# Wine Pairing Recommendation Interface

# Project
This project is a Wine Pairing Recommendation System built with Python that provides personalized wine recommendations based on food descriptors. It leverages web scraping (Selenium), Natural Language Processing (NLP) techniques like Word2Vec, TF-IDF, PCA, and clustering (K-Means) for pairing wines with food. Aroma-based pairings are calculated using cosine similarity, and the results are visualized with Plotly, Matplotlib, and served via a Streamlit web interface.

# Data
The data used in this project is obtained from several online wine shops.

# Key Features
- Wine Database Construction: Built a wine database using web scraping with Selenium to collect wine details from online sources.
- Descriptor Analysis: Applied NLP techniques such as Word2Vec, TF-IDF, and PCA to analyze wine and food descriptors for better matching.
- Wine Pairing: Used K-Means clustering for grouping similar wine types. Calculated cosine similarity between wine and food descriptors to recommend suitable pairings.
- Visualization: Visualized the wine pairing results using Matplotlib (radar plots) and Plotly (interactive charts).
- Streamlit Interface: Developed a user-friendly interface with Streamlit for real-time food and wine pairing suggestions.

# Technologies Used
- Python: Main programming language
- Selenium: For web scraping and building a wine database
- Word2Vec: NLP technique for word embeddings
- TF-IDF: Text feature extraction for food and wine descriptors
- PCA: Dimensionality reduction for feature analysis
- K-Means: Clustering wines for better pairing analysis
- Cosine Similarity: For measuring similarity between aromas
- Matplotlib: For radar plots and visualizations
- Plotly: For interactive plots and charts
- Streamlit: For building the web interface

# Interface visualization (Streamlit)
TBC

# Usage
To run this notebook, simply clone this repository and open it in Jupyter Notebook or Google Colab. You can also view it on GitHub.

