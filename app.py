import subprocess
import streamlit as st
from sentence_transformers import SentenceTransformers
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from google.colab import drive
from tqdm.notebook import tqdm
from tqdm import tqdm
import pandas as pd
import re

# Install necessary dependencies
st.write("Installing dependencies...")
subprocess.run(["pip", "install", "-U", "sentence_transformers"])
# Add any other dependencies you might need

# Specify the URL of the CSV file
file_url = 'https://drive.google.com/file/d/1vs1o2PbEFF50CaaQ3u9iepS4SJrWhO0X/view?usp=sharing'

# Load the data directly from the URL
data = pd.read_csv(file_url)

# Pre-Processing
def clean_text(text):
    # Customize this function with your preferred cleaning techniques
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove punctuation and special characters
    text = re.sub(r"\s+", " ", text)  # Remove extra whitespace
    # ... Add other cleaning steps as needed ...
    return text

# Model 1 - stsb-distilbert-base
# Load a different sentence embedding model
model = SentenceTransformer('all-mpnet-base-v2')

# Encode plot descriptions to vectors
plot_descriptions = data['plot'].tolist()

# Set batch size
batch_size = 32

# Initialize an empty list to store embeddings
plot_embeddings = []

# Use tqdm to display progress bar
for i in tqdm(range(0, len(plot_descriptions), batch_size), desc='Encoding plot descriptions'):
    batch = plot_descriptions[i:i + batch_size]
    embeddings_batch = model.encode(batch)
    plot_embeddings.extend(embeddings_batch)

# Add the embeddings to the DataFrame
data['plot_embedding'] = plot_embeddings

def recommend_movies(user_input, num_recommendations=5):
    # Encode user input
    user_embedding = model.encode([user_input])[0]

    # Compute cosine similarity between user input and plot descriptions
    similarities = cosine_similarity([user_embedding], plot_embeddings)[0]

    # Get indices of the most similar movies
    indices = similarities.argsort()[-num_recommendations:][::-1]

    # Display recommended movies
    recommendations = data.loc[indices, ['title', 'plot', 'image']]
    return recommendations

# User input
user_input = input("Describe the kind of movie you want to watch: ")

# Get recommendations based on user input
recommendations = recommend_movies(user_input)

# Display recommended movies with images
print("\nRecommended Movies:")
for i, (title, plot, image) in recommendations.iterrows():
    print(f"\nTitle: {title}\nPlot: {plot}\nImage: {image}\n")

    # Display image using the provided link
    from IPython.display import Image, display
    display(Image(url=image))
