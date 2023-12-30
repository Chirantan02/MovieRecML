**Movie Recommendation System with Sentence Encoding**

Welcome to the Movie Recommendation System with Machine Learning, a project designed to provide personalized movie suggestions based on user input. Leveraging advanced sentence encoding algorithms, this system transforms movie plot descriptions into numerical vectors, enabling efficient similarity computations for personalized recommendations.

**Key Features:**

- **Sentence Embedding Model:** The system utilizes the powerful 'all-mpnet-base-v2' SentenceTransformer model for converting movie plot descriptions into rich numerical representations.

- **Batch Processing:** To handle large datasets seamlessly, a batch processing approach is employed. The plot descriptions are encoded in batches, optimizing memory usage and computational efficiency.

**How It Works:**
- Sentence Embedding Model:
The recommendation system employs the SentenceTransformer library with the 'all-mpnet-base-v2' model for encoding plot descriptions into vectors. This allows for a nuanced understanding of movie plots using machine learning.

- User Interaction:
Users can describe the kind of movie they want to watch by providing input. The system then generates movie recommendations tailored to the user's preferences.

- Cosine Similarity:
Cosine similarity is used to measure the likeness between the user's input and the plot descriptions of various movies. This similarity score is then employed to recommend movies with the highest relevance.

**Code Example:**

```python
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
```



Experience the magic of machine learning in movie recommendations with this user-friendly and efficient system. Enhance your movie-watching journey by exploring personalized suggestions tailored just for you!
