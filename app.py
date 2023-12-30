


# Mount Google Drive
drive.mount('/content/drive')

# Specify the full path to the CSV file in Google Drive
file_path = '/content/drive/MyDrive/combined_data.csv'

# Load the data
data = pd.read_csv(file_path)

im using googe drive for importing files. i got this erorr:  File "/mount/src/movierecml/app.py", line 10, in <module>

    drive.mount('/content/drive')

NameError: name 'drive' is not defined

## Pre-Processing

def clean_text(text):
    # Customize this function with your preferred cleaning techniques
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove punctuation and special characters
    text = re.sub(r"\s+", " ", text)  # Remove extra whitespace
    # ... Add other cleaning steps as needed ...
    return text
def clean_text(text):
    # Your text cleaning/preprocessing logic here
    return processed_text



import numpy as np
print(data)

# Model 1 - stsb-distilbert-base

# # Load a different sentence embedding model

# model = SentenceTransformer('all-mpnet-base-v2')
# #model = SentenceTransformer('/content/drive/MyDrive/saved_models')

# # Encode plot descriptions to vectors
# plot_descriptions = data['plot'].tolist()

# # Set batch size
# batch_size = 1000

# # Initialize an empty list to store embeddings
# plot_embeddings = []

# # Use tqdm to display progress bar
# for i in tqdm(range(0, len(plot_descriptions), batch_size), desc='Encoding plot descriptions'):
#     batch = plot_descriptions[i:i + batch_size]
#     embeddings_batch = model.encode(batch)
#     plot_embeddings.extend(embeddings_batch)

# plot_embeddings_np = np.array(plot_embeddings)

# #model.save(/content/drive/MyDrive/saved_models2)

#np.save('/content/drive/MyDrive/saved_embeddings', plot_embeddings_np)
# Add the embeddings to the DataFrame
model = SentenceTransformer('/content/drive/MyDrive/saved_models')
# Load the saved embeddings
plot_embeddings_np = np.load('/content/drive/MyDrive/saved_embeddings.npy')

# Add the embeddings to the DataFrame
data['plot_embedding'] = list(plot_embeddings_np)
#data['plot_embedding'] = plot_embeddings



def recommend_movies(user_input, num_recommendations=10):
    # Encode user input
    user_embedding = model.encode([user_input])[0]

    # Compute cosine similarity between user input and plot descriptions
    similarities = cosine_similarity([user_embedding], plot_embeddings_np)[0]

    # Get indices of the most similar movies
    indices = similarities.argsort()[-num_recommendations:][::-1]

    # Display recommended movies
    recommendations = data.loc[indices, ['title', 'plot', 'image']]
    return recommendations

from google.colab import drive
drive.mount('/content/drive')
#model.save('/content/drive/MyDrive/saved_models')

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








Sample query: person starts a company and gets rich
