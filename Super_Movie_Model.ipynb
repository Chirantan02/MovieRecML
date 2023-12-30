{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "collapsed_sections": [
        "jMLdgbP5thvp"
      ],
      "authorship_tag": "ABX9TyMkpFu76PNnT+6SbFaV4kgp"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "### Basics"
      ],
      "metadata": {
        "id": "zJCGKFCLr4NG"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Requirements"
      ],
      "metadata": {
        "id": "skTWg_B4slRK"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install -U sentence_transformer\n",
        "from sentence_transformers import SentenceTransformer\n",
        "from sklearn.metrics.pairwise import cosine_similarity\n",
        "from sklearn.preprocessing import normalize\n",
        "from google.colab import drive\n",
        "from tqdm.notebook import tqdm\n",
        "from tqdm import tqdm\n",
        "import pandas as pd\n",
        "import re\n",
        "import\n",
        "\n",
        "# Mount Google Drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "# Specify the full path to the CSV file in Google Drive\n",
        "file_path = '/content/drive/MyDrive/combined_data.csv'\n",
        "\n",
        "# Load the data\n",
        "data = pd.read_csv(file_path)"
      ],
      "metadata": {
        "id": "Bj-oyAlksrvy"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Pre-Processing"
      ],
      "metadata": {
        "id": "jMLdgbP5thvp"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def clean_text(text):\n",
        "    # Customize this function with your preferred cleaning techniques\n",
        "    text = text.lower()  # Convert to lowercase\n",
        "    text = re.sub(r\"[^a-zA-Z0-9\\s]\", \"\", text)  # Remove punctuation and special characters\n",
        "    text = re.sub(r\"\\s+\", \" \", text)  # Remove extra whitespace\n",
        "    # ... Add other cleaning steps as needed ...\n",
        "    return text\n",
        "def clean_text(text):\n",
        "    # Your text cleaning/preprocessing logic here\n",
        "    return processed_text"
      ],
      "metadata": {
        "id": "P2HX7a83toQX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Model 1 - stsb-distilbert-base"
      ],
      "metadata": {
        "id": "EgV0m5FftxA3"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Load a different sentence embedding model\n",
        "model = SentenceTransformer('all-mpnet-base-v2')\n",
        "\n",
        "# Encode plot descriptions to vectors\n",
        "plot_descriptions = data['plot'].tolist()\n",
        "\n",
        "# Set batch size\n",
        "batch_size = 32\n",
        "\n",
        "# Initialize an empty list to store embeddings\n",
        "plot_embeddings = []\n",
        "\n",
        "# Use tqdm to display progress bar\n",
        "for i in tqdm(range(0, len(plot_descriptions), batch_size), desc='Encoding plot descriptions'):\n",
        "    batch = plot_descriptions[i:i + batch_size]\n",
        "    embeddings_batch = model.encode(batch)\n",
        "    plot_embeddings.extend(embeddings_batch)\n",
        "\n",
        "# Add the embeddings to the DataFrame\n",
        "data['plot_embedding'] = plot_embeddings\n",
        "\n",
        "def recommend_movies(user_input, num_recommendations=5):\n",
        "    # Encode user input\n",
        "    user_embedding = model.encode([user_input])[0]\n",
        "\n",
        "    # Compute cosine similarity between user input and plot descriptions\n",
        "    similarities = cosine_similarity([user_embedding], plot_embeddings)[0]\n",
        "\n",
        "    # Get indices of the most similar movies\n",
        "    indices = similarities.argsort()[-num_recommendations:][::-1]\n",
        "\n",
        "    # Display recommended movies\n",
        "    recommendations = data.loc[indices, ['title', 'plot', 'image']]\n",
        "    return recommendations\n"
      ],
      "metadata": {
        "id": "Qv8LQLhStyjx"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# User input\n",
        "user_input = input(\"Describe the kind of movie you want to watch: \")\n",
        "\n",
        "# Get recommendations based on user input\n",
        "recommendations = recommend_movies(user_input)\n",
        "\n",
        "# Display recommended movies with images\n",
        "print(\"\\nRecommended Movies:\")\n",
        "for i, (title, plot, image) in recommendations.iterrows():\n",
        "    print(f\"\\nTitle: {title}\\nPlot: {plot}\\nImage: {image}\\n\")\n",
        "\n",
        "    # Display image using the provided link\n",
        "    from IPython.display import Image, display\n",
        "    display(Image(url=image))\n"
      ],
      "metadata": {
        "id": "BCKTH2h2uQE4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Sample query: person starts a company and gets rich"
      ],
      "metadata": {
        "id": "HryYcVkZJGbv"
      }
    }
  ]
}