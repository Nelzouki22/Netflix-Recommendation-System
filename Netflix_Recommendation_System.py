import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Install Kaggle package
!pip install kaggle

# Download the Netflix dataset from Kaggle
!kaggle datasets download -d shivamb/netflix-shows -p ./ --unzip

# Load dataset
df = pd.read_csv('./netflix_titles.csv')
df = df.dropna(subset=['description', 'title'])

# Feature extraction using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])

# Calculate cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 recommendations
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

# Test the recommendation system
print(get_recommendations('Breaking Bad'))

