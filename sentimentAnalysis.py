# %%
import pandas as pd
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score,accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



# %%

# Load the dataset
path = './training.1600000.processed.noemoticon.csv'

sample_size = 2000

# Read only the first num_samples rows from the dataset
data = pd.read_csv(path, encoding='latin-1', header=None, names=['target', 'id', 'date', 'flag', 'user', 'text'], nrows=sample_size, error_bad_lines=False, engine='python')


# %%
# Preprocess the dataset
data['text'] = data['text'].apply(lambda x: re.sub(r"http\S+|www\S+|@\S+", "", x))  # Remove URLs and mentions
data['text'] = data['text'].apply(lambda x: re.sub(r"\d+", "", x))  # Remove numbers
data['text'] = data['text'].apply(lambda x: re.sub(r"[^\w\s]", "", x))  # Remove special characters


# %%
# Create feature vectors using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'])


# %%

# Plot histogram of text lengths
text_lengths = data['text'].apply(len)
plt.hist(text_lengths, bins=50)
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.title('Histogram of Text Lengths')
plt.show()

# %%
# Apply k-Means clustering

#start timer
start_time = time.time()

kmeans = KMeans(n_clusters=2)
kmeans_labels = kmeans.fit_predict(X)

kmeans_training_time = time.time() - start_time

# %%
# Apply Agglomerative clustering

#start timer
start_time = time.time()

agglomerative = AgglomerativeClustering(n_clusters=2)
agglomerative_labels = agglomerative.fit_predict(X.toarray())

agglomerative_training_time = time.time() - start_time

# %%
# Evaluate the clustering results
kmeans_silhouette_score = silhouette_score(X, kmeans_labels)
agglomerative_silhouette_score = silhouette_score(X, agglomerative_labels)


# %%
# Calculate accuracy
kmeans_accuracy = accuracy_score(data['target'], kmeans_labels)
agglomerative_accuracy = accuracy_score(data['target'], agglomerative_labels)


# %%
# Print accuracy
print("k-Means Accuracy:", kmeans_accuracy)
print("Agglomerative Accuracy:", agglomerative_accuracy)


# %%
# Count the number of tweets in each cluster
kmeans_cluster_counts = pd.Series(kmeans_labels).value_counts()
agglomerative_cluster_counts = pd.Series(agglomerative_labels).value_counts()

# Create bar plot for cluster counts
plt.bar(kmeans_cluster_counts.index, kmeans_cluster_counts.values, label='k-Means')
plt.bar(agglomerative_cluster_counts.index, agglomerative_cluster_counts.values, label='Agglomerative')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.title('Number of Tweets in Each Cluster')
plt.legend()
plt.show()


# %%
# Print execution time
print("k-Means Training Time:", kmeans_training_time)
print("Agglomerative Training Time:", agglomerative_training_time)

# %%

# Print the evaluation scores
print("k-Means Silhouette Score:", kmeans_silhouette_score)
print("Agglomerative Silhouette Score:", agglomerative_silhouette_score)


# %%
# Print the difference in performance
difference = kmeans_silhouette_score - agglomerative_silhouette_score
print("Difference in Silhouette Scores:", difference)


