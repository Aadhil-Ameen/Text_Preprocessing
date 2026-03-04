# ==========================================================
# SAFE TWITTER DATASET LOADING + PREPROCESSING
# ==========================================================

import pandas as pd
import re
import string
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if not already downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')


# ==========================================================v
# 1️⃣ LOAD DATASET SAFELY
# ==========================================================

# After running the upload cell, make sure 'Twitter_Data.csv' appears in the
# file browser (left sidebar -> folder icon). Then, you can re-run this cell.

df = pd.read_csv(
    "Twitter_Data.csv",
    sep=",",              # because your file is COMMA separated
    engine="python",       # handles complex rows
    quoting=csv.QUOTE_NONE,
    on_bad_lines="skip"    # skips broken rows
)

# Clean column names
df.columns = df.columns.str.strip()

print("Columns:", df.columns.tolist())
print(df.head())


# ==========================================================
# 2️⃣ MATCH YOUR ACTUAL COLUMN NAMES
# ==========================================================

TEXT_COLUMN = "clean_text"
LABEL_COLUMN = "category"

# Convert sentiment safely
df[LABEL_COLUMN] = pd.to_numeric(
    df[LABEL_COLUMN],
    errors="coerce"
).fillna(0).astype(int)


# ==========================================================
# 3️⃣ TEXT PREPROCESSING
# ==========================================================

stopwords = set(stopwords.words('english')) # Using NLTK stopwords

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\x00-\x7F]+', ' ', text) # Ensure consistent regex for non-ASCII removal
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = " ".join(text.split())
    words = text.split()
    words = [w for w in words if w not in stopwords]
    return " ".join(words)

df["processed_text"] = df[TEXT_COLUMN].apply(preprocess_text)

print("\nProcessed Sample:")
print(df[["clean_text", "processed_text"]].head())


# ==========================================================
# 4️⃣ SIMILARITY (OPTIONAL)
# ==========================================================

def jaccard_similarity(text1, text2):
    words1 = set(text1.split())
    words2 = set(text2.split())
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    if union == 0:
        return 0.0
    return intersection / union

vectorizer = TfidfVectorizer(max_features=3000)
tfidf_matrix = vectorizer.fit_transform(df["processed_text"][:4]) # Changed to first 4 tweets

cosine_score_1_2 = cosine_similarity(
    tfidf_matrix[0:1],
    tfidf_matrix[1:2]
)[0][0]

cosine_score_1_4 = cosine_similarity(
    tfidf_matrix[0:1],
    tfidf_matrix[3:4]
)[0][0] # Changed similarity between first and fourth tweet

jaccard_score_1_2 = jaccard_similarity(df["processed_text"][0], df["processed_text"][1])
jaccard_score_1_4 = jaccard_similarity(df["processed_text"][0], df["processed_text"][3]) # Changed similarity between first and fourth tweet

print("==================================================")
print("Cosine Similarity (Tweet 1 and 2):", round(cosine_score_1_2, 4))
print("Cosine Similarity (Tweet 1 to 4):", round(cosine_score_1_4, 4))
print("Jaccard Similarity (Tweet 1 and 2):", round(jaccard_score_1_2, 4))
print("Jaccard Similarity (Tweet 1 to 4):", round(jaccard_score_1_4, 4))
print("==================================================")


# ==========================================================
# 5️⃣ SAVE CLEANED FILE
# ==========================================================

df.to_csv("twitter_processed_output.csv", index=False)

print("\nProcessed dataset saved successfully!")