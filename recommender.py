"""
Content-Based Movie Recommendation System
-------------------------------------------
Core engine: converts each movie's genres + description into a TF-IDF
vector, then uses cosine similarity to find movies most similar to a
given movie (or to a user's average taste across several liked movies).

Author: <your name>
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self, csv_path="movies.csv"):
        self.df = pd.read_csv(csv_path)
        self._build_model()

    def _build_model(self):
        # Combine genres (weighted by repeating) + description into one
        # "content soup" per movie so TF-IDF captures both taste signals.
        genres_text = self.df["genres"].str.replace("|", " ", regex=False)
        self.df["content"] = (genres_text + " ") * 2 + self.df["description"]

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["content"])

        # Precompute full similarity matrix (fine for small/medium datasets)
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

        self.title_to_index = pd.Series(
            self.df.index, index=self.df["title"].str.lower()
        )

    def _get_index(self, title):
        title = title.lower().strip()
        if title not in self.title_to_index:
            matches = self.df[self.df["title"].str.lower().str.contains(title)]
            if matches.empty:
                raise ValueError(f"No movie found matching '{title}'")
            return matches.index[0]
        return self.title_to_index[title]

    def recommend_similar(self, title, top_n=5):
        """Recommend movies similar to a single given movie."""
        idx = self._get_index(title)
        scores = list(enumerate(self.similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:top_n]

        results = self.df.iloc[[i for i, _ in scores]].copy()
        results["similarity"] = [round(s, 3) for _, s in scores]
        return results[["title", "year", "genres", "rating", "similarity"]]

    def recommend_for_profile(self, liked_titles, top_n=5):
        """Recommend movies based on a user's average taste across
        several liked movies (simple user-profile based filtering)."""
        indices = [self._get_index(t) for t in liked_titles]
        profile_vector = np.asarray(self.tfidf_matrix[indices].mean(axis=0))

        scores = cosine_similarity(profile_vector, self.tfidf_matrix).flatten()
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        ranked = [r for r in ranked if r[0] not in indices][:top_n]

        results = self.df.iloc[[i for i, _ in ranked]].copy()
        results["similarity"] = [round(s, 3) for _, s in ranked]
        return results[["title", "year", "genres", "rating", "similarity"]]

    def top_rated(self, genre=None, top_n=5):
        """Simple popularity/rating-based fallback recommender."""
        data = self.df
        if genre:
            data = data[data["genres"].str.contains(genre, case=False)]
        return data.sort_values("rating", ascending=False).head(top_n)[
            ["title", "year", "genres", "rating"]
        ]


if __name__ == "__main__":
    rec = MovieRecommender()
    print("Movies similar to 'Iron Circuit':")
    print(rec.recommend_similar("Iron Circuit", top_n=5))
