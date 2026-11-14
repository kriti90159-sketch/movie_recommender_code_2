# Movie Recommendation System (Content-Based Filtering)

A simple content-based movie recommender built with Python, pandas, and
scikit-learn. It recommends movies using **TF-IDF vectorization** on genres
and descriptions, ranked by **cosine similarity**.

## Project Structure
```
movie_recommender/
├── data/
│   └── movies.csv        # Dataset of 5,000 movies (procedurally generated)
├── recommender.py         # Core recommendation engine (MovieRecommender class)
├── main.py                 # Console/CLI interface
├── app.py                  # Streamlit GUI interface
└── README.md
```

## Setup
```bash
pip install pandas scikit-learn streamlit
```

## Run (Console version)
```bash
python main.py
```

## Run (GUI version)
```bash
streamlit run app.py
```

## Features
1. **Similar Movies** – enter a movie you liked, get similar ones.
2. **Taste Profile** – enter multiple movies you liked, get personalized picks.
3. **Top Rated** – browse top rated movies, optionally filtered by genre.

## How It Works
1. Each movie's genres + description are combined into a "content" string.
2. `TfidfVectorizer` converts this text into numeric vectors, weighting
   words by how distinctive they are across the dataset.
3. `cosine_similarity` measures how close two movies' vectors are (closer
   direction = more similar taste).
4. For a single movie, we rank all other movies by similarity to it.
   For a taste profile, we average the vectors of the liked movies first,
   then rank against that average.

## Possible Extensions
- Swap in a real dataset (e.g., MovieLens 100k) for more movies.
- Add collaborative filtering using real user ratings.
- Add a hybrid model combining content-based + collaborative scores.
