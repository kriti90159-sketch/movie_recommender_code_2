"""
Streamlit GUI for the Movie Recommendation System.
Run: streamlit run app.py

Rebuilt to handle a large catalog (~5000 movies): dropdowns are replaced with
a type-to-search box so the UI stays fast and usable at scale.
"""

import streamlit as st
from recommender import MovieRecommender

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")


@st.cache_resource
def load_recommender():
    return MovieRecommender()


rec = load_recommender()
all_titles = rec.df["title"].tolist()


def search_titles(query, limit=25):
    """Simple case-insensitive substring search over the catalog."""
    if not query:
        return []
    q = query.lower().strip()
    matches = [t for t in all_titles if q in t.lower()]
    return matches[:limit]


st.title("🎬 Movie Recommendation System")
st.caption(f"Content-based recommender (TF-IDF + Cosine Similarity) · {len(all_titles):,} movies in catalog")

tab1, tab2, tab3 = st.tabs(["🎯 Similar Movies", "🧑 My Taste Profile", "🏆 Top Rated"])

# ------------------------------------------------------------------
# TAB 1: Similar to a single movie
# ------------------------------------------------------------------
with tab1:
    st.subheader("Find movies similar to one you liked")
    query = st.text_input("Type a movie title to search", key="q1", placeholder="e.g. Silent Lighthouse")
    results = search_titles(query)

    if query and not results:
        st.warning("No movies matched that search. Try a different keyword.")
    elif results:
        selected = st.selectbox(f"Matches ({len(results)} shown, max 25)", results, key="s1")
        top_n = st.slider("Number of recommendations", 3, 15, 5, key="n1")
        if st.button("Recommend", key="btn1"):
            out = rec.recommend_similar(selected, top_n=top_n)
            st.dataframe(out, use_container_width=True, hide_index=True)
    else:
        st.info("Start typing above to search the catalog.")

# ------------------------------------------------------------------
# TAB 2: Taste profile from multiple liked movies
# ------------------------------------------------------------------
with tab2:
    st.subheader("Recommend based on multiple liked movies")
    query2 = st.text_input("Search and add movies you liked", key="q2", placeholder="e.g. Kingdom of Cairo")
    results2 = search_titles(query2)

    if "liked_movies" not in st.session_state:
        st.session_state.liked_movies = []

    if results2:
        pick = st.selectbox(f"Matches ({len(results2)} shown, max 25)", results2, key="s2")
        if st.button("➕ Add to my list", key="add2"):
            if pick not in st.session_state.liked_movies:
                st.session_state.liked_movies.append(pick)

    if st.session_state.liked_movies:
        st.write("**Your liked movies:**")
        for m in st.session_state.liked_movies:
            col1, col2 = st.columns([5, 1])
            col1.write(f"- {m}")
            if col2.button("✕", key=f"rm_{m}"):
                st.session_state.liked_movies.remove(m)
                st.rerun()

        top_n2 = st.slider("Number of recommendations", 3, 15, 5, key="n2")
        if st.button("Recommend for me", key="btn2"):
            out = rec.recommend_for_profile(st.session_state.liked_movies, top_n=top_n2)
            st.dataframe(out, use_container_width=True, hide_index=True)
    else:
        st.info("Search above and add a few movies you liked to build your taste profile.")

# ------------------------------------------------------------------
# TAB 3: Top rated, filterable by genre
# ------------------------------------------------------------------
with tab3:
    st.subheader("Top rated movies")
    genres = sorted(set(g for gs in rec.df["genres"] for g in gs.split("|")))
    genre_choice = st.selectbox("Filter by genre", ["All"] + genres, key="genre3")
    top_n3 = st.slider("Number of movies", 5, 25, 10, key="n3")
    genre_filter = None if genre_choice == "All" else genre_choice
    st.dataframe(
        rec.top_rated(genre=genre_filter, top_n=top_n3),
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.caption(
    f"Dataset: {len(all_titles):,} movies · Model: TF-IDF vectorization on genres + "
    "description, ranked by cosine similarity"
)
