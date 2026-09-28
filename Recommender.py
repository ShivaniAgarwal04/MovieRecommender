from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "ml-100k"


# ============================================================
# GLOBAL VARIABLES
# ============================================================

_ratings = None
_movies = None

_user_item = None
_user_similarity = None

_tfidf = None
_content_similarity = None


# ============================================================
# LOAD DATASET
# ============================================================

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load MovieLens ratings and movie metadata.
    """

    global _ratings, _movies

    ratings_file = DATASET_DIR / "u.data"
    movies_file = DATASET_DIR / "u.item"

    if not ratings_file.exists() or not movies_file.exists():
        raise FileNotFoundError(
            "MovieLens dataset not found.\n"
            "Please run: python generate_dataset.py"
        )

    # -----------------------------
    # Load ratings
    # -----------------------------

    if _ratings is None:

        _ratings = pd.read_csv(
            ratings_file,
            sep="\t",
            names=[
                "userId",
                "movieId",
                "rating",
                "timestamp"
            ],
            encoding="utf-8"
        )

    # -----------------------------
    # Load movie metadata
    # -----------------------------

    if _movies is None:

        _movies = pd.read_csv(
            movies_file,
            sep="|",
            header=None,
            encoding="latin-1",
            names=[
                "movieId",
                "title",
                "releaseDate",
                "videoReleaseDate",
                "imdbUrl",
                "unknown",
                "Action",
                "Adventure",
                "Animation",
                "Children",
                "Comedy",
                "Crime",
                "Documentary",
                "Drama",
                "Fantasy",
                "FilmNoir",
                "Horror",
                "Musical",
                "Mystery",
                "Romance",
                "SciFi",
                "Thriller",
                "War",
                "Western"
            ]
        )

    return _ratings, _movies


# ============================================================
# USER-ITEM MATRIX
# ============================================================

def build_user_item_matrix(
    ratings_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create User × Movie rating matrix.
    """

    matrix = ratings_df.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        fill_value=0
    )

    return matrix.astype(np.float32)


# ============================================================
# INITIALIZATION
# ============================================================

def _initialize():

    global _user_item
    global _user_similarity
    global _tfidf
    global _content_similarity

    ratings, movies = load_data()

    # ========================================================
    # COLLABORATIVE FILTERING
    # ========================================================

    if _user_item is None:

        print("Building user-item matrix...")

        _user_item = build_user_item_matrix(ratings)

        print("Calculating user similarity...")

        _user_similarity = cosine_similarity(
            _user_item.values
        )

    # ========================================================
    # CONTENT-BASED FILTERING
    # ========================================================

    if _tfidf is None:

        print("Building TF-IDF movie representation...")

        genre_cols = [
            "unknown",
            "Action",
            "Adventure",
            "Animation",
            "Children",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "FilmNoir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "SciFi",
            "Thriller",
            "War",
            "Western"
        ]

        # Convert genre indicators into text
        genre_text = (
            movies[genre_cols]
            .fillna(0)
            .astype(str)
            .agg(" ".join, axis=1)
        )

        # Movie title + genre information
        content = (
            movies["title"].fillna("").astype(str)
            + " "
            + genre_text
        )

        _tfidf = TfidfVectorizer(
            stop_words="english"
        )

        tfidf_matrix = _tfidf.fit_transform(content)

        print("Calculating movie similarity...")

        _content_similarity = cosine_similarity(
            tfidf_matrix
        )


# ============================================================
# COLLABORATIVE FILTERING RECOMMENDER
# ============================================================

def recommend_cf(
    user_id: int,
    n: int = 5
) -> List[Tuple[str, float]]:
    """
    Recommend movies using user-based collaborative filtering.
    """

    _initialize()

    ratings, movies = load_data()

    # -----------------------------------------
    # Validate user
    # -----------------------------------------

    if user_id not in _user_item.index:
        return []

    # -----------------------------------------
    # User index
    # -----------------------------------------

    user_idx = _user_item.index.get_loc(user_id)

    # -----------------------------------------
    # Similarity with other users
    # -----------------------------------------

    similarities = _user_similarity[user_idx].copy()

    # Do not compare user with themselves
    similarities[user_idx] = 0

    # -----------------------------------------
    # Calculate predicted scores
    # -----------------------------------------

    denominator = (
        np.sum(np.abs(similarities))
        + 1e-8
    )

    scores = (
        similarities @ _user_item.values
    ) / denominator

    # -----------------------------------------
    # Remove movies already watched/rated
    # -----------------------------------------

    rated_movies = _user_item.loc[user_id].values

    scores[rated_movies > 0] = -np.inf

    # -----------------------------------------
    # Get top N
    # -----------------------------------------

    top_indices = np.argsort(scores)[::-1]

    top_indices = [
        idx
        for idx in top_indices
        if np.isfinite(scores[idx])
    ][:n]

    # -----------------------------------------
    # Movie ID mapping
    # -----------------------------------------

    movie_ids = (
        _user_item.columns
        .to_numpy()[top_indices]
    )

    movie_map = (
        movies
        .set_index("movieId")["title"]
    )

    recommendations = []

    for idx, movie_id in zip(
        top_indices,
        movie_ids
    ):

        title = movie_map.get(
            int(movie_id),
            "Unknown"
        )

        score = float(scores[idx])

        recommendations.append(
            (title, score)
        )

    return recommendations


# ============================================================
# CONTENT-BASED RECOMMENDER
# ============================================================

def recommend_content(
    movie_id: int,
    n: int = 5
) -> List[Tuple[str, float]]:
    """
    Recommend movies similar to a selected movie
    using TF-IDF and cosine similarity.
    """

    _initialize()

    _, movies = load_data()

    # -----------------------------------------
    # Find movie
    # -----------------------------------------

    matches = movies.index[
        movies["movieId"] == movie_id
    ].tolist()

    if not matches:
        return []

    movie_index = matches[0]

    # -----------------------------------------
    # Similarity scores
    # -----------------------------------------

    similarities = (
        _content_similarity[movie_index]
        .copy()
    )

    # Remove selected movie itself
    similarities[movie_index] = -np.inf

    # -----------------------------------------
    # Top N similar movies
    # -----------------------------------------

    indices = np.argsort(
        similarities
    )[::-1][:n]

    recommendations = []

    for idx in indices:

        if np.isfinite(similarities[idx]):

            title = str(
                movies.iloc[idx]["title"]
            )

            score = float(
                similarities[idx]
            )

            recommendations.append(
                (title, score)
            )

    return recommendations


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n==============================")
    print("COLLABORATIVE FILTERING")
    print("==============================")

    cf_results = recommend_cf(
        user_id=1,
        n=5
    )

    for rank, (title, score) in enumerate(
        cf_results,
        start=1
    ):
        print(
            f"{rank}. {title} "
            f"(Score: {score:.4f})"
        )

    print("\n==============================")
    print("CONTENT-BASED FILTERING")
    print("==============================")

    content_results = recommend_content(
        movie_id=1,
        n=5
    )

    for rank, (title, score) in enumerate(
        content_results,
        start=1
    ):
        print(
            f"{rank}. {title} "
            f"(Similarity: {score:.4f})"
        )
