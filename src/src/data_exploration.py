"""
data_exploration.py
-------------------
Dataset: TMDB 5000 Movies Dataset
Source: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
Description:
    This dataset contains metadata for ~5000 movies from The Movie Database (TMDB).
    It includes information such as budget, revenue, genres, runtime, popularity,
    vote average, and more.

Important Columns:
    - budget:        Production budget in USD
    - revenue:       Box office revenue in USD (TARGET column)
    - popularity:    TMDB popularity score (based on views, votes, watchlists)
    - runtime:       Movie duration in minutes
    - vote_average:  Average user rating (0–10)
    - vote_count:    Number of user votes
    - release_date:  Date the movie was released
    - title:         Movie title

ML Task: REGRESSION
    We will predict a movie's revenue based on budget, popularity, runtime,
    vote average, and vote count.
"""

import pandas as pd
import numpy as np

def explore_data(filepath="data/dataset.csv"):
    # ── Pandas Task 1: Load dataset ──────────────────────────────────────────
    df = pd.read_csv(filepath)
    print("=" * 60)
    print("DATASET EXPLORATION")
    print("=" * 60)

    # ── Pandas Task 2: Display first rows ────────────────────────────────────
    print("\n[1] First 5 rows:")
    print(df.head())

    # ── Pandas Task 3: Dataset info ──────────────────────────────────────────
    print("\n[2] Dataset info:")
    df.info()

    # ── Pandas Task 4: Summary statistics ────────────────────────────────────
    print("\n[3] Summary statistics:")
    print(df.describe())

    # ── Pandas Task 5: Check for missing values ───────────────────────────────
    print("\n[4] Missing values per column:")
    print(df.isnull().sum())

    # ── Pandas Task 7: Select specific columns ────────────────────────────────
    key_cols = ["title", "budget", "revenue", "popularity", "runtime",
                "vote_average", "vote_count"]
    available = [c for c in key_cols if c in df.columns]
    print(f"\n[5] Key columns selected:\n{df[available].head()}")

    # ── Pandas Task 8: Filter rows with positive budget AND revenue ───────────
    df_filtered = df[(df["budget"] > 0) & (df["revenue"] > 0)]
    print(f"\n[6] Rows with budget>0 and revenue>0: {len(df_filtered)} "
          f"(out of {len(df)})")

    # ── Pandas Task 9: Sort by revenue descending ─────────────────────────────
    print("\n[7] Top 5 highest-revenue movies:")
    print(df_filtered.sort_values("revenue", ascending=False)[
        ["title", "revenue", "budget"]].head())

    # ── Pandas Task 10: Group by release year ─────────────────────────────────
    if "release_date" in df.columns:
        df_filtered = df_filtered.copy()
        df_filtered["release_year"] = pd.to_datetime(
            df_filtered["release_date"], errors="coerce").dt.year
        print("\n[8] Average revenue by decade (grouped):")
        df_filtered["decade"] = (df_filtered["release_year"] // 10) * 10
        print(df_filtered.groupby("decade")["revenue"].mean()
              .apply(lambda x: f"${x:,.0f}"))

    # ── NumPy Task 1: Basic stats with NumPy ──────────────────────────────────
    rev = df_filtered["revenue"].values
    print(f"\n[NumPy] Revenue — mean: ${np.mean(rev):,.0f}, "
          f"median: ${np.median(rev):,.0f}, std: ${np.std(rev):,.0f}")

    # ── NumPy Task 2: Log transform check ────────────────────────────────────
    log_rev = np.log1p(rev)
    print(f"[NumPy] Log-revenue — mean: {np.mean(log_rev):.4f}, "
          f"std: {np.std(log_rev):.4f}")

    print("\n[Exploration complete]")
    return df

if __name__ == "__main__":
    explore_data()
