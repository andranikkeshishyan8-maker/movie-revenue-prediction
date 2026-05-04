""
visualization.py
----------------
Creates and saves Matplotlib and Plotly visualizations for the TMDB movies dataset.

Matplotlib plots (saved as .png):
    1. Bar chart  — Average revenue by decade
    2. Histogram  — Distribution of movie budgets
    3. Scatter    — Budget vs Revenue (helps explain the ML problem)

Plotly plots (saved as .html):
    1. Interactive scatter — Budget vs Revenue colored by rating
    2. Interactive bar     — Top 20 highest-grossing movies
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

os.makedirs("outputs/plots", exist_ok=True)


# ═══════════════════════════════════════════════════════
#  MATPLOTLIB VISUALIZATIONS
# ═══════════════════════════════════════════════════════

def plot_revenue_by_decade(df):
    """Bar chart — Average revenue by decade (Matplotlib plot 1)."""
    if "release_date" not in df.columns and "release_year" not in df.columns:
        print("Skipping decade plot — no release date column.")
        return

    tmp = df.copy()
    if "release_year" not in tmp.columns:
        tmp["release_year"] = pd.to_datetime(
            tmp["release_date"], errors="coerce").dt.year
    tmp = tmp[(tmp["budget"] > 0) & (tmp["revenue"] > 0)].copy()
    tmp["decade"] = (tmp["release_year"] // 10) * 10
    decade_avg = tmp.groupby("decade")["revenue"].mean() / 1_000_000

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(decade_avg.index.astype(str), decade_avg.values,
           color="#E50914", edgecolor="black", width=0.6)
    ax.set_title("Average Box Office Revenue by Decade", fontsize=15, fontweight="bold")
    ax.set_xlabel("Decade", fontsize=12)
    ax.set_ylabel("Average Revenue (Millions USD)", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/plots/plot_1_revenue_by_decade.png", dpi=150)
    plt.close()
    print("Saved: outputs/plots/plot_1_revenue_by_decade.png")


def plot_budget_distribution(df):
    """Histogram — Distribution of movie budgets (Matplotlib plot 2)."""
    tmp = df[df["budget"] > 0].copy()
    budgets_m = tmp["budget"] / 1_000_000

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(budgets_m, bins=40, color="#221F1F", edgecolor="#E50914", alpha=0.85)
    ax.set_title("Distribution of Movie Production Budgets", fontsize=15, fontweight="bold")
    ax.set_xlabel("Budget (Millions USD)", fontsize=12)
    ax.set_ylabel("Number of Movies", fontsize=12)
    ax.axvline(budgets_m.median(), color="#E50914", linestyle="--",
               linewidth=2, label=f"Median: ${budgets_m.median():.1f}M")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/plot_2_budget_histogram.png", dpi=150)
    plt.close()
    print("Saved: outputs/plots/plot_2_budget_histogram.png")


def plot_budget_vs_revenue(df):
    """Scatter — Budget vs Revenue (Matplotlib plot 3 — explains ML problem)."""
    tmp = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()
    x = tmp["budget"] / 1_000_000
    y = tmp["revenue"] / 1_000_000

    # ── NumPy Task: Compute correlation coefficient ───────────────────────────
    corr = np.corrcoef(x, y)[0, 1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, alpha=0.4, color="#E50914", edgecolors="none", s=25)
    ax.set_title(f"Budget vs Revenue\n(Pearson r = {corr:.3f})",
                 fontsize=15, fontweight="bold")
    ax.set_xlabel("Production Budget (Millions USD)", fontsize=12)
    ax.set_ylabel("Box Office Revenue (Millions USD)", fontsize=12)

    # Trend line using NumPy polyfit
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, m * x_line + b, color="black", linewidth=2,
            linestyle="--", label=f"Trend: y = {m:.2f}x + {b:.1f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/plot_3_budget_vs_revenue.png", dpi=150)
    plt.close()
    print("Saved: outputs/plots/plot_3_budget_vs_revenue.png")


# ═══════════════════════════════════════════════════════
#  PLOTLY VISUALIZATIONS
# ═══════════════════════════════════════════════════════

def plotly_budget_revenue_scatter(df):
    """Interactive scatter — Budget vs Revenue colored by vote rating (Plotly 1)."""
    tmp = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()
    rating_col = "vote_average" if "vote_average" in tmp.columns else "rating"
    if rating_col not in tmp.columns:
        tmp["rating_display"] = 5.0
        rating_col = "rating_display"
    tmp["budget_m"] = tmp["budget"] / 1_000_000
    tmp["revenue_m"] = tmp["revenue"] / 1_000_000
    title_col = "title" if "title" in tmp.columns else None

    fig = px.scatter(
        tmp,
        x="budget_m",
        y="revenue_m",
        color=rating_col,
        hover_name=title_col,
        color_continuous_scale="RdYlGn",
        labels={
            "budget_m": "Budget (Millions USD)",
            "revenue_m": "Revenue (Millions USD)",
            rating_col: "Rating"
        },
        title="Interactive: Movie Budget vs Revenue (colored by Rating)",
        template="plotly_dark",
        opacity=0.7
    )
    fig.update_traces(marker=dict(size=6))
    fig.write_html("outputs/plots/plot_4_interactive_budget_revenue.html")
    print("Saved: outputs/plots/plot_4_interactive_budget_revenue.html")


def plotly_top_movies_bar(df):
    """Interactive bar chart — Top 20 highest-grossing movies (Plotly 2)."""
    tmp = df[df["revenue"] > 0].copy()
    title_col = "title" if "title" in tmp.columns else None
    if title_col is None:
        print("Skipping top movies bar — no title column.")
        return

    top20 = tmp.nlargest(20, "revenue")[["title", "revenue", "budget"]].copy()
    top20["revenue_m"] = top20["revenue"] / 1_000_000
    top20["budget_m"] = top20["budget"] / 1_000_000
    top20 = top20.sort_values("revenue_m")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top20["title"],
        x=top20["revenue_m"],
        orientation="h",
        name="Revenue",
        marker_color="#E50914"
    ))
    fig.add_trace(go.Bar(
        y=top20["title"],
        x=top20["budget_m"],
        orientation="h",
        name="Budget",
        marker_color="#564d4d"
    ))
    fig.update_layout(
        barmode="overlay",
        title="Top 20 Highest-Grossing Movies: Revenue vs Budget",
        xaxis_title="Amount (Millions USD)",
        yaxis_title="Movie",
        template="plotly_dark",
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    fig.write_html("outputs/plots/plot_5_top20_movies.html")
    print("Saved: outputs/plots/plot_5_top20_movies.html")


def create_all_visualizations(df):
    print("\n[Generating visualizations...]")
    plot_revenue_by_decade(df)
    plot_budget_distribution(df)
    plot_budget_vs_revenue(df)
    plotly_budget_revenue_scatter(df)
    plotly_top_movies_bar(df)
    print("[All visualizations saved]\n")


if __name__ == "__main__":
    df = pd.read_csv("data/dataset.csv")
    create_all_visualizations(df)
