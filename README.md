# ForTheMovie
An advanced content-based movie recommender system using TF-IDF and cosine similarity with support for ratings, tags, search, filters, and sorting via an interactive Streamlit interface.

# 🎬 Advanced Movie Recommender System

## Overview

This project is an **AI-powered movie recommendation engine** built with Streamlit, using **TF-IDF vectorization**, **cosine similarity**, and optional **tags and ratings** for intelligent, content-based filtering. It allows users to search movies interactively, filter results by year, and sort recommendations based on similarity, rating, or genre — all inside a modern UI.

## Project Objective

Movie recommendation systems today either rely heavily on collaborative filtering or offer limited user control. This project aims to:

- Deliver **smart recommendations** based on genres, titles, and user tags.
- Offer **filtering tools** for users to control time range and recommendation count.
- Prioritize **high-quality recommendations** using rating data when available.
- Provide a responsive and interactive **Streamlit interface**.
- Serve as a base for further improvements like hybrid models or collaborative filtering.

## Core Technologies Used

| Technology           | Purpose                                                       |
|----------------------|---------------------------------------------------------------|
| Streamlit            | Web-based interactive front-end                               |
| TF-IDF + Cosine Sim. | Core content-based similarity logic                           |
| Pandas & NumPy       | Data manipulation and computation                             |
| scikit-learn         | TF-IDF vectorization, similarity scoring                      |
| Python Regex         | Smart parsing of movie titles and detecting sequels           |

## Project Structure

movie-recommender/
│
├── movie_recommender.py # Main Streamlit application
├── requirements.txt # List of required Python packages
├── movies.csv # Required dataset (MovieLens)
├── ratings.csv # Optional (for rating-aware sorting)
├── tags.csv # Optional (for tag-enhanced similarity)
└── README.md # Project documentation


## Installation

Install the dependencies:

```bash
pip install streamlit pandas numpy scikit-learn

Place the following dataset files (from MovieLens) in the same directory:

movies.csv (required)

ratings.csv (optional but recommended)

tags.csv (optional)

How It Works
1. Preprocessing
Cleans movie titles and extracts release years.

Replaces missing genres or tags.

Merges tags and average ratings (if present).

Constructs a combined_features string for each movie.

2. TF-IDF + Cosine Similarity
Converts all combined_features into a TF-IDF matrix.

Applies cosine similarity to generate pairwise similarity scores.

3. Recommendation Logic
Matches input movie with exact or partial title.

Sorts results based on selected criteria: similarity, rating, or genre.

Filters by user-defined year range and recommendation count.

Displays annotated results with similarity, rating, year, and genre.

| Parameter                | Description                               |
| ------------------------ | ----------------------------------------- |
| `combined_features`      | String of genres + title + tags per movie |
| `num_recommendations`    | How many similar movies to return         |
| `start_year`, `end_year` | Year filter range                         |
| `sort_by`                | Can be `similarity`, `rating`, or `genre` |

How to Use
bash
Copy
Edit
streamlit run movie_recommender.py
Search for a movie name (e.g., Batman, Inception).

Select it from the dropdown.

Set the number of recommendations.

Optionally adjust the year range filter.

Choose how to sort results.

Click "Get Recommendations" and explore!

Output
A ranked list of movies with:

🎯 Match percentage

🎭 Genres

📅 Year

⭐ Average rating (if available)

Expandable UI cards with similarity progress bars.

Supports filtering, search, and control in real time.

Planned Features
Hybrid Recommender System
Integrate collaborative filtering using user-based or matrix factorization models.

Genre + Tag Specific Filtering
Let users restrict recommendations to specific genres or tags only.

Export Recommendations
Allow users to export results as CSV or shareable links.

Movie Poster Integration
Fetch posters using IMDb or TMDB APIs for a visual upgrade.

Dashboard Analytics
Track most-searched movies, common filters, user preferences.

License
This project is open-source under the MIT License. You are free to use, modify, and distribute with attribution.

Contribution Guidelines
We welcome your contributions!

Submit issues or feature suggestions

Fork and raise pull requests

Improve UI or performance

Help integrate collaborative filtering models

Contact and Credits
Author: Swapnendu Sikdar
Email: swapnendusikdar@gmail.com
Institution: Jadavpur University
Field: Electrical Engineering
Interests: AI/ML, Deep Learning, Machine Learning, Computer Vision, Smart Infrastructure

Acknowledgments
MovieLens Dataset
Streamlit
scikit-learn
