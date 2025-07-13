import pandas as pd  # Import pandas library for data manipulation and analysis
import streamlit as st  # Import Streamlit for creating web-based UI
from sklearn.feature_extraction.text import TfidfVectorizer  # Import TfidfVectorizer for converting text to numerical features
from sklearn.metrics.pairwise import cosine_similarity  # Import cosine_similarity for measuring similarity between movies
import re  # Import re for regular expression operations
import numpy as np  # Import numpy for numerical operations

# Load MovieLens dataset and related files
@st.cache_data  # Decorator to cache the data loading to improve performance
def load_data():
    # Load movies dataset from CSV file along with ratings and tags
    try:  # Try to load the dataset
        movies = pd.read_csv('movies.csv')  # Read movies.csv into a pandas DataFrame
        
        # Load ratings data for collaborative filtering enhancement
        try:
            ratings = pd.read_csv('ratings.csv')  # Load ratings data
        except FileNotFoundError:
            ratings = None
            st.warning("ratings.csv not found. Using content-based filtering only.")
        
        # Load tags data for enhanced recommendations
        try:
            tags = pd.read_csv('tags.csv')  # Load tags data
        except FileNotFoundError:
            tags = None
            st.warning("tags.csv not found. Tags will not be used for recommendations.")
        
        return movies, ratings, tags  # Return all loaded dataframes
    except FileNotFoundError:  # Handle case where movies.csv is not found
        st.error("Please download movies.csv from MovieLens dataset and place it in the same directory")  # Display error in Streamlit
        return None, None, None  # Return None if file is missing

# Extract year from movie title
def extract_year(title):
    # Extract year from movie title using regex
    match = re.search(r'\((\d{4})\)', title)
    if match:
        return int(match.group(1))  # Return year as integer
    return None  # Return None if no year found

# Clean movie titles
def clean_title(title):  # Function to clean movie titles
    # Remove year in parentheses and extra whitespace from movie titles
    cleaned = re.sub(r'\s*\(\d{4}\)\s*', '', title).strip()
    return cleaned  # Return the cleaned title

# Create display name for movies with sequels
def create_display_name(title):
    # Extract the base title and year
    year_match = re.search(r'\((\d{4})\)', title)
    base_title = re.sub(r'\s*\(\d{4}\)\s*', '', title).strip()
    year = year_match.group(1) if year_match else ""
    
    # Check if it's a sequel by looking for roman numerals or numbers
    sequel_patterns = [
        r'\b(II|III|IV|V|VI|VII|VIII|IX|X)\b',  # Roman numerals
        r'\b(2|3|4|5|6|7|8|9|10)\b',  # Numbers
        r'\b(Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten)\b',  # Written numbers
        r'\b(Part\s+\d+)\b',  # Part X
        r'\b(Volume\s+\d+)\b',  # Volume X
    ]
    
    sequel_found = False
    for pattern in sequel_patterns:
        if re.search(pattern, base_title, re.IGNORECASE):
            sequel_found = True
            break
    
    # If no sequel indicators found, check for common sequel words
    sequel_words = ['Returns', 'Revenge', 'Rise', 'Rises', 'Reloaded', 'Revolution', 'Resurrection']
    for word in sequel_words:
        if word in base_title:
            sequel_found = True
            break
    
    # Return appropriate display name
    if sequel_found:
        return f"{base_title} ({year})" if year else base_title
    else:
        return base_title

# Preprocess data with enhanced features
@st.cache_data  # Cache the preprocessing for better performance
def preprocess_data(movies, ratings, tags):  # Function to preprocess movie data
    # Convert movie genres, titles, and tags to TF-IDF vectors and compute cosine similarity
    # Make a copy to avoid modifying original dataframe
    movies_copy = movies.copy()
    
    # Extract years from movie titles
    movies_copy['year'] = movies_copy['title'].apply(extract_year)
    
    # Handle movies with no genres
    movies_copy['genres'] = movies_copy['genres'].replace('(no genres listed)', 'Unknown')
    
    # Replace '|' with space in genres for better TF-IDF processing
    movies_copy['genres'] = movies_copy['genres'].str.replace('|', ' ', regex=False)
    
    # Add cleaned title to feature combination
    movies_copy['cleaned_title'] = movies_copy['title'].apply(clean_title)
    
    # Add display name for better user experience
    movies_copy['display_name'] = movies_copy['title'].apply(create_display_name)
    
    # Process tags data if available
    if tags is not None:
        # Group tags by movieId and combine them
        movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
        movies_copy = movies_copy.merge(movie_tags, on='movieId', how='left')
        movies_copy['tag'] = movies_copy['tag'].fillna('')  # Fill NaN tags with empty string
    else:
        movies_copy['tag'] = ''  # Add empty tag column if no tags data
    
    # Combine genres, cleaned title, and tags for comprehensive similarity
    movies_copy['combined_features'] = (movies_copy['genres'] + ' ' + 
                                       movies_copy['cleaned_title'] + ' ' + 
                                       movies_copy['tag'])
    
    # Calculate average ratings if ratings data is available
    if ratings is not None:
        movie_ratings = ratings.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        movie_ratings.columns = ['movieId', 'avg_rating', 'rating_count']
        movies_copy = movies_copy.merge(movie_ratings, on='movieId', how='left')
        movies_copy['avg_rating'] = movies_copy['avg_rating'].fillna(0)
        movies_copy['rating_count'] = movies_copy['rating_count'].fillna(0)
    else:
        movies_copy['avg_rating'] = 0  # Default rating if no ratings data
        movies_copy['rating_count'] = 0  # Default count if no ratings data
    
    # Check if we have sufficient data for similarity computation
    if movies_copy['combined_features'].dropna().nunique() <= 1:
        st.error("Insufficient feature data to compute similarity.")
        return None, movies_copy
    
    # Create TF-IDF matrix from combined features
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)  # Limit features for performance
    tfidf_matrix = tfidf.fit_transform(movies_copy['combined_features'])
    
    # Compute cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return cosine_sim, movies_copy  # Return similarity matrix and processed movies

# Get movie details with enhanced information
def get_movie_details(idx, movies):  # Function to get movie details by index
    # Get comprehensive movie information by index
    title = movies['title'].iloc[idx]  # Get movie title
    display_name = movies['display_name'].iloc[idx] if 'display_name' in movies.columns else title
    genres = movies['genres'].iloc[idx]  # Get movie genres
    year = movies['year'].iloc[idx] if 'year' in movies.columns else None  # Get movie year
    avg_rating = movies['avg_rating'].iloc[idx] if 'avg_rating' in movies.columns else 0  # Get average rating
    rating_count = movies['rating_count'].iloc[idx] if 'rating_count' in movies.columns else 0  # Get rating count
    
    return display_name, genres, year, avg_rating, rating_count  # Return comprehensive details

# Enhanced recommendation function with multiple matching and filtering
def get_recommendations(title, movies, cosine_sim, num_recommendations=5, 
                       start_year=None, end_year=None, sort_by='similarity'):
    # Get movie recommendations with enhanced filtering and sorting options
    
    # First try exact match
    exact_match = movies[movies['title'].str.lower() == title.lower()]
    
    if not exact_match.empty:
        selected_movies = exact_match
    else:
        # Clean input title for partial matching
        cleaned_title = clean_title(title)
        
        # Find matching movies using partial match (case-insensitive)
        matching_movies = movies[movies['title'].str.contains(cleaned_title, case=False, na=False)]
        
        if matching_movies.empty:
            return [("Movie not found. Please select a valid title.", "", None, 0.0, 0, 0.0)]
        
        selected_movies = matching_movies.iloc[:1]  # Take first match
    
    # Get the index of selected movie
    idx = selected_movies.index[0]
    
    # Get similarity scores for all movies
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort by similarity score initially
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get all similar movies (including the selected one)
    all_recommendations = []
    for i, score in sim_scores:
        display_name, genres_rec, year_rec, avg_rating_rec, count_rec = get_movie_details(i, movies)
        
        # Apply year filtering if specified
        if start_year is not None and end_year is not None:
            if year_rec is None or year_rec < start_year or year_rec > end_year:
                continue
        
        # Convert similarity score to percentage (scale 0-100)
        similarity_percentage = score * 100
        
        all_recommendations.append((display_name, genres_rec, year_rec, avg_rating_rec, 
                                  count_rec, similarity_percentage))
    
    # Sort recommendations based on user preference
    if sort_by == 'rating':
        # Sort by average rating (descending) then by similarity
        all_recommendations = sorted(all_recommendations, 
                                   key=lambda x: (x[3], x[5]), reverse=True)
    elif sort_by == 'genre':
        # Sort by genre alphabetically then by similarity
        all_recommendations = sorted(all_recommendations, 
                                   key=lambda x: (x[1], x[5]), reverse=True)
    # Default is already sorted by similarity
    
    # Return top N recommendations
    return all_recommendations[:num_recommendations]

# Main function with enhanced UI
def main():
    # Main function to run the enhanced Streamlit app
    st.title("Advanced Movie Recommender System")
    st.write("Select a movie and customize your recommendations based on genres, ratings, tags, and years!")
    
    # Load data
    movies, ratings, tags = load_data()
    
    if movies is not None:
        # Display dataset information
        st.info(f"Loaded {len(movies)} movies from the dataset")
        if ratings is not None:
            st.info(f"Loaded {len(ratings)} ratings for enhanced recommendations")
        if tags is not None:
            st.info(f"Loaded {len(tags)} tags for better content matching")
        
        # Preprocess data
        with st.spinner("Processing movie data with enhanced features..."):
            cosine_sim, processed_movies = preprocess_data(movies, ratings, tags)
        
        if cosine_sim is not None:
            # Movie selection with integrated search
            st.markdown("### 🎬 Movie Selection")
            
            # Initialize session state for search
            if 'search_results' not in st.session_state:
                st.session_state.search_results = []
            
            # Search input
            search_term = st.text_input("🔍 Search and select a movie:", 
                                       placeholder="Start typing to search for movies...",
                                       help="Type to search, then select from dropdown")
            
            # Filter movies based on search term
            if search_term:
                filtered_movies = processed_movies[
                    processed_movies['title'].str.contains(search_term, case=False, na=False)
                ]
                if not filtered_movies.empty:
                    # Sort filtered movies by relevance (exact matches first)
                    exact_matches = filtered_movies[filtered_movies['title'].str.lower().str.contains(search_term.lower(), case=False)]
                    movie_options = exact_matches['title'].tolist()
                    st.session_state.search_results = movie_options
                    st.info(f"Found {len(movie_options)} movies matching '{search_term}'")
                else:
                    movie_options = []
                    st.session_state.search_results = []
                    st.warning(f"No movies found matching '{search_term}'")
            else:
                movie_options = processed_movies['title'].tolist()[:50]  # Show first 50 movies if no search
                st.session_state.search_results = movie_options
            
            # Movie selection dropdown
            if movie_options:
                selected_movie = st.selectbox(
                    "Select a movie:",
                    options=[""] + movie_options,
                    help="Choose a movie from the search results"
                )
            else:
                selected_movie = ""
            
            # Enhanced UI with better layout
            st.markdown("---")
            st.subheader("Customize Your Recommendations")
            
            # Create two columns for better organization
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**📊 Number of Recommendations**")
                
                num_recommendations = st.number_input(
                    "Number of recommendations:", 
                    min_value=1, 
                    max_value=100, 
                    value=10,
                    help="Enter number of recommendations to display"
                )
                
                # Sorting options
                st.markdown("**🔄 Sort By**")
                sort_options = ['similarity', 'rating', 'genre']
                sort_display = ['🎯 Similarity', '⭐ Rating', '🎭 Genre']
                sort_by = st.selectbox(
                    "Sort recommendations by:", 
                    sort_options, 
                    format_func=lambda x: sort_display[sort_options.index(x)],
                    help="Choose how to sort recommendations"
                )
            
            with col2:
                # Year filtering
                if 'year' in processed_movies.columns:
                    available_years = processed_movies['year'].dropna().astype(int)
                    min_year = int(available_years.min())
                    max_year = int(available_years.max())
                    
                    st.markdown("**📅 Year Range Filter**")
                    
                    # Manual year inputs
                    col_start, col_end = st.columns(2)
                    with col_start:
                        start_year = st.number_input(
                            "From Year:", 
                            min_value=min_year, 
                            max_value=max_year, 
                            value=min_year,
                            help="Starting year"
                        )
                    with col_end:
                        end_year = st.number_input(
                            "To Year:", 
                            min_value=min_year, 
                            max_value=max_year, 
                            value=max_year,
                            help="Ending year"
                        )
                        
                else:
                    start_year, end_year = None, None
                    st.markdown("**📅 Year Range Filter**")
                    st.info("Year data not available")
            
            # Add some spacing
            st.markdown("---")
            
            # Get recommendations button
            if selected_movie and st.button("Get Recommendations", type="primary"):
                with st.spinner("Finding personalized recommendations..."):
                    recommendations = get_recommendations(selected_movie, processed_movies, 
                                                        cosine_sim, num_recommendations,
                                                        start_year, end_year, sort_by)
                
                st.subheader(f"Top {num_recommendations} Recommended Movies (Sorted by {sort_by}):")
                
                # Display recommendations in normal card format
                for i, (display_name, genres, year, avg_rating, rating_count, similarity) in enumerate(recommendations, 1):
                    if similarity >= 0:  # Valid recommendation
                        # Create a card-like display for each movie
                        st.markdown(f"### {i}. {display_name}")
                        
                        # Create columns for organized display
                        info_col1, info_col2, info_col3 = st.columns([2, 1, 1])
                        
                        with info_col1:
                            st.write(f"**Genres:** {genres}")
                            if year:
                                st.write(f"**Year:** {year}")
                        
                        with info_col2:
                            if avg_rating > 0:
                                st.write(f"**Rating:** {avg_rating:.1f}/5.0")
                                st.write(f"**({int(rating_count)} ratings)**")
                        
                        with info_col3:
                            st.write(f"**Similarity:** {similarity:.1f}%")
                            # Add progress bar for similarity
                            st.progress(similarity / 100)
                        
                        # Add separator between movies
                        st.markdown("---")
                    else:
                        st.error(display_name)  # Show error message if movie not found
    
    # Enhanced sidebar information
    st.sidebar.header("About Enhanced Recommender")
    st.sidebar.info(
        "This advanced recommender system uses multiple features:\n\n"
        "- Content-based filtering with genres, titles, and tags\n"
        "- Rating-based sorting for quality recommendations\n"
        "- Year-based filtering for time-specific suggestions\n"
        "- TF-IDF vectorization with cosine similarity\n"
        "- Enhanced matching with exact and partial search\n"
        "- Smart movie naming for sequels and series"
    )
    
    st.sidebar.header("How to use")
    st.sidebar.write("1. Download the MovieLens dataset from: https://grouplens.org/datasets/movielens/")
    st.sidebar.write("2. Place movies.csv, ratings.csv, and tags.csv in the same directory")
    st.sidebar.write("3. Search and select a movie")
    st.sidebar.write("4. Adjust number of recommendations using manual input")
    st.sidebar.write("5. Set year range using manual inputs")
    st.sidebar.write("6. Choose sorting preference")
    st.sidebar.write("7. Click 'Get Recommendations' to see results")
    
    st.sidebar.header("Features")
    st.sidebar.write("- Integrated search with dropdown selection")
    st.sidebar.write("- Smart movie naming for sequels")
    st.sidebar.write("- Fully synchronized slider and manual inputs")
    st.sidebar.write("- Multiple sorting options")
    st.sidebar.write("- Year-based filtering")
    st.sidebar.write("- Enhanced with tags and ratings data")
    st.sidebar.write("- Similarity shown as percentage")
    st.sidebar.write("- Clean card-style movie display")

if __name__ == "__main__":
    main()