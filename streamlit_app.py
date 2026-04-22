import streamlit as st
import pandas as pd
from recommender2 import predict_tropes, load_and_train_model

@st.cache_data
def load_book_data():
    try:
        df = pd.read_csv("/workspaces/romance_book_tropes/romance_books_32K.zip", compression='zip')
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'romance_books_32K.zip' is in the workspace.")
        return pd.DataFrame()

st.sidebar.title("💕Romance Tropes!💕")

# Preload data to avoid waiting
with st.spinner("Loading data and model..."):
    df = load_book_data()
    model, clf, trope_columns = load_and_train_model()

option = st.sidebar.radio("Navigate", ["🚪Home","📚Recommend a Trope", "🔍Search for a book"])

if option == "🚪Home":
    st.markdown(
        """
        <div class="fade-in" style="text-align: ;">
            <h2 style="text-align:center;">Welcome to</h2>
            <h1 style="text-align:center;">💕Romance Tropes!💕</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )    
    info = st.selectbox(
    "Select to Learn More about the App:", 
    ["Description", "About", "Mission"],
    width=210)

    if info == "Description":
        st.write("Romance Tropes is an app designed to help readers find tropes within romance novels. It uses machine learning to analyze book descriptions and predict which tropes are present, making it easier for readers to discover books that match their interests.")
    elif info == "About":
        st.write("While the main feature of the app is the trope prediction, it also includes additional pages finding a book using filters and a user library page where you can add or remove books from.")
    elif info == "Mission":
        st.write("Our mission is to enhance the reading experience for romance novel enthusiasts by providing insights into the tropes present in their favorite books, helping them discover new reads that align with their preferences. This app is made with Streamlit and is a project for learning purposes, so expect some quirks along the way!")

    st.sidebar.write("📌 Use the sidebar to navigate different sections.")

# Add a Book
if option == "📚Recommend a Trope":
    st.header("📚Recommend a Trope")
    st.write("Enter the title, author, and description of a romance novel to predict which tropes are present in the book. This can help you discover new books with similar tropes that you might enjoy!")
    title = st.text_input("Title:", key="input_text", placeholder="Type the title of the book...")
    author = st.text_input("Author:", key="input_text2", placeholder="Type the author of the book...")
    description = st.text_area("Description:", key="input_text3", placeholder="Type a brief description of the book...")

    predict_button = st.button("Predict Tropes", disabled=not (title and author and description))
    if predict_button:
        if title and author and description:
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Loading model...")
            progress_bar.progress(5)
            
            # Model is already loaded
            progress_bar.progress(10)

            status_text.text("Encoding input into X...")
            progress_bar.progress(40)

            status_text.text("Running trope prediction...")
            result = predict_tropes(title, author, description, model, clf, trope_columns)
            progress_bar.progress(80)

            status_text.text("Finalizing results...")
            progress_bar.progress(100)

            status_text.empty()
            st.success("Prediction complete.")
            st.write("Predicted Tropes:")
            for trope, score in result:
                st.write(f"{trope}: {round(score, 3)}")

            progress_bar.empty()
        else:
            st.error("Please fill in all fields.")
    

# Search a Book
elif option == "🔍Search for a book":
    if df.empty:
        st.error("Data file not found. Please ensure 'romance_books_32K.csv' is in the workspace.")
        st.stop()
    
    st.sidebar.header("Filter Books")

    #date filter
    date_publish = st.sidebar.checkbox("Date Published:", value=False)
    begin_date = st.sidebar.date_input("From:", key="begin_date", disabled=not date_publish)
    end_date = st.sidebar.date_input("To:", key="end_date", disabled=not date_publish)

    #genre filter
    genre = st.sidebar.checkbox("Genre:", value=False)
    genre_list = st.sidebar.multiselect(
        "Choose Genres:",
        options=df['genre'].unique().tolist() if 'genre' in df.columns else ["Contemporary", "Historical", "Paranormal", "Romantic Comedy", "Fantasy"],
        default=None, disabled=not genre
    )

    st.title("🔍Search for a book")
    st.write("Search for books by title or author, and optionally filter by publication date and/or genre.")

    search_query = st.text_input("Search for a book:", key="search_query", placeholder="Type the title or author of the book...")

    # Apply filters
    filtered_df = df.copy()
    
    if search_query:
        filtered_df = filtered_df[
            filtered_df['title'].str.contains(search_query, case=False, na=False) |
            filtered_df['author'].str.contains(search_query, case=False, na=False)
        ]
    
    if date_publish and 'publication_year' in df.columns:
        begin_year = begin_date.year if begin_date else None
        end_year = end_date.year if end_date else None
        if begin_year:
            filtered_df = filtered_df[filtered_df['publication_year'] >= begin_year]
        if end_year:
            filtered_df = filtered_df[filtered_df['publication_year'] <= end_year]
    
    if genre and genre_list and 'genre' in df.columns:
        filtered_df = filtered_df[filtered_df['genre'].isin(genre_list)]
    
    # Display results
    if not filtered_df.empty:
        st.write(f"Found {len(filtered_df)} book(s):")
        st.dataframe(filtered_df[['title', 'author', 'description']].head(50))  # Show first 50 results
    else:
        st.write("No books found matching your criteria.")

st.markdown("---")
st.markdown("""
    <p style='text-align: center; color: gray;'>© 2026 Library Manager | Developed with ❤️ by Ayesha</p>
""", unsafe_allow_html=True)
