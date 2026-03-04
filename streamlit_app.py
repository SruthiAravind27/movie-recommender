import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Sample dataset
# -----------------------------
data = pd.DataFrame({
    "title": [
        "Inception", "The Dark Knight", "Interstellar", "Stranger Things",
        "Breaking Bad", "The Matrix", "Friends", "The Office",
        "The Mandalorian", "The Witcher", "Titanic", "Avengers: Endgame"
    ],
    "description": [
        "A thief who steals corporate secrets through dream-sharing technology.",
        "Batman faces the Joker, who wreaks havoc on Gotham City.",
        "A team of explorers travel through a wormhole in space.",
        "A group of kids encounter supernatural forces in their town.",
        "A chemistry teacher turns to making meth to provide for his family.",
        "A hacker discovers the reality he lives in is a simulation.",
        "Comedy series about a group of friends living in New York.",
        "Mockumentary-style comedy following office employees.",
        "A lone bounty hunter navigates the outer reaches of the galaxy.",
        "A monster hunter struggles to find his place in a dangerous world.",
        "A tragic love story aboard the Titanic.",
        "Superheroes unite to defeat a powerful enemy."
    ]
})

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="🎬 Movie/TV Recommender",
    page_icon="🎥",
    layout="wide"
)

st.title("🎬 Movie/TV Show Recommender Chatbot")
st.write("Let's find your next favorite movie or TV show!")

# -----------------------------
# Session state
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Questions
questions = [
    "💭 What kind of movies or shows are you interested in right now?",
    "⭐ What are some of your favorite movies or TV shows?"
]

# -----------------------------
# Ask the next unanswered question
# -----------------------------
next_question_index = len(st.session_state.chat_history)
if next_question_index < len(questions):
    user_input = st.text_input(questions[next_question_index])
    if user_input:
        st.session_state.chat_history.append({"role": "user", "text": user_input})
        st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.stop()  # Fixed for newer versions
else:
    # -----------------------------
    # Show recommendations
    # -----------------------------
    all_user_text = " ".join([msg['text'] for msg in st.session_state.chat_history if msg['role']=="user"])
    
    def get_recommendations(user_text, top_k=5):
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(data['description'].tolist() + [user_text])
        cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        top_indices = cosine_sim[0].argsort()[-top_k:][::-1]
        recommended = data.iloc[top_indices].copy()
        recommended['similarity'] = cosine_sim[0][top_indices]
        return recommended

    recommendations = get_recommendations(all_user_text)

    st.subheader("🎯 Here are your recommendations:")
    for idx, row in recommendations.iterrows():
        st.markdown(f"**{row['title']}** - {row['description']}")
        st.slider(f"Rate {row['title']} (0-5 stars):", 0, 5, 0, key=f"rating_{idx}")

    st.info("💡 Click below to start over and get new recommendations!")
    if st.button("🔄 Start Over"):
        st.session_state.chat_history = []
        st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.stop()

# -----------------------------
# Sidebar fun illustrations
# -----------------------------
with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/movie-camera-emoji.png", width=80)
    st.write("🎬 Movie Night!")
    st.image("https://img.icons8.com/emoji/96/popcorn-emoji.png", width=80)
    st.write("🍿 Grab some snacks!")
    st.image("https://img.icons8.com/emoji/96/clapper-board-emoji.png", width=80)
    st.write("🎥 Action!")