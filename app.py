import streamlit as st
import pandas as pd
import joblib

# Set page configuration for a modern, wide look
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load the pre-trained KNN model using caching to optimize performance
@st.cache_resource
def load_genre_model():
    return joblib.load('genre_knn.pkl')

try:
    model = load_genre_model()
    classes = model.classes_
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Header section with clean design
st.markdown("<h1 style='text-align: center; color: #1DB954;'>🎵 Music Genre Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666666; font-size: 1.1rem;'>Adjust the audio feature sliders below to predict the track's genre and view the probability distribution.</p>", unsafe_allow_html=True)
st.markdown("---")

# Layout with two side-by-side columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 🎛️ Audio Features")
    
    # Use standard Streamlit form to implement the "On submit" behavior cleanly
    with st.form("audio_features_form"):
        tempo = st.slider(
            "Tempo (BPM)", 
            min_value=40.0, 
            max_value=220.0, 
            value=120.0, 
            step=1.0,
            help="The overall estimated tempo of a track in beats per minute (BPM)."
        )
        
        energy = st.slider(
            "Energy", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.01,
            help="Perceptual measure of intensity and activity (e.g., fast, loud, noisy)."
        )
        
        danceability = st.slider(
            "Danceability", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.01,
            help="How suitable a track is for dancing based on tempo, rhythm stability, and beat strength."
        )
        
        acousticness = st.slider(
            "Acousticness", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.3, 
            step=0.01,
            help="A confidence measure of whether the track is acoustic."
        )
        
        # Full-width submission button
        submit_btn = st.form_submit_button("🔮 Predict Genre", use_container_width=True)

with col2:
    st.markdown("### 📊 Prediction Results")
    
    if submit_btn:
        # Construct dataframe with exact feature names expected by the model pipeline
        input_df = pd.DataFrame([{
            'tempo': tempo,
            'energy': energy,
            'danceability': danceability,
            'acousticness': acousticness
        }])
        
        # Perform inference
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        # Display primary prediction badge
        st.markdown(
            f'''
            <div style="
                background-color: #1DB954; 
                padding: 15px; 
                border-radius: 10px; 
                text-align: center; 
                margin-bottom: 25px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            ">
                <span style="color: white; font-size: 0.9rem; uppercase; letter-spacing: 1px; display: block; font-weight: 600;">MOST LIKELY GENRE</span>
                <span style="color: white; font-size: 2rem; font-weight: 800; display: block; margin-top: 5px;">{prediction}</span>
            </div>
            ''', 
            unsafe_allow_html=True
        )
        
        # Display full class probability breakdown
        st.markdown("#### 📈 Probability Breakdown")
        
        for idx, (genre, prob) in enumerate(zip(classes, probabilities)):
            # Highlight the predicted class slightly
            is_predicted = (genre == prediction)
            text_weight = "bold" if is_predicted else "normal"
            text_color = "#1DB954" if is_predicted else "#333333"
            
            st.markdown(f"<div style='display: flex; justify-content: space-between; margin-bottom: 2px;'><span style='font-weight: {text_weight}; color: {text_color};'>{genre}</span><span style='font-weight: {text_weight}; color: {text_color};'>{prob*100:.1f}%</span></div>", unsafe_allow_html=True)
            st.progress(float(prob))
            
    else:
        # Default empty state with a nice info box
        st.info("💡 Adjust the sliders and click **Predict Genre** to view predictions and probabilities.")
