import streamlit as st
import numpy as np
import pickle
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import altair as alt

from gtts import gTTS
from io import BytesIO

# Load the pre-trained model and vectorizer
vectorizer = pickle.load(open('vectorize.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔",
                       "sadness": "😔", "shame": "😳", "surprise": "😮"}


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []

    negation = False
    for word in text:
        if word in ["not", "no"]:
            negation = True
        elif negation and word not in ["!", ".", ",", "?"]:
            y.append("not_" + word)
        else:
            y.append(word)
            negation = False

    return " ".join(y)


# Streamlit app
def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions in Text")
  
    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1,col2=st.columns(2)

        # Preprocessing
        input_text = transform_text(raw_text)
        # Vectorize
        input_vector = vectorizer.transform([input_text])
        # Prediction
        result = model.predict(input_vector)[0]
        probability=model.predict_proba(input_vector)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[result]
            st.write("{}:{}".format(result, emoji_icon))
            st.write("Confidence:{}".format(np.max(probability)))

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=model.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]
            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
            st.altair_chart(fig, use_container_width=True)

            st.success(f'Prediction: {result}')

        # Text-to-speech functionality
        tts = gTTS(raw_text)
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        st.audio(audio_bytes, format='audio/wav')

    else:
        st.warning("Please enter a message to predict its emotion.")

if __name__ == '__main__':
    main()
# pip install streamlit-webrtc
# pip install gtts
