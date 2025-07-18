import gradio as gr
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Constants
VOCAB_SIZE = 10000
MAXLEN = 200

# Load model
model = tf.keras.models.load_model("model.h5")

# Load IMDB word index
word_index = tf.keras.datasets.imdb.get_word_index()

# Text preprocessing
def review_to_sequence(review):
    words = review.lower().split()
    seq = [word_index.get(word, 2) for word in words]
    return pad_sequences([seq], maxlen=MAXLEN)

# Prediction logic
def predict_sentiment(review):
    sequence = review_to_sequence(review)
    pred = model.predict(sequence)[0][0]
    label = "🟢 Positive 😀" if pred > 0.5 else "🔴 Negative 😞"
    return f"{label} ({pred:.2f})"

# Gradio Interface
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=4, placeholder="Enter your movie review here..."),
    outputs=gr.Textbox(label="Sentiment"),
    title="🎬 Movie Review Sentiment Classifier",
    description="Paste your movie review below. This app predicts whether it's a positive or negative review.",
    theme="soft"
)

demo.launch()
