import os
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# --- DEEP LEARNING ---
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- NLP & GROQ ---
from transformers import pipeline
from groq import Groq

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==========================================
# IMAGE MODEL LOADING
# ==========================================
image_model = None
class_names = ['Butterfly', 'Cat', 'Elephant']

try:
    image_model = load_model("animal_classifier.h5", compile=False)
    print("✅ Image model loaded successfully")
except Exception as e:
    print(f"❌ Image model error: {e}")
    image_model = None

# ==========================================
# NLP & GROQ INITIALIZATION
# ==========================================
try:
    nlp_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    print("✅ NLP model loaded")
except Exception as e:
    print(f"❌ NLP error: {e}")
    nlp_model = None

try:
    groq_client = Groq(api_key="YOUR_API_KEY_HERE")
    print("✅ Groq ready")
except Exception as e:
    print(f"❌ Groq error: {e}")
    groq_client = None

# ==========================================
# ROUTES
# ==========================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return render_template('index.html', image_error="No file uploaded")

    file = request.files['file']
    if file and image_model:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        preds = image_model.predict(img_array)

        if isinstance(preds, dict):
            preds = preds[list(preds.keys())[0]]

        confidence = float(np.max(preds))
        predicted_index = int(np.argmax(preds))

        if confidence < 0.60:
            prediction = "Unknown/Uncertain"
        else:
            prediction = class_names[predicted_index]

        return render_template(
            'index.html',
            image_prediction=f"{prediction} ({confidence*100:.2f}%)",
            image_path=filepath
        )
    return render_template('index.html', image_error="Model not ready")

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    user_text = request.form.get('nlp_text')
    if user_text and nlp_model:
        result = nlp_model(user_text)[0]
        label = result['label'].lower()
        score = result['score']

        if score < 0.60:
            final = "Neutral"
        elif "positive" in label:
            final = "Positive"
        elif "negative" in label:
            final = "Negative"
        else:
            final = "Neutral"

        return render_template('index.html', nlp_prediction=final, user_text=user_text)
    return render_template('index.html', nlp_error="Enter text")

@app.route('/ask_agent', methods=['POST'])
def ask_agent():
    user_prompt = request.form.get('agent_prompt')
    if user_prompt and groq_client:
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            agent_response = response.choices[0].message.content
            return render_template('index.html',
                                 agent_response=agent_response,
                                 user_prompt=user_prompt)
        except Exception as e:
            return render_template('index.html', agent_error=f"Error: {e}")
    return render_template('index.html', agent_error="Enter prompt")

if __name__ == '__main__':
    app.run(debug=True)