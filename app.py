import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import re
import os
try:
    import openai_whisper as whisper
except ImportError:
    try:
        import whisper
    except ImportError:
        whisper = None
        print("⚠️ Whisper not available - voice input will be disabled")

from langdetect import detect
from deep_translator import GoogleTranslator

# ✅ Load DistilBERT classification model & tokenizer
model_path = "models/distilbert_base_optuna"  # Updated to use DistilBERT model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# ✅ Load Whisper model for voice input (if available)
whisper_model = None
if whisper and hasattr(whisper, 'load_model'):
    try:
        whisper_model = whisper.load_model("base")
        print("✅ Whisper model loaded successfully")
    except Exception as e:
        print(f"⚠️ Failed to load Whisper model: {e}")
        whisper_model = None
else:
    print("⚠️ Whisper not properly installed - voice input disabled")

# ✅ Religion-related keywords
religion_keywords = [
    "muslim", "islam", "islamic", "jew", "jewish", "judaism",
    "christian", "christianity", "bible", "jesus", "god", "catholic", "pope",
    "hindu", "hinduism", "buddha", "buddhist", "atheist", "religion", "religious"
]

# ✅ Language codes to full names
lang_labels = {
    "en": "English",
    "hr": "English",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "unknown": "Unknown"
}

def detect_lang(text):
    try:
        code = detect(text)
        return lang_labels.get(code, 'English')  # e.g. 'French' or fallback code
    except:
        return lang_labels["unknown"]

def translate_to_en(text, detected_lang_label):
    label_to_code = {v: k for k, v in lang_labels.items()}
    lang_code = label_to_code.get(detected_lang_label, "unknown")

    if lang_code not in ["fr", "de", "it"]:
        return "No translation applied"

    try:
        return GoogleTranslator(source=lang_code, target="en").translate(text)
    except Exception as e:
        print(f"⚠️ Translation error: {e}")
        return "No translation applied"

def extract_keywords(text):
    return [kw for kw in religion_keywords if re.search(rf"\b{kw}\b", text.lower())]

def highlight_words(text):
    words = text.split()
    base_inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)  # Updated max_length for DistilBERT
    with torch.no_grad():
        base_output = model(**base_inputs)
        base_score = F.softmax(base_output.logits, dim=1)[0][1].item()

    highlights = []
    for i, word in enumerate(words):
        modified_text = " ".join(words[:i] + words[i+1:])
        mod_inputs = tokenizer(modified_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            mod_output = model(**mod_inputs)
            mod_score = F.softmax(mod_output.logits, dim=1)[0][1].item()
        diff = base_score - mod_score
        label = f"{'⬆️' if diff > 0 else '⬇️'} {abs(diff):.2f}"
        highlights.append((word, label))
    return highlights

def predict_hate(text):
    if not text.strip():
        return "⚠️ Empty input", "—", "—", []

    detected = detect_lang(text)
    translated_text = translate_to_en(text, detected)
    
    # Check for religion keywords in both original and translated text
    text_to_check = translated_text if translated_text != "No translation applied" else text
    found_keywords = extract_keywords(text_to_check)
    
    # If no religion keywords found, don't make prediction
    if not found_keywords:
        return "⚠️ No prediction available - No religion-related keywords detected", detected, translated_text, []

    inputs = tokenizer(
        text_to_check,
        return_tensors="pt", truncation=True, padding=True, max_length=256  # Updated max_length for DistilBERT
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)[0]

    highlights = highlight_words(text_to_check)
    label_probs = {
        "Religious Hate": round(probs[1].item(), 3),
        "Not Hate": round(probs[0].item(), 3)
    }

    return label_probs, detected, translated_text, highlights

def transcribe_and_predict(audio_input):
    if whisper_model is None:
        return "⚠️ Voice input not available - Whisper model not loaded", "—", "—", []
    
    try:
        # Handle different audio input formats
        if audio_input is None:
            return "⚠️ No audio provided", "—", "—", []
        
        # If audio_input is a tuple (sample_rate, audio_data), use the file path
        if isinstance(audio_input, tuple):
            audio_path = audio_input[1] if len(audio_input) > 1 else audio_input[0]
        else:
            audio_path = audio_input
            
        # Ensure we have a valid file path
        if audio_path is None or not os.path.exists(str(audio_path)):
            return "⚠️ Invalid audio file", "—", "—", []
            
        transcription = whisper_model.transcribe(str(audio_path))["text"]
        return predict_hate(transcription)
    except Exception as e:
        return f"⚠️ Transcription error: {str(e)}", "—", "—", []

def file_input_handler(file):
    with open(file.name, "r") as f:
        text = f.read()
    return predict_hate(text)

# 🎨 Examples + UI
examples = [
    "Je suis fier d'être chrétien.",
    "La Bible est pleine de sagesse.",
    "Die Religion ist das Problem.",
    "Sono ebreo e stufo dei pregiudizi.",
    "Pope Francis inspires many.",
]

intro_title = "<h1 style='text-align: center; font-size: 36px;'>🇨🇭 Swiss Religious Hate Speech Detector (DistilBERT)"
description = """
🕊️ This model detects **religious hate speech** in Switzerland's four national languages: **French**, **German**, **Italian**, and **English**.

🤖 **Powered by DistilBERT:** This version uses an optimized DistilBERT model trained with hyperparameter tuning for improved accuracy and efficiency.

🌍 The app auto-detects the language, translates it to English, and analyzes it for religious hate.

⚠️ **Required Keywords:** The model ONLY makes predictions when religion-related keywords are detected. Text must contain at least one of these keywords:
- **Islam/Muslim:** muslim, islam, islamic
- **Judaism:** jew, jewish, judaism  
- **Christianity:** christian, christianity, bible, jesus, god, catholic, pope
- **Other religions:** hindu, hinduism, buddha, buddhist
- **General:** atheist, religion, religious

💡 Word importance scores explain which words influenced the prediction.
"""

with gr.Blocks(theme=gr.themes.Base()) as app:
    gr.Markdown(f"# {intro_title}")
    gr.Markdown(description)

    with gr.Tabs():
        with gr.TabItem("📝 Paste Text"):
            gr.Interface(
                fn=predict_hate,
                inputs=gr.Textbox(lines=4, placeholder="Enter a comment..."),
                outputs=[
                    gr.Label(label="Prediction"),
                    gr.Textbox(label="Detected Language"),
                    gr.Textbox(label="Translated Text"),
                    gr.HighlightedText(label="Word Importance")
                ],
                examples=examples,
                live=False
            )

        with gr.TabItem("🎤 Voice Input"):
            if whisper_model is not None:
                gr.Interface(
                    fn=transcribe_and_predict,
                    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath", label="Record or upload audio"),
                    outputs=[
                        gr.Label(label="Prediction"),
                        gr.Textbox(label="Detected Language"),
                        gr.Textbox(label="Translated Text"),
                        gr.HighlightedText(label="Word Importance")
                    ]
                )
            else:
                gr.Markdown("⚠️ **Voice input unavailable** - Whisper model not loaded. Please install: `pip install openai-whisper`")

        with gr.TabItem("📄 File Upload"):
            gr.Interface(
                fn=file_input_handler,
                inputs=gr.File(label="Upload a .txt file"),
                outputs=[
                    gr.Label(label="Prediction"),
                    gr.Textbox(label="Detected Language"),
                    gr.Textbox(label="Translated Text"),
                    gr.HighlightedText(label="Word Importance")
                ]
            )

if __name__ == "__main__":
    app.launch()