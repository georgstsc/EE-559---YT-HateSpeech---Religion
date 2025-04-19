import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import re
import whisper
from langdetect import detect
from deep_translator import GoogleTranslator

# ✅ Load classification model & tokenizer
model_path = "models/bert-tiny-hate"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# ✅ Load Whisper model for voice input
whisper_model = whisper.load_model("base")

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
    base_inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
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

    inputs = tokenizer(
        translated_text if translated_text != "No translation applied" else text,
        return_tensors="pt", truncation=True, padding=True, max_length=256
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)[0]

    highlights = highlight_words(translated_text if translated_text != "No translation applied" else text)
    label_probs = {
        "Religious Hate": round(probs[1].item(), 3),
        "Not Hate": round(probs[0].item(), 3)
    }

    return label_probs, detected, translated_text, highlights

def transcribe_and_predict(audio_path):
    transcription = whisper_model.transcribe(audio_path)["text"]
    return predict_hate(transcription)

def file_input_handler(file):
    with open(file.name, "r") as f:
        text = f.read()
    return predict_hate(text)

# 🎨 Examples + UI
examples = [
    "Musulmans, partez d'ici!",
    "Je suis fier d'être chrétien.",
    "La Bible est pleine de sagesse.",
    "Die Religion ist das Problem.",
    "Sono ebreo e stufo dei pregiudizi.",
    "Pope Francis inspires many.",
]

intro_title = "<h1 style='text-align: center; font-size: 36px;'>🇨🇭 Swiss Religious Hate Speech Detector"
description = """
🕊️ This model detects **religious hate speech** in Switzerland's four national languages: **French**, **German**, **Italian**, and **English**.

🌍 The app auto-detects the language, translates it to English, and analyzes it for religious hate.

⚠️ **Disclaimer:** The model was trained on comments containing religion-related keywords.
Best results are achieved when text includes terms like *"muslim"*, *"jewish"*, *"christian"*, *"bible"*, *"god"*, etc.

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
            gr.Interface(
                fn=transcribe_and_predict,
                inputs=gr.Audio(type="filepath", label="Speak a comment"),
                outputs=[
                    gr.Label(label="Prediction"),
                    gr.Textbox(label="Detected Language"),
                    gr.Textbox(label="Translated Text"),
                    gr.HighlightedText(label="Word Importance")
                ]
            )

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
