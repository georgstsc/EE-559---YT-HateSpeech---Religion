import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import re
import whisper

# 📁 Load model + tokenizer
model_path = "models/bert-tiny-hate"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# 🔊 Whisper model
whisper_model = whisper.load_model("base")

# 🕊️ Religion-related keywords
religion_keywords = [
    "muslim", "islam", "islamic", "jew", "jewish", "judaism",
    "christian", "christianity", "bible", "jesus", "god", "catholic", "pope",
    "hindu", "hinduism", "buddha", "buddhist", "atheist", "religion", "religious"
]

def extract_keywords(text):
    return [kw for kw in religion_keywords if re.search(rf"\b{kw}\b", text.lower())]

# 🧠 Word-level "explanation"
def highlight_words(text):
    words = text.split()
    base_inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        base_output = model(**base_inputs)
        base_score = F.softmax(base_output.logits, dim=1)[0][1].item()  # hate class

    highlights = []
    for i, word in enumerate(words):
        modified = words[:i] + words[i+1:]
        modified_text = " ".join(modified)
        mod_inputs = tokenizer(modified_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            mod_output = model(**mod_inputs)
            mod_score = F.softmax(mod_output.logits, dim=1)[0][1].item()

        diff = base_score - mod_score
        label = f"{'⬆️' if diff > 0 else '⬇️'} {abs(diff):.2f}"
        highlights.append((word, label))

    return highlights


# 🔍 Main prediction
def predict_hate(text):
    if not text.strip():
        return "⚠️ Empty input", "—", []

    keywords = extract_keywords(text)
    if not keywords:
        return "⚠️ Not Enough Context", "No keywords found.", []

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)[0]

    keyword_display = ", ".join(keywords)
    highlights = highlight_words(text)

    # ✨ Return both class probabilities
    return {
        "Religious Hate": round(probs[1].item(), 3),
        "Not Hate": round(probs[0].item(), 3)
    }, keyword_display, highlights

# 🎙️ Voice input handler
def transcribe_and_predict(audio_path):
    transcription = whisper_model.transcribe(audio_path)["text"]
    pred, keywords, highlights = predict_hate(transcription)
    return pred, transcription, keywords, highlights

# 📁 File upload handler
def file_input_handler(file):
    with open(file.name, "r") as f:
        text = f.read()
    return predict_hate(text)

# 🧾 UI Meta
title = "🛡️ Religious Hate Speech Detector"
description = """
This model detects **religious hate speech** in user comments.

> ⚠️ The model was trained using comments that contain religion-related keywords. 
For best results, try sentences with terms like *"muslim"*, *"jewish"*, *"christian"*, etc.
"""
examples = [
    "Muslims are not welcome here.",
    "I'm so proud to be Christian 🙏",
    "The Bible is full of wisdom.",
    "Religion is the root of all problems.",
    "I’m Jewish and tired of being discriminated against.",
    "What’s wrong with being an atheist?",
    "Pope Francis is a global inspiration."
]

# 🎛️ Text mode
text_tab = gr.Interface(
    fn=predict_hate,
    inputs=gr.Textbox(lines=4, placeholder="Paste a comment..."),
    outputs=[
        gr.Label(num_top_classes=2, label="Prediction"),
        gr.Textbox(label="Matched Keywords"),
        gr.HighlightedText(label="Word-Level Importance")
    ],
    title=title,
    description=description,
    examples=examples,
    allow_flagging="never"
)

# 🎤 Voice mode
voice_tab = gr.Interface(
    fn=transcribe_and_predict,
    inputs=gr.Audio(type="filepath", label="Speak a sentence"),
    outputs=[
        gr.Label(num_top_classes=2, label="Prediction"),
        gr.Textbox(label="Transcribed Text"),
        gr.Textbox(label="Matched Keywords"),
        gr.HighlightedText(label="Word-Level Importance")
    ],
    title=title,
    description=description,
    allow_flagging="never"
)

# 📄 File upload mode
file_tab = gr.Interface(
    fn=file_input_handler,
    inputs=gr.File(label="Upload .txt file"),
    outputs=[
        gr.Label(num_top_classes=2, label="Prediction"),
        gr.Textbox(label="Matched Keywords"),
        gr.HighlightedText(label="Word-Level Importance")
    ],
    title=title,
    description=description,
    allow_flagging="never"
)

# 🚀 Launch
demo = gr.TabbedInterface(
    interface_list=[text_tab, voice_tab, file_tab],
    tab_names=["📝 Paste Text", "🎤 Voice Input", "📄 File Upload"]
)

if __name__ == "__main__":
    demo.launch()
