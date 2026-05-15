"""Gradio app for HuggingFace Spaces — same model as api/main.py."""
import time
import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

tokenizer = AutoTokenizer.from_pretrained("checkpoints/lora")
base = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
)
model = PeftModel.from_pretrained(base, "checkpoints/lora").eval()


def predict(text: str):
    if not text or not text.strip():
        return {}, 0.0, 0.0
    t0 = time.perf_counter()
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    # Newer transformers/tokenizers also return `token_type_ids`, which DistilBERT.forward rejects.
    with torch.no_grad():
        probs = torch.softmax(
            model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]).logits,
            dim=-1,
        )[0]
    return ({ID2LABEL[i]: float(probs[i]) for i in range(3)},
            float(probs.max()), round((time.perf_counter() - t0) * 1000, 2))


gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Financial news text", lines=3, placeholder="Apple reported record earnings beating analyst expectations."),
    outputs=[gr.Label(label="Sentiment", num_top_classes=3),
             gr.Number(label="Confidence", precision=4),
             gr.Number(label="Latency (ms)", precision=2)],
    title="Financial Sentiment Analysis (DistilBERT + LoRA)",
    description="3-class sentiment classification fine-tuned on FinancialPhraseBank.",
    examples=[["Apple reported record earnings beating analyst expectations."],
              ["The company filed for bankruptcy amid mounting losses."],
              ["The board met today to review quarterly performance."]],
).launch(server_name="0.0.0.0", server_port=7860)
