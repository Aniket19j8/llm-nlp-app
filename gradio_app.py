"""
Gradio UI — Modular frontend for all NLP capabilities
======================================================
Connects to the FastAPI backend and provides an interactive
interface for summarization, sentiment, rewriting, NER, and keywords.
"""

import os
import json
import gradio as gr
import httpx

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


def call_api(endpoint: str, payload: dict) -> dict:
    """Helper to call the FastAPI backend."""
    try:
        resp = httpx.post(
            f"{API_BASE}/api/v1/{endpoint}",
            json=payload,
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tab handlers
# ---------------------------------------------------------------------------

def summarize_text(text: str, max_len: int, min_len: int):
    if not text.strip():
        return "⚠️  Please enter some text.", ""
    result = call_api("summarize", {
        "text": text,
        "max_length": int(max_len),
        "min_length": int(min_len),
    })
    if "error" in result:
        return f"❌ Error: {result['error']}", ""
    meta = (
        f"📊 Original: {result['original_length']} words → "
        f"Summary: {result['summary_length']} words "
        f"(compression: {result['compression_ratio']:.1%}) | "
        f"Backend: {result['backend']}"
    )
    return result["summary"], meta


def analyse_sentiment(text: str):
    if not text.strip():
        return "⚠️  Please enter some text.", ""
    result = call_api("sentiment", {"text": text})
    if "error" in result:
        return f"❌ Error: {result['error']}", ""
    emoji = "😊" if result["label"] == "POSITIVE" else "😞"
    main = f"{emoji} **{result['label']}** (score: {result['score']:.4f}, confidence: {result['confidence']})"
    details = json.dumps(result["all_scores"], indent=2)
    return main, details


def rewrite_text(text: str, style: str):
    if not text.strip():
        return "⚠️  Please enter some text.", ""
    result = call_api("rewrite", {"text": text, "style": style.lower()})
    if "error" in result:
        return f"❌ Error: {result['error']}", ""
    return result["rewritten"], f"Style: {result['style']} | Backend: {result['backend']}"


def extract_entities(text: str):
    if not text.strip():
        return "⚠️  Please enter some text."
    result = call_api("ner", {"text": text})
    if "error" in result:
        return f"❌ Error: {result['error']}"
    if not result["entities"]:
        return "No entities found."
    lines = [f"Found **{result['count']}** entities:\n"]
    for e in result["entities"]:
        lines.append(f"- **{e['text']}** [{e['label']}] (confidence: {e['score']:.2%})")
    return "\n".join(lines)


def extract_keywords(text: str):
    if not text.strip():
        return "⚠️  Please enter some text."
    result = call_api("keywords", {"text": text})
    if "error" in result:
        return f"❌ Error: {result['error']}"
    lines = ["**Top Keywords:**\n"]
    for kw in result["keywords"]:
        bar = "█" * int(kw["score"] * 100)
        lines.append(f"- **{kw['keyword']}** (freq: {kw['frequency']}, score: {kw['score']:.4f}) {bar}")
    return "\n".join(lines)


def get_health():
    try:
        resp = httpx.get(f"{API_BASE}/health", timeout=5.0)
        data = resp.json()
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Build Gradio App
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="🧠 LLM-Powered NLP Platform",
    theme=gr.themes.Soft(primary_hue="blue"),
    css="""
    .gradio-container { max-width: 960px !important; }
    footer { display: none !important; }
    """,
) as demo:
    gr.Markdown(
        """
        # 🧠 LLM-Powered NLP Microservice
        **Production GenAI platform** with intelligent fallback routing
        (Local Models → OpenAI → OpenRouter) for **100% availability**.
        """
    )

    with gr.Tab("📝 Summarization"):
        with gr.Row():
            with gr.Column():
                sum_input = gr.Textbox(
                    label="Input Text",
                    placeholder="Paste a long article or document here…",
                    lines=10,
                )
                with gr.Row():
                    sum_max = gr.Slider(50, 500, value=150, step=10, label="Max Length")
                    sum_min = gr.Slider(10, 200, value=30, step=5, label="Min Length")
                sum_btn = gr.Button("Summarize", variant="primary")
            with gr.Column():
                sum_output = gr.Textbox(label="Summary", lines=8)
                sum_meta = gr.Textbox(label="Metadata")
        sum_btn.click(summarize_text, [sum_input, sum_max, sum_min], [sum_output, sum_meta])

    with gr.Tab("💬 Sentiment Analysis"):
        with gr.Row():
            with gr.Column():
                sent_input = gr.Textbox(
                    label="Input Text",
                    placeholder="Enter text to analyse sentiment…",
                    lines=5,
                )
                sent_btn = gr.Button("Analyse", variant="primary")
            with gr.Column():
                sent_result = gr.Markdown(label="Result")
                sent_details = gr.Code(label="Score Details", language="json")
        sent_btn.click(analyse_sentiment, [sent_input], [sent_result, sent_details])

    with gr.Tab("✍️ Text Rewriting"):
        with gr.Row():
            with gr.Column():
                rew_input = gr.Textbox(label="Input Text", lines=5)
                rew_style = gr.Dropdown(
                    ["Formal", "Casual", "Concise", "Academic", "Creative"],
                    value="Formal",
                    label="Target Style",
                )
                rew_btn = gr.Button("Rewrite", variant="primary")
            with gr.Column():
                rew_output = gr.Textbox(label="Rewritten Text", lines=8)
                rew_meta = gr.Textbox(label="Metadata")
        rew_btn.click(rewrite_text, [rew_input, rew_style], [rew_output, rew_meta])

    with gr.Tab("🏷️ Named Entities"):
        ner_input = gr.Textbox(label="Input Text", lines=5)
        ner_btn = gr.Button("Extract Entities", variant="primary")
        ner_output = gr.Markdown(label="Entities")
        ner_btn.click(extract_entities, [ner_input], [ner_output])

    with gr.Tab("🔑 Keywords"):
        kw_input = gr.Textbox(label="Input Text", lines=5)
        kw_btn = gr.Button("Extract Keywords", variant="primary")
        kw_output = gr.Markdown(label="Keywords")
        kw_btn.click(extract_keywords, [kw_input], [kw_output])

    with gr.Tab("🏥 Health"):
        health_btn = gr.Button("Check Health")
        health_output = gr.Code(language="json")
        health_btn.click(get_health, [], [health_output])

    gr.Markdown("---\n*Built with FastAPI + Gradio + Transformers • Fallback: Local → OpenAI → OpenRouter*")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
