# ui/streamlit_app.py
import os
import time
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="LLM NLP Demo", page_icon="🧠", layout="centered")

# --- Header / health ---
st.title("🧠 LLM NLP Demo")
with st.sidebar:
    st.markdown("**Backend:** " + BACKEND_URL)
    try:
        r = requests.get(f"{BACKEND_URL}/healthz", timeout=5)
        if r.ok:
            st.success("API: healthy")
        else:
            st.warning(f"API: {r.status_code}")
    except Exception as e:
        st.error(f"API not reachable: {e}")

tabs = st.tabs(["🔍 Sentiment", "✍️ Rewrite (Tone)"])

# --- Sentiment tab ---
with tabs[0]:
    st.subheader("Sentiment Analysis (local PyTorch + Transformers)")
    text = st.text_area("Enter text", height=140, key="sent_text", placeholder="Type something like: I absolutely love this!")
    col_a, col_b = st.columns([1,1])
    with col_a:
        if st.button("Analyze sentiment", type="primary"):
            if not text.strip():
                st.warning("Please enter some text.")
            else:
                with st.spinner("Running sentiment..."):
                    t0 = time.time()
                    try:
                        resp = requests.post(
                            f"{BACKEND_URL}/v1/sentiment",
                            json={"text": text},
                            timeout=60,
                        )
                        if resp.ok:
                            data = resp.json()
                            latency = (time.time() - t0) * 1000
                            st.success(f"Label: **{data['label']}**  |  Confidence: **{data['score']:.3f}**")
                            st.caption(f"Latency: {latency:.0f} ms")
                        else:
                            st.error(f"Error {resp.status_code}: {resp.text}")
                    except Exception as e:
                        st.error(f"Request failed: {e}")
    with col_b:
        st.markdown("**Examples**")
        if st.button("Example: Positive"):
            st.session_state["sent_text"] = "I absolutely love this new feature—great job!"
        if st.button("Example: Negative"):
            st.session_state["sent_text"] = "This was terrible; it wasted my time."

# --- Rewrite tab ---
with tabs[1]:
    st.subheader("Rewrite in a chosen tone (OpenAI ➜ OpenRouter fallback)")
    text2 = st.text_area("Text to rewrite", height=180, key="rw_text",
                         placeholder="e.g., the delivery was late and the product was damaged. i want a refund.")
    tone = st.selectbox(
        "Tone",
        ["professional", "formal", "friendly", "concise", "apologetic", "assertive", "enthusiastic"],
        index=0,
    )
    if st.button("Rewrite"):
        if not text2.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Rewriting..."):
                t0 = time.time()
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/v1/rewrite",
                        json={"text": text2, "tone": tone},
                        timeout=90,
                    )
                    latency = (time.time() - t0) * 1000
                    if resp.ok:
                        data = resp.json()
                        st.text_area("Result", value=data["rewrite"], height=180)
                        st.caption(f"Latency: {data.get('latency_ms', latency):.0f} ms")
                    else:
                        # surface server-provided JSON error if present
                        try:
                            msg = resp.json()
                        except Exception:
                            msg = resp.text
                        st.error(f"Error {resp.status_code}: {msg}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

st.markdown("---")
st.caption("Sentiment uses local DistilBERT (PyTorch+Transformers). Rewrite uses OpenAI, "
           "with automatic fallback to OpenRouter (free Llama) if rate-limited.")
