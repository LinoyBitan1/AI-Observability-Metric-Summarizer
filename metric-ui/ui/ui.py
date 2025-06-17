# main_page.py - AI Model Metric Summarizer
import streamlit as st
import requests
from datetime import datetime
import pandas as pd
import os

# --- Config ---
API_URL = os.getenv("MCP_API_URL", "http://localhost:8000")
PROM_URL = os.getenv("PROM_URL", "http://localhost:9090")

# --- Page Setup ---
st.set_page_config(page_title="AI Metric Tools", layout="wide")
st.markdown("""
<style>
    html, body, [class*="css"] { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto; }
    h1, h2, h3 { font-weight: 600; color: #1c1c1e; letter-spacing: -0.5px; }
    .stMetric { border-radius: 12px; background-color: #f9f9f9; padding: 1em; box-shadow: 0 2px 8px rgba(0,0,0,0.05); color: #1c1c1e !important; }
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: #1c1c1e !important; font-weight: 600; }
    .block-container { padding-top: 2rem; }
    .stButton>button { border-radius: 8px; padding: 0.5em 1.2em; font-size: 1em; }
    footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --- Page Selector ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["📊 Metric Summarizer", "🤖 Chat with Prometheus"])

# --- Shared Utilities ---
@st.cache_data(ttl=300)
def get_models():
    try:
        res = requests.get(f"{API_URL}/models")
        return res.json()
    except Exception as e:
        st.sidebar.error(f"Error fetching models: {e}")
        return []

@st.cache_data(ttl=300)
def get_namespaces():
    try:
        res = requests.get(f"{API_URL}/models")
        models = res.json()
        # Extract unique namespaces from model names (format: "namespace | model")
        namespaces = sorted(list(set(model.split(" | ")[0] for model in models if " | " in model)))
        return namespaces
    except Exception as e:
        st.sidebar.error(f"Error fetching namespaces: {e}")
        return []

model_list = get_models()
namespaces = get_namespaces()

# Add namespace selector in sidebar
selected_namespace = st.sidebar.selectbox("Select Namespace", namespaces)

# Filter models by selected namespace
filtered_models = [model for model in model_list if model.startswith(f"{selected_namespace} | ")]
model_name = st.sidebar.selectbox("Select Model", filtered_models)

st.sidebar.markdown("### Select Timestamp Range")
if "selected_date" not in st.session_state:
    st.session_state["selected_date"] = datetime.now().date()
if "selected_time" not in st.session_state:
    st.session_state["selected_time"] = datetime.now().time()
selected_date = st.sidebar.date_input("Date", value=st.session_state["selected_date"])
selected_time = st.sidebar.time_input("Time", value=st.session_state["selected_time"])
selected_datetime = datetime.combine(selected_date, selected_time)
now = datetime.now()
if selected_datetime > now:
    st.sidebar.warning("Please select a valid timestamp before current time.")
    st.stop()
selected_start = int(selected_datetime.timestamp())
selected_end = int(now.timestamp())

# --- 📊 Metric Summarizer Page ---
if page == "📊 Metric Summarizer":
    st.markdown("<h1>📊 AI Model Metric Summarizer</h1>", unsafe_allow_html=True)
    if st.button("🔍 Analyze Metrics"):
        with st.spinner("Analyzing metrics..."):
            try:
                response = requests.post(f"{API_URL}/analyze", json={
                    "model_name": model_name,
                    "start_ts": selected_start,
                    "end_ts": selected_end
                })
                result = response.json()
                st.session_state["prompt"] = result.get("health_prompt", "")
                st.session_state["summary"] = result.get("llm_summary", "")
                st.session_state["model_name"] = model_name
                st.session_state["metric_data"] = result.get("metrics", {})
                st.success("✅ Summary generated successfully!")
            except Exception as e:
                st.error(f"Error during analysis: {e}")

    if "summary" in st.session_state:
        col1, col2 = st.columns([1.3, 1.7])
        with col1:
            st.markdown("### 🧠 Model Insights Summary")
            st.markdown(st.session_state["summary"])
            st.markdown("### 💬 Ask Assistant")
            question = st.text_input("Ask a follow-up question")
            if st.button("Ask"):
                with st.spinner("Assistant is thinking..."):
                    try:
                        reply = requests.post(f"{API_URL}/chat", json={
                            "model_name": st.session_state["model_name"],
                            "prompt_summary": st.session_state["prompt"],
                            "question": question
                        })
                        st.markdown("**Assistant's Response:**")
                        st.markdown(reply.json()["response"])
                    except Exception as e:
                        st.error(f"Chat failed: {e}")
        with col2:
            st.markdown("### 📊 Metric Dashboard")
            metric_data = st.session_state.get("metric_data", {})
            metrics = [
                "Prompt Tokens Created", "P95 Latency (s)", "Requests Running",
                "GPU Usage (%)", "Output Tokens Created", "Inference Time (s)"
            ]
            cols = st.columns(3)
            for i, label in enumerate(metrics):
                df = metric_data.get(label)
                if df:
                    try:
                        values = [point["value"] for point in df]
                        avg_val = sum(values) / len(values)
                        max_val = max(values)
                        with cols[i % 3]:
                            st.metric(label=label, value=f"{avg_val:.2f}", delta=f"↑ Max: {max_val:.2f}")
                    except Exception as e:
                        with cols[i % 3]:
                            st.metric(label=label, value="Error", delta=f"{e}")
                else:
                    with cols[i % 3]:
                        st.metric(label=label, value="N/A", delta="No data")

            st.markdown("### 📈 Trend Over Time")
            dfs = []
            for label in ["GPU Usage (%)", "P95 Latency (s)"]:
                raw_data = metric_data.get(label, [])
                if raw_data:
                    try:
                        timestamps = [datetime.fromisoformat(p["timestamp"]) for p in raw_data]
                        values = [p["value"] for p in raw_data]
                        df = pd.DataFrame({label: values}, index=timestamps)
                        dfs.append(df)
                    except Exception as e:
                        st.warning(f"Chart error for {label}: {e}")
            if dfs:
                chart_df = pd.concat(dfs, axis=1).fillna(0)
                st.line_chart(chart_df)
            else:
                st.info("No data available to generate chart.")

# --- 🤖 Chat with Prometheus Page ---
elif page == "🤖 Chat with Prometheus":
    st.markdown("<h1>Chat with Prometheus</h1>", unsafe_allow_html=True)
    st.markdown(f"Currently selected namespace: **{selected_namespace}**")
    st.markdown("Ask questions like: `What's the P95 latency?`, `Is GPU usage stable?`, etc.")
    user_question = st.text_input("Your question")
    if st.button("Chat with Metrics"):
        if not user_question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Querying and summarizing..."):
                try:
                    response = requests.post(f"{API_URL}/chat-metrics", json={
                        "model_name": model_name,
                        "question": user_question,
                        "start_ts": selected_start,
                        "end_ts": selected_end,
                        "namespace": selected_namespace  # Add namespace to the request
                    })
                    data = response.json()
                    promql = data.get("promql", "")
                    summary = data.get("summary", "")
                    if not summary:
                        st.error("Error: Missing summary in response from AI.")
                    else:
                        st.markdown("**Generated PromQL:**")
                        if promql:
                            st.code(promql, language="yaml")
                        else:
                            st.info("No direct PromQL generated for this question.")
                        st.markdown("**AI Summary:**")
                        st.text(summary)
                except Exception as e:
                    st.error(f"Error: {e}")