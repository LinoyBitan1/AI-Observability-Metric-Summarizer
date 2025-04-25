import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import matplotlib.pyplot as plt

# --- Config ---
PROMETHEUS_URL = "http://prometheus-operated.llama-1.svc.cluster.local:9090"
DEEPINFRA_API_KEY = "api-key"
DEEPINFRA_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
DEEPINFRA_URL = f"https://api.deepinfra.com/v1/inference/{DEEPINFRA_MODEL}"

ALL_METRICS = {
    "Prompt Tokens Created": "vllm:prompt_tokens_created",
    "P95 Latency (s)": "vllm:e2e_request_latency_seconds_count",
    "Requests Running": "vllm:num_requests_running",
    "GPU Usage (%)": "vllm:gpu_cache_usage_perc",
    "Output Tokens Created": "vllm:request_generation_tokens_created",
    "Inference Time (s)": "vllm:request_inference_time_seconds_count"
}
DASHBOARD_METRICS = ["GPU Usage (%)", "P95 Latency (s)"]

# --- Session Init ---
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "prompt" not in st.session_state:
    st.session_state.prompt = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_user_input" not in st.session_state:
    st.session_state.last_user_input = ""

# --- Utilities ---
def fetch_metrics(query):
    response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}, verify=False)
    response.raise_for_status()
    result = response.json()["data"]["result"]
    rows = []
    for r in result:
        metric = r["metric"]
        value = float(r["value"][1])
        timestamp = datetime.fromtimestamp(float(r["value"][0]))
        metric["value"] = value
        metric["timestamp"] = timestamp
        rows.append(metric)
    return pd.DataFrame(rows)

def build_prompt(metric_dfs, model_name):
    prompt = f"""You're an observability expert analyzing AI model performance.\nModel: `{model_name}`\n\nLatest metrics:\n"""
    for label, df in metric_dfs.items():
        if df.empty:
            prompt += f"- {label}: No data\n"
        else:
            prompt += f"- {label}: Avg={df['value'].mean():.2f}, Max={df['value'].max():.2f}, Count={len(df)}\n"
    prompt += "\nPlease summarize:\n1. Key findings\n2. Possible issues\n3. Recommended action items"
    return prompt

def summarize_with_deepinfra(prompt):
    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"input": prompt, "temperature": 0.5, "max_tokens": 300}
    response = requests.post(DEEPINFRA_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["results"][0]["generated_text"].strip()

def get_model_names():
    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/series",
            params={"match[]": "vllm:e2e_request_latency_seconds_count"},
            verify=False
        )
        response.raise_for_status()
        series = response.json()["data"]
        return sorted(set(i.get("model_name") for i in series if "model_name" in i)) or ["Unknown"]
    except:
        return ["Unknown"]

# --- Layout Config ---
st.set_page_config(page_title="AI Model Metrics Dashboard", page_icon="📊", layout="wide")

# --- Logo + Title Styling ---
st.markdown("""
<div style="display: flex; align-items: center; gap: 10px;">
    <span style="font-size: 2.2em;">📈</span>
    <h1 style="margin-bottom: 0;">AI Model Metrics Summarizer</h1>
</div>
""", unsafe_allow_html=True)

# --- Page Toggle ---
page = st.sidebar.radio("Navigation", ["📊 Analyze Metrics", "💬 Chat with Assistant"])
model_names = get_model_names()
model = st.sidebar.selectbox("Select AI Model", model_names)
st.sidebar.selectbox("LLM for Summarization", [DEEPINFRA_MODEL], disabled=True)

# --- Page: Analyze Metrics ---
if page == "📊 Analyze Metrics":
    if st.button("Analyze Metrics"):
        with st.spinner("Fetching and analyzing..."):
            try:
                metric_dfs = {label: fetch_metrics(query) for label, query in ALL_METRICS.items()}
                summary_prompt = build_prompt(metric_dfs, model)
                summary = summarize_with_deepinfra(summary_prompt)

                st.session_state.analyzed = True
                st.session_state.prompt = summary_prompt

                col1, col2 = st.columns([1.5, 2])
                with col1:
                    st.subheader("📋 Summary")
                    st.markdown(f"**Model:** `{model}`")
                    st.markdown(f"**LLM:** `{DEEPINFRA_MODEL}`")
                    st.markdown(summary)

                with col2:
                    st.subheader("📈 GPU & Latency Dashboard")
                    for label, df in metric_dfs.items():
                        if not df.empty:
                            st.metric(label, f"{df['value'].mean():.2f}", delta=f"Max: {df['value'].max():.2f}")
                    avg_data = {
                        label: metric_dfs[label]["value"].mean()
                        for label in DASHBOARD_METRICS if not metric_dfs[label].empty
                    }
                    if avg_data:
                        pie_df = pd.Series(avg_data)
                        fig, ax = plt.subplots()
                        pie_df.plot.pie(autopct="%.1f%%", ax=ax, title="GPU vs Latency (Avg)", ylabel="")
                        st.pyplot(fig)
            except Exception as e:
                st.error(f"❌ Error: {e}")

# --- Page: Chat Interface ---
elif page == "💬 Chat with Assistant":
    st.subheader("💬 Ask Questions About Your Metrics")
    user_input = st.chat_input("Ask something like 'what's going right?'")

    if user_input and user_input != st.session_state.last_user_input:
        st.session_state.last_user_input = user_input
        prompt = (
            f"You're a helpful MLOps assistant. Metrics for model `{model}`:\n\n{st.session_state.prompt}\n\n"
            f"User question: {user_input}"
            if st.session_state.analyzed else
            f"You're a helpful MLOps assistant. Metrics haven't been analyzed yet.\nUser question: {user_input}"
        )

        with st.spinner("Thinking..."):
            reply = summarize_with_deepinfra(prompt)

        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("ai", reply))

    for role, msg in st.session_state.chat_history:
        st.chat_message(role).write(msg)

# --- Footer Branding ---
st.markdown("---")
st.caption("Built with ❤️ by your-team · Powered by Prometheus & DeepInfra")
