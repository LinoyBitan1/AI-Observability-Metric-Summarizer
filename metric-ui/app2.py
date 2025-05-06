import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import json

# --- Config from ENV ---
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL")
LLM_URL = os.getenv("LLM_URL")
LLM_API_TOKEN = os.getenv("LLM_API_TOKEN", "")
LLM_MODEL_SUMMARIZATION = "meta-llama/Llama-3.2-3B-Instruct"
model_list_json = os.getenv("LLM_MODELS", '["Unknown"]')
model_names = json.loads(model_list_json)

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
for key in ["analyzed", "prompt", "chat_history", "last_user_input"]:
    if key not in st.session_state:
        st.session_state[key] = False if key == "analyzed" else [] if key == "chat_history" else ""

def fetch_metrics(query, model_name, start, end):
    promql_query = f'{query}{{model_name="{model_name}"}}'
    step = "30s"

    response = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query_range",
        params={
            "query": promql_query,
            "start": start,
            "end": end,
            "step": step
        },
        verify=False
    )
    response.raise_for_status()
    result = response.json()["data"]["result"]

    rows = []
    for series in result:
        for val in series["values"]:
            ts = datetime.fromtimestamp(float(val[0]))
            value = float(val[1])
            row = dict(series["metric"])
            row["timestamp"] = ts
            row["value"] = value
            rows.append(row)

    return pd.DataFrame(rows)


def build_prompt(metric_dfs, model_name):
    summary_block = f"You're an observability expert analyzing AI model performance.\nModel: {model_name}\n\nLatest metrics:\n"
    for label, df in metric_dfs.items():
        if df.empty:
            summary_block += f"- {label}: No data\n"
        else:
            summary_block += f"- {label}: Avg={df['value'].mean():.2f}, Max={df['value'].max():.2f}, Count={len(df)}\n"

    return f"""
You are an MLOps specialist evaluating the health and performance of the AI model {model_name} based on the following metrics:

{summary_block}

Please provide a clear and concise analysis covering the following:

1. **What's Going Well** – Highlight metrics or trends that indicate strong performance or improvement.  
2. **What's Going Wrong** – Identify any issues, anomalies, or metrics indicating degradation or concern.  
3. **Recommended Actions** – Suggest practical next steps to maintain strengths and address issues.  

Use bullet points and technical clarity in your response.
"""

def summarize_with_llm(prompt):
    headers = {"Content-Type": "application/json"}
    if LLM_API_TOKEN:
        headers["Authorization"] = f"Bearer {LLM_API_TOKEN}"

    payload = {
        "model": LLM_MODEL_SUMMARIZATION,
        "prompt": prompt,
        "temperature": 0.5,
        "max_tokens": 600
    }

    response = requests.post(f"{LLM_URL}/v1/completions", headers=headers, json=payload, verify=False)
    response.raise_for_status()
    return response.json()["choices"][0]["text"].strip()

# --- Layout Config ---
st.set_page_config(page_title="AI Model Metrics Dashboard", page_icon="📊", layout="wide")

# --- Sidebar: Model & Timestamp ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["📈 Analyze Metrics", "💬 Chat with Assistant"])
model = st.sidebar.selectbox("Select AI Model", model_names)
st.sidebar.markdown("**Select Timestamp**")

if "selected_date" not in st.session_state:
    st.session_state["selected_date"] = datetime.now().date()
if "selected_time" not in st.session_state:
    st.session_state["selected_time"] = datetime.now().time()

selected_date = st.sidebar.date_input("Date", value=st.session_state["selected_date"], key="date_input")
selected_time = st.sidebar.time_input("Time", value=st.session_state["selected_time"], key="time_input")
formatted_time = selected_time.strftime("%I:%M %p")  # Format to AM/PM
st.sidebar.markdown(f"**Selected Time:** {formatted_time}")


now = datetime.now()

# Prevent user from selecting a future date
if selected_date > now.date():
    st.warning("You've selected a future date. Please select today or a past date.")
    st.stop()

# Prevent user from selecting a future time on the same day
if selected_date == now.date() and selected_time > now.time():
    st.warning("You've selected a future time. Please select a time before the current time.")
    st.stop()

# Update the session state only when changed
st.session_state["selected_date"] = selected_date
st.session_state["selected_time"] = selected_time

selected_datetime = datetime.combine(selected_date, selected_time)
selected_start = int(selected_datetime.timestamp())
selected_end = int(now.timestamp())  # Always use current time as end



# --- Main Title ---
st.markdown("""
<div style="display: flex; align-items: center; gap: 10px;">
    <span style="font-size: 2.2em;">📈</span>
    <h1 style="margin-bottom: 0;">AI Model Metrics Summarizer</h1>
</div>
""", unsafe_allow_html=True)

# --- Analyze Page ---
if page == "📈 Analyze Metrics":
    if st.button("Analyze Metrics"):
        with st.spinner("Fetching and analyzing..."):
            try:
                metric_dfs = {
                    label: fetch_metrics(query, model, selected_start, selected_end)
                    for label, query in ALL_METRICS.items()
                }

                prompt = build_prompt(metric_dfs, model)
                st.session_state.prompt = prompt
                summary = summarize_with_llm(prompt)
                st.session_state.analyzed = True

                col1, col2 = st.columns([1.5, 2])
                with col1:
                    st.subheader("Summary")
                    st.markdown(f"**Model:** {model}")
                    st.markdown(summary)

                with col2:
                    st.subheader("GPU & Latency Trends")

                    # Metric cards for all metrics
                    # Metric cards for all metrics
                    for label in ALL_METRICS:
                        df = metric_dfs[label]
                        if df.empty:
                            avg_val = 0
                            max_val = 0
                        else:
                            avg_val = df["value"].mean()
                            max_val = df["value"].max()
                        st.metric(label, f"{avg_val:.2f}", delta=f"Max: {max_val:.2f}")


                    # Create consistent time index
                    full_time_index = pd.date_range(
                        start=datetime.fromtimestamp(selected_start),
                        end=datetime.fromtimestamp(selected_end),
                        freq="30S"
                    )
                    chart_df = pd.DataFrame(index=full_time_index)

                    # Populate chart data for selected metrics
                    # Populate chart data for selected metrics, filling empty ones with 0s
                    for label in DASHBOARD_METRICS:
                        df = metric_dfs[label]
                        if df.empty:
                            chart_df[label] = 0  # Flat 0 line
                        else:
                            df_plot = df[["timestamp", "value"]].set_index("timestamp").rename(columns={"value": label})
                            chart_df = chart_df.join(df_plot, how="left")

                    chart_df.fillna(0, inplace=True)


                    # Plot line chart below metrics
                    if not chart_df.empty:
                        st.line_chart(chart_df)






            except Exception as e:
                st.error(f"❌ Error: {e}")

# --- Chat Page ---
elif page == "💬 Chat with Assistant":
    st.subheader("💬 Ask Questions About Your Metrics")
    user_input = st.chat_input("Ask something like 'what's going wrong?'")

    if user_input and user_input != st.session_state.last_user_input:
        st.session_state.last_user_input = user_input

        if st.session_state.analyzed:
            prompt = f"""
You are a senior MLOps analyst evaluating model {model}.

Metrics summary:
{st.session_state.prompt.strip()}

User's question: "{user_input}"

Respond clearly and professionally.
"""
        else:
            prompt = f"""
The user asked a question but metrics haven't been analyzed yet. Kindly ask them to use the 'Analyze Metrics' section before chatting.

User's question: "{user_input}"
"""

        with st.spinner("Thinking..."):
            reply = summarize_with_llm(prompt)

        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("ai", reply))

    for role, msg in st.session_state.chat_history:
        st.chat_message(role).write(msg)

# --- Footer ---
st.markdown("---")
st.caption("Built with ❤️ by your-team · Powered by Prometheus & Llama Models")