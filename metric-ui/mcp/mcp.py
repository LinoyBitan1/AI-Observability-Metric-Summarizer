from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import requests
from datetime import datetime
from scipy.stats import linregress
import os

app = FastAPI()

# --- CONFIG ---
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
LLM_URL = os.getenv("LLM_URL", "http://localhost:8080")
LLM_API_TOKEN = os.getenv("LLM_API_TOKEN", "")
LLM_MODEL_SUMMARIZATION = "meta-llama/Llama-3.2-3B-Instruct"

# Handle token input from volume or literal
token_input = os.getenv("THANOS_TOKEN", "/var/run/secrets/kubernetes.io/serviceaccount/token")
if os.path.exists(token_input):
    with open(token_input, "r") as f:
        THANOS_TOKEN = f.read().strip()
else:
    THANOS_TOKEN = token_input

# CA bundle location (mounted via ConfigMap)
CA_BUNDLE_PATH = "/etc/pki/ca-trust/extracted/pem/ca-bundle.crt"
verify = CA_BUNDLE_PATH if os.path.exists(CA_BUNDLE_PATH) else True

# --- Metric Queries ---
ALL_METRICS = {
    "Prompt Tokens Created": "vllm:request_prompt_tokens_created",
    "P95 Latency (s)": "vllm:e2e_request_latency_seconds_count",
    "Requests Running": "vllm:num_requests_running",
    "GPU Usage (%)": "vllm:gpu_cache_usage_perc",
    "Output Tokens Created": "vllm:request_generation_tokens_created",
    "Inference Time (s)": "vllm:request_inference_time_seconds_count"
}

# --- Request Models ---
class AnalyzeRequest(BaseModel):
    model_name: str
    start_ts: int
    end_ts: int

class ChatRequest(BaseModel):
    model_name: str
    prompt_summary: str
    question: str

# --- Helpers ---
def fetch_metrics(query, model_name, start, end):
    if "|" in model_name:
        namespace, model_name = map(str.strip, model_name.split("|", 1))
        promql_query = f'{query}{{model_name="{model_name}", namespace="{namespace}"}}'
    else:
        model_name = model_name.strip()
        promql_query = f'{query}{{model_name="{model_name}"}}'

    headers = {"Authorization": f"Bearer {THANOS_TOKEN}"}
    response = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query_range",
        headers=headers,
        params={"query": promql_query, "start": start, "end": end, "step": "30s"},
        verify=verify
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

def detect_anomalies(df, label):
    if df.empty:
        return "No data"
    mean = df["value"].mean()
    std = df["value"].std()
    p90 = df["value"].quantile(0.9)
    latest_val = df["value"].iloc[-1]
    if latest_val > p90:
        return f"⚠️ {label} spike (latest={latest_val:.2f}, >90th pct)"
    elif latest_val < (mean - std):
        return f"⚠️ {label} unusually low (latest={latest_val:.2f}, mean={mean:.2f})"
    return f"{label} stable (latest={latest_val:.2f}, mean={mean:.2f})"

def describe_trend(df):
    if df.empty or len(df) < 2:
        return "not enough data"
    df = df.sort_values("timestamp")
    x = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds()
    y = df["value"]
    if x.nunique() <= 1:
        return "flat"
    slope, *_ = linregress(x, y)
    if slope > 0.01:
        return "increasing"
    elif slope < -0.01:
        return "decreasing"
    return "stable"

def compute_health_score(metric_dfs):
    score, reasons = 0, []
    if "P95 Latency (s)" in metric_dfs and not metric_dfs["P95 Latency (s)"].empty:
        mean = metric_dfs["P95 Latency (s)"]["value"].mean()
        if mean > 2:
            score -= 2
            reasons.append(f"High Latency (avg={mean:.2f}s)")
    if "GPU Usage (%)" in metric_dfs and not metric_dfs["GPU Usage (%)"].empty:
        mean = metric_dfs["GPU Usage (%)"]["value"].mean()
        if mean < 10:
            score -= 1
            reasons.append(f"Low GPU Utilization (avg={mean:.2f}%)")
    if "Requests Running" in metric_dfs and not metric_dfs["Requests Running"].empty:
        mean = metric_dfs["Requests Running"]["value"].mean()
        if mean > 10:
            score -= 1
            reasons.append(f"Too many requests (avg={mean:.2f})")
    return score, reasons

def build_prompt(metric_dfs, model_name):
    score, _ = compute_health_score(metric_dfs)
    header = f"You are evaluating model **{model_name}**.\n\n🩺 Health Score: {score}\n\n📊 **Metrics**:\n"
    lines = []
    for label, df in metric_dfs.items():
        if df.empty:
            lines.append(f"- {label}: No data")
            continue
        trend = describe_trend(df)
        anomaly = detect_anomalies(df, label)
        avg = df["value"].mean()
        lines.append(f"- {label}: Avg={avg:.2f}, Trend={trend}, {anomaly}")
    return f"""{header}
{chr(10).join(lines)}

🔍 Please analyze:
1. What's going well?
2. What's problematic?
3. Recommendations?
""".strip()

def build_chat_prompt(user_question: str, metrics_summary: str) -> str:
    return f"""
You are a senior MLOps engineer responding to real-time model observability questions.

Your task is to respond using only the technical context from production metrics.

---
Metrics Summary:
{metrics_summary.strip()}
---

Guidelines:
- Do NOT repeat the user's question.
- Do NOT mention you're an assistant or explain your reasoning.
- NEVER write "please let me know" or "I'm here to help."
- Do NOT mention rewriting or summarizing anything.
- Do NOT exceed 100 words.
- Use the actual metric names as-is (e.g., "GPU Usage (%)", "P95 Latency (s)").
- Be confident and direct like a professional engineer writing in Slack or Jira.
- End your answer after your technical recommendation. No filler.

---
User Prompt:
{user_question.strip()}
---

Now respond with a concise, technical answer only.
""".strip()

def summarize_with_llm(prompt: str) -> str:
    headers = {"Content-Type": "application/json"}
    if LLM_API_TOKEN:
        headers["Authorization"] = f"Bearer {LLM_API_TOKEN}"
    payload = {
        "model": LLM_MODEL_SUMMARIZATION,
        "prompt": prompt,
        "temperature": 0.5,
        "max_tokens": 600
    }
    response = requests.post(f"{LLM_URL}/v1/completions", headers=headers, json=payload, verify=verify)
    response.raise_for_status()
    response_json = response.json()
    if "choices" not in response_json or not response_json["choices"]:
        raise ValueError("Invalid LLM response format")
    return response_json["choices"][0]["text"].strip()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/models")
def list_models():
    try:
        headers = {"Authorization": f"Bearer {THANOS_TOKEN}"}
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/series",
            headers=headers,
            params={
                "match[]": 'vllm:request_prompt_tokens_created',
                "start": int((datetime.now().timestamp()) - 3600),
                "end": int(datetime.now().timestamp())
            },
            verify=verify
        )
        response.raise_for_status()
        series = response.json()["data"]

        model_set = set()
        for entry in series:
            model = entry.get("model_name", "").strip()
            namespace = entry.get("namespace", "").strip()
            if model and namespace:
                model_set.add(f"{namespace} | {model}")
        return sorted(model_set)

    except Exception as e:
        print("Error in /models:", e)
        return []


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    metric_dfs = {
        label: fetch_metrics(query, req.model_name, req.start_ts, req.end_ts)
        for label, query in ALL_METRICS.items()
    }
    prompt = build_prompt(metric_dfs, req.model_name)
    summary = summarize_with_llm(prompt)

    serialized_metrics = {
        label: df[["timestamp", "value"]].to_dict(orient="records")
        for label, df in metric_dfs.items()
    }

    return {
        "model_name": req.model_name,
        "health_prompt": prompt,
        "llm_summary": summary,
        "metrics": serialized_metrics
    }

@app.post("/chat")
def chat(req: ChatRequest):
    prompt = build_chat_prompt(user_question=req.question, metrics_summary=req.prompt_summary)
    response = summarize_with_llm(prompt)
    return {"response": response}
