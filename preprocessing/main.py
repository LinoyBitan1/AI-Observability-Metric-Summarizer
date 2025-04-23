from fastapi import FastAPI
from fastapi.responses import JSONResponse
import requests
from prometheus_client.parser import text_string_to_metric_families
import os

app = FastAPI()
PROM_METRICS_URL = os.getenv("PROM_METRICS_URL", "http://vllm.default.svc:8080/metrics")

@app.get("/metrics-summary")
def summarize_metrics():
    try:
        response = requests.get(PROM_METRICS_URL)
        if response.status_code != 200:
            return JSONResponse(status_code=500, content={"error": "Failed to fetch metrics"})

        parsed = parse_metrics(response.text)
        return JSONResponse(content=parsed)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

def parse_metrics(metrics_text: str):
    data = {}
    for family in text_string_to_metric_families(metrics_text):
        for sample in family.samples:
            name, value = sample.name, sample.value
            if name in ["process_cpu_seconds_total", "vllm_engine_num_prompt_tokens", "vllm_num_unfinished_requests"]:
                data[name] = value

    return {
        "cpu_usage": f"{data.get('process_cpu_seconds_total', 0.0):.2f} seconds",
        "num_prompt_tokens": int(data.get("vllm_engine_num_prompt_tokens", 0)),
        "unfinished_requests": int(data.get("vllm_num_unfinished_requests", 0))
    }
