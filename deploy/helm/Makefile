# Makefile for RAG Deployment
# Replaces the original deploy.sh script with additional uninstall functionality
ifeq ($(NAMESPACE),)
ifeq (,$(filter list-models% help,$(MAKECMDGOALS)))
$(error NAMESPACE is not set)
endif
endif

MAKEFLAGS += --no-print-directory

# Default values
HF_TOKEN ?= $(shell bash -c 'read -r -p "Enter Hugging Face Token: " HF_TOKEN; echo $$HF_TOKEN')
LLM_SERVICE_CHART := llm-service
COMPONENTS := $(LLM_SERVICE_CHART) $(METRIC_UI_CHART_PATH) $(PROMETHEUS_CHART_PATH)
TOLERATIONS_TEMPLATE=[{"key":"$(1)","effect":"NoSchedule","operator":"Exists"}]
METRIC_MCP_CHART_PATH ?= metric-mcp
METRIC_MCP_RELEASE_NAME ?= metric-mcp
METRIC_UI_CHART_PATH ?= ui
METRIC_UI_RELEASE_NAME ?= ui
PROMETHEUS_CHART_PATH ?= prometheus
PROMETHEUS_RELEASE_NAME ?= prometheus

REGION ?= us-east-1

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  install       - Install the RAG deployment (creates namespace, secrets, and deploys Helm chart)"
	@echo "  install-cpu   - Install the RAG deployment without GPU (creates namespace, secrets, and deploys Helm chart)"
	@echo "  uninstall     - Uninstall the RAG deployment and clean up resources"
	@echo "  status        - Check status of the deployment"
	@echo "  wait          - Wait for all pods to be ready and verify deployment health"
	@echo "  list-models       - List available models for GPU"
	@echo "  list-models-cpu   - List available models for CPU"
	@echo ""
	@echo "Configuration options (set via environment variables or make arguments):"
	@echo "  NAMESPACE                - Target namespace (default: llama-stack-rag)"
	@echo "  HF_TOKEN                 - Hugging Face Token (will prompt if not provided)"
	@echo "  {SAFETY,LLM}             - Model id as defined in values (eg. llama-3-2-1b-instruct)"
	@echo "  {SAFETY,LLM}_URL         - Model URL"
	@echo "  {SAFETY,LLM}_API_TOKEN   - Model API token for remote models"
	@echo "  {SAFETY,LLM}_TOLERATION  - Model pod toleration"

# Create namespace and deploy
namespace:
	@oc create namespace $(NAMESPACE) &> /dev/null && oc label namespace $(NAMESPACE) modelmesh-enabled=false ||:
	@oc project $(NAMESPACE) &> /dev/null ||:

helm-install-%: namespace
	$(if $(HELM_ARGS_FUNC), $(eval HELM_ARGS := $(call $(HELM_ARGS_FUNC))))
	@echo "$*: Deploying Helm chart in namespace $(NAMESPACE)..."
	@helm upgrade --install $* ./$* -n $(NAMESPACE) $(HELM_ARGS) $(EXTRA_HELM_ARGS) > /dev/null && echo "$*: Helm chart installed successfully"

helm-uninstall-%:
	@echo "$*: Uninstalling Helm chart from namespace $(NAMESPACE)..."
	@helm uninstall $* -n $(NAMESPACE) &> /dev/null || echo "$* is not installed or already removed."

list-models-%:
	@helm template dummy-release $(LLM_SERVICE_CHART) --set _debugListModels=true --values $(LLM_SERVICE_CHART)/values-$*.yaml |grep ^model

.PHONY: list-models
list-models: list-models-gpu


helm_llm_service_args = \
    --set secret.hf_token=$(HF_TOKEN) \
    $(if $(LLM),--set models.$(LLM).enabled=true,) \
    $(if $(SAFETY),--set models.$(SAFETY).enabled=true,) \
    $(if $(LLM_TOLERATION),--set-json models.$(LLM).inferenceService.tolerations='$(call TOLERATIONS_TEMPLATE,$(LLM_TOLERATION))',) \
    $(if $(SAFETY_TOLERATION),--set-json models.$(SAFETY).inferenceService.tolerations='$(call TOLERATIONS_TEMPLATE,$(SAFETY_TOLERATION))',)

install-llm-service-%: namespace
	@$(MAKE) helm-install-$(LLM_SERVICE_CHART) HELM_ARGS_FUNC=helm_llm_service_args EXTRA_HELM_ARGS="--values=$(LLM_SERVICE_CHART)/values-$*.yaml "
	@echo "Waiting for model services to deploy. It will take around 10-15 minutes depending on the size of the model..."
	@oc wait -n $(NAMESPACE) --for=condition=Ready --timeout=60m inferenceservice --all ||:

.PHONY: install-llm-service
install-llm-service: install-llm-service-gpu


.PHONY: list-models-to-configmap
list-models-to-configmap: namespace
	@echo "Generating model ID list from values-gpu.yaml for specified models: $(LLM) $(SAFETY)..."

MODELS_JSON := $(shell \
  ids="[]"; \
  if [ -n "$(LLM)" ]; then \
    id=$$(yq e '.models.$(LLM).id' $(LLM_SERVICE_CHART)/values-gpu.yaml); \
    if [ "$$id" != "null" ]; then \
      ids=$$(echo $$ids | jq --arg id "$$id" '. + [$$id]'); \
    fi; \
  fi; \
  if [ -n "$(SAFETY)" ]; then \
    id=$$(yq e '.models.$(SAFETY).id' $(LLM_SERVICE_CHART)/values-gpu.yaml); \
    if [ "$$id" != "null" ]; then \
      ids=$$(echo $$ids | jq --arg id "$$id" '. + [$$id]'); \
    fi; \
  fi; \
  echo $$ids \
)


.PHONY: install-prometheus
install-prometheus: namespace
	@echo "Deploying Prometheus Helm release..."
	helm upgrade --install $(PROMETHEUS_RELEASE_NAME) $(PROMETHEUS_CHART_PATH) -n $(NAMESPACE) \
	   --set-json listModels.modelId.enabledModelIds='$(MODELS_JSON)'

	@echo "Waiting for Prometheus operator CSV to be Succeeded..."
	oc wait --for=jsonpath='{.status.phase}'=Succeeded csv -n $(NAMESPACE) -l operators.coreos.com/prometheus.$(NAMESPACE) --timeout=10m || \
	echo "Timed out waiting for CSV"

	@echo "Waiting for Prometheus operator controller pod to be Ready..."
	oc wait --for=condition=Ready pod -n $(NAMESPACE) -l app.kubernetes.io/component=controller --timeout=5m || \
	echo "Timed out waiting for controller pod"

	sleep 30;


.PHONY: install-metric-mcp
install-metric-mcp: namespace
	@echo "Extracting LLM service URL"
	LLM_URL=$$(oc get inferenceservice llama-3-2-3b-instruct -n $(NAMESPACE) -o jsonpath='{.status.url}'); \
	echo "Detected LLM_URL=$$LLM_URL"; \
	echo "Detected MODEL_JSON=$$MODELS_JSON"; \
	helm upgrade --install $(METRIC_MCP_RELEASE_NAME) $(METRIC_MCP_CHART_PATH) \
		-n $(NAMESPACE) \
		--set-json llm.url="\"$$LLM_URL\"" \
		--set-json listModels.modelId.enabledModelIds='$(MODELS_JSON)'

.PHONY: install-metric-ui
install-metric-ui: namespace
	@echo "Deploying Metric UI"

	helm upgrade --install $(METRIC_UI_RELEASE_NAME) $(METRIC_UI_CHART_PATH) -n $(NAMESPACE)

.PHONY: install-ui
install-ui: namespace list-models-to-configmap install-prometheus install-metric-mcp install-metric-ui
	@$(eval HELM_ARGS := $(call helm_llama_stack_args))

	@echo "Waiting for deployment to be ready..."
	@$(MAKE) wait

install-gpu: install-llm-service-gpu install-ui
install-cpu: install-llm-service-cpu install-ui

.PHONY: install
install: install-gpu

# Uninstall the deployment and clean up
.PHONY: uninstall
uninstall: $(addprefix helm-uninstall-,$(COMPONENTS)) remove-pvcs
	@echo "Deleting remaining pods in namespace $(NAMESPACE)"
	@oc delete pods -n $(NAMESPACE) --all
	@echo "Checking for any remaining resources in namespace $(NAMESPACE)..."
	@echo "If you want to completely remove the namespace, run: oc delete project $(NAMESPACE)"
	@echo "Remaining resources in namespace $(NAMESPACE):"
	@$(MAKE) status


.PHONY: remove-pvcs
remove-pvcs:
	@echo "Removing pgvector and minio PVCs"
	@oc get pvc -n $(NAMESPACE) -o custom-columns=NAME:.metadata.name | grep -E '^(pg|minio)-data' | xargs -I {} oc delete pvc -n $(NAMESPACE) {} ||:

# Check deployment status
.PHONY: status
status:
	@echo "Listing pods..."
	oc get pods -n $(NAMESPACE) || true

	@echo "Listing services..."
	oc get svc -n $(NAMESPACE) || true

	@echo "Listing routes..."
	oc get routes -n $(NAMESPACE) || true

	@echo "Listing secrets..."
	oc get secrets -n $(NAMESPACE) | grep huggingface-secret || true

	@echo "Listing pvcs..."
	oc get pvc -n $(NAMESPACE) || true

# Wait for all pods to be ready
.PHONY: wait
wait:

	@echo "Delete failed jobs in namespace $(NAMESPACE)..."
	oc get pods -n $(NAMESPACE) --field-selector=status.phase=Failed -o jsonpath='{range .items[?(@.metadata.ownerReferences[0].kind=="Job")]}{.metadata.namespace}{";"}{.metadata.name}{"\n"}{end}' | while IFS=";" read ns pod; do \
	  echo "Deleting FAILED pod $$pod from namespace $$ns"; \
	  oc delete pod "$$pod" -n "$$ns"; \
	done

	@echo "Waiting for all pods to be ready in namespace $(NAMESPACE)..."
	@end=$$(($$(date +%s)+60)); \
	while [ $$(date +%s) -lt $$end ]; do \
	  not_ready=$$(oc get pods --no-headers | grep -vE 'Running|Succeeded|Completed'); \
	  if [ -z "$$not_ready" ]; then \
	    echo "All pods are Ready or Completed."; \
		break; \
	  fi; \
	  sleep 2; \
	done; \
	echo "Timeout: Some pods are not Ready or Completed."; \

	@echo "Verifying routes are accessible..."
	@for route in $(oc get routes -n $(NAMESPACE) -o name); do \
		echo "Checking route ${route}..."; \
		host=$(oc get ${route} -n $(NAMESPACE) -o jsonpath='{.spec.host}'); \
		if [ -n "${host}" ]; then \
			echo "Route hostname: ${host}"; \
			echo "Note: Manual verification of route accessibility is recommended"; \
		else \
			echo "WARNING: No hostname found for ${route}"; \
		fi; \
	done
