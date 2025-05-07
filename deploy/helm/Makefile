# Makefile for AI Metric Summarizer Deployment
# Replaces the original deploy.sh script with additional uninstall functionality

ifeq ($(NAMESPACE),)
ifeq (,$(filter list-models% help,$(MAKECMDGOALS)))
$(error NAMESPACE is not set)
endif
endif

# Default values
HF_TOKEN ?= $(shell bash -c 'read -r -p "Enter Hugging Face Token: " HF_TOKEN; echo $$HF_TOKEN')
LLM_SERVICE_CHART_PATH ?= llm-service
LLM_SERVICE_RELEASE_NAME ?= llm-service
METRIC_UI_CHART_PATH ?= metric-ui
METRIC_UI_RELEASE_NAME ?= metric-ui
PROMETHEUS_CHART_PATH ?= prometheus
PROMETHEUS_RELEASE_NAME ?= prometheus
TOLERATIONS_TEMPLATE=[{"key":"$(1)","effect":"NoSchedule","operator":"Exists"}]

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
	@echo "  RELEASE_NAME             - Helm release name (default: rag)"
	@echo "  CHART_PATH               - Path to the Helm chart (default: rag-ui)"
	@echo "  HF_TOKEN                 - Hugging Face Token (will prompt if not provided)"
	@echo "  {SAFETY,LLM}             - Model id as defined in values (eg. llama-3-2-1b-instruct)"
	@echo "  {SAFETY,LLM}_URL         - Model URL"
	@echo "  {SAFETY,LLM}_API_TOKEN   - Model API token for remote models"
	@echo "  {SAFETY,LLM}_TOLERATION  - Model pod toleration"

# Create namespace and deploy
.PHONY: namespace
namespace:
	@echo "Creating namespace $(NAMESPACE)..."
	oc create namespace $(NAMESPACE) && oc label namespace $(NAMESPACE) modelmesh-enabled=false || oc project $(NAMESPACE) ||:

list-models-%:
	@helm template dummy-release $(LLM_SERVICE_CHART_PATH) --set _debugListModels=true --values $(LLM_SERVICE_CHART_PATH)/values-$*.yaml |grep ^model

.PHONY: list-models
list-models: list-models-gpu

helm_llm_service_args = \
    $(if $(LLM),--set-json models.$(LLM).enabled='true',) \
    $(if $(SAFETY),--set-json models.$(SAFETY).enabled='true',) \
    $(if $(LLM_TOLERATION),--set-json models.$(LLM).inferenceService.tolerations='$(call TOLERATIONS_TEMPLATE,$(LLM_TOLERATION))',) \
    $(if $(SAFETY_TOLERATION),--set-json models.$(SAFETY).inferenceService.tolerations='$(call TOLERATIONS_TEMPLATE,$(SAFETY_TOLERATION))',)

install-llm-service-%: namespace
	@$(eval HELM_ARGS := $(call helm_llm_service_args))

	@echo "Deploying Helm chart $(LLM_SERVICE_CHART_PATH) as release $(LLM_SERVICE_RELEASE_NAME) in namespace $(NAMESPACE)..."; \
	helm upgrade --install $(LLM_SERVICE_RELEASE_NAME) $(LLM_SERVICE_CHART_PATH) -n $(NAMESPACE) --values $(LLM_SERVICE_CHART_PATH)/values-$*.yaml \
		--set hf_token=$(HF_TOKEN) $(HELM_ARGS) $(EXTRA_HELM_ARGS)
	@echo "Waiting for model services to deploy. It will take around 10-15 minutes depending on the size of the model..."
	oc wait -n $(NAMESPACE) --for=condition=Ready --timeout=60m inferenceservice --all ||:

.PHONY: install-llm-service
install-llm-service: install-llm-service-gpu

.PHONY: list-models-to-configmap
list-models-to-configmap: namespace
	@echo "Generating model ID list from values-gpu.yaml for specified models: $(LLM) $(SAFETY)..."

MODELS_JSON := $(shell \
  ids="[]"; \
  if [ -n "$(LLM)" ]; then \
    id=$$(yq e '.models.$(LLM).id' $(LLM_SERVICE_CHART_PATH)/values-gpu.yaml); \
    if [ "$$id" != "null" ]; then \
      ids=$$(echo $$ids | jq --arg id "$$id" '. + [$$id]'); \
    fi; \
  fi; \
  if [ -n "$(SAFETY)" ]; then \
    id=$$(yq e '.models.$(SAFETY).id' $(LLM_SERVICE_CHART_PATH)/values-gpu.yaml); \
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


.PHONY: install-metric-ui
install-metric-ui: namespace
	@echo "Extracting LLM service URL"
	LLM_URL=$$(oc get inferenceservice llama-3-2-3b-instruct -n $(NAMESPACE) -o jsonpath='{.status.url}'); \
	echo "Detected LLM_URL=$$LLM_URL"; \
	echo "Detected MODEL_JSON=$$MODELS_JSON"; \
	helm upgrade --install $(METRIC_UI_RELEASE_NAME) $(METRIC_UI_CHART_PATH) \
		-n $(NAMESPACE) \
		--set-json llm.url="\"$$LLM_URL\"" \
		--set-json listModels.modelId.enabledModelIds='$(MODELS_JSON)'


.PHONY: install-ui
install-ui: namespace list-models-to-configmap install-prometheus install-metric-ui
	@$(eval HELM_ARGS := $(call helm_llama_stack_args))

	@echo "Waiting for deployment to be ready..."
	@$(MAKE) wait

install-%: install-llm-service-% install-ui
	@echo "Installed from target install-$*"

.PHONY: install
install: install-gpu

# Uninstall the deployment and clean up
.PHONY: uninstall
uninstall: uninstall-helm-release uninstall-llm-service uninstall-prometheus uninstall-metric-ui uninstall-configmap remove-secrets remove-pvcs
	@echo "Checking for any remaining resources in namespace $(NAMESPACE)..."
	@echo "If you want to completely remove the namespace, run: oc delete project $(NAMESPACE)"
	@echo "Remaining resources in namespace $(NAMESPACE):"
	@$(MAKE) status


.PHONY: uninstall-llm-service
uninstall-llm-service:

	@echo "Uninstalling Helm release $(LLM_SERVICE_RELEASE_NAME) from namespace $(NAMESPACE)..."
	helm uninstall $(LLM_SERVICE_RELEASE_NAME) -n $(NAMESPACE) || echo "LLM services are not installed or already removed."

.PHONY: uninstall-prometheus
uninstall-prometheus:
	@echo "Uninstalling Prometheus Helm release..."
	helm uninstall $(PROMETHEUS_RELEASE_NAME) -n $(NAMESPACE) || echo "Prometheus not installed or already removed."

.PHONY: uninstall-metric-ui
uninstall-metric-ui:
	@echo "Uninstalling Metric UI Helm release..."
	helm uninstall $(METRIC_UI_RELEASE_NAME) -n $(NAMESPACE) || echo "Metric UI not installed or already removed."

.PHONY: uninstall-configmap
uninstall-configmap:
	@echo "Deleting 'llm-models' ConfigMap..."
	oc delete configmap llm-models -n $(NAMESPACE) || echo "ConfigMap not found or already removed."


.PHONY: uninstall-helm-release
uninstall-helm-release:

	@echo "Uninstalling Helm release $(RELEASE_NAME) from namespace $(NAMESPACE)..."
	helm uninstall $(RELEASE_NAME) -n $(NAMESPACE) || echo "Helm release not found or already removed."


.PHONY: remove-pvcs
remove-pvcs:

	@echo "Removing pgvector and minio PVCs"
	oc delete pvc -n $(NAMESPACE) $$(oc get pvc -n $(NAMESPACE) -o custom-columns=NAME:.metadata.name --no-headers | grep "^pg-data")
	oc delete pvc -n $(NAMESPACE) $$(oc get pvc -n $(NAMESPACE) -o custom-columns=NAME:.metadata.name --no-headers | grep "^minio-data")

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