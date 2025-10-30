# Makefile for running different model tests
PYTHON := python
MAIN := main.py

# Default parameters
STEPS := 10
CFG := 4

# Qwen Image Edit test		s
qwen_vanilla:
	$(PYTHON) $(MAIN) --pipeline_vendor QwenImageEditPlusPipeline \
		--steps $(STEPS) --cfg $(CFG) \
		--model_type vanilla \
		--model_path "ovedrive/Qwen-Image-Edit-2509-4bit"

qwen_gguf_4bit:
	$(PYTHON) $(MAIN) --pipeline_vendor QwenImageEditPlusPipeline \
		--steps $(STEPS) --cfg $(CFG) \
		--model_type GGUF \
		--model_path "https://huggingface.co/calcuis/qwen-image-edit-plus-gguf/blob/main/qwen-image-edit-plus-v2-iq4_nl.gguf"

qwen_gguf_3bit:
	$(PYTHON) $(MAIN) --pipeline_vendor QwenImageEditPlusPipeline \
		--steps $(STEPS) --cfg $(CFG) \
		--model_type GGUF \
		--model_path "https://huggingface.co/calcuis/qwen-image-edit-plus-gguf/blob/main/qwen-image-edit-plus-v2-iq3_s.gguf"

qwen_gguf_moe:
	$(PYTHON) $(MAIN) --pipeline_vendor QwenImageEditPlusPipeline \
		--steps $(STEPS) --cfg $(CFG) \
		--model_type GGUF \
		--model_path "https://huggingface.co/calcuis/qwen-image-edit-plus-gguf/blob/main/qwen-image-edit-plus-v2-mxfp4_moe.gguf"

# Flux Kontext tests
flux_vanilla:
	$(PYTHON) $(MAIN) --pipeline_vendor FluxKontextPipeline \
		--cfg 1 --steps 20 \
		--model_type vanilla

flux_gguf_default:
	$(PYTHON) $(MAIN) --pipeline_vendor FluxKontextPipeline \
		--model_type GGUF

flux_gguf_q2:
	$(PYTHON) $(MAIN) --pipeline_vendor FluxKontextPipeline \
		--model_type GGUF --steps 20 --cfg 1 \
		--model_path "https://huggingface.co/calcuis/kontext-gguf/blob/main/flux-kontext-lite-q2_k.gguf"

flux_gguf_q4:
	$(PYTHON) $(MAIN) --pipeline_vendor FluxKontextPipeline \
		--model_type GGUF --steps 20 --cfg 1 \
		--model_path "https://huggingface.co/calcuis/kontext-gguf/blob/main/flux-kontext-lite-q4_0.gguf"

flux_gguf_q8:
	$(PYTHON) $(MAIN) --pipeline_vendor FluxKontextPipeline \
		--model_type GGUF --steps 20 --cfg 1 \
		--model_path "https://huggingface.co/calcuis/kontext-gguf/blob/main/flux-kontext-lite-q8_0.gguf"

# Run all Qwen tests
qwen_all: qwen_vanilla qwen_gguf_4bit qwen_gguf_3bit qwen_gguf_moe

# Run all Flux tests
flux_all: flux_vanilla flux_gguf_default flux_gguf_q2 flux_gguf_q4 flux_gguf_q8

# Run everything
all: qwen_all flux_all
	@echo "All tests completed successfully!"

.PHONY: all qwen_all flux_all \
	qwen_vanilla qwen_gguf_4bit qwen_gguf_3bit qwen_gguf_moe \
	flux_vanilla flux_gguf_default flux_gguf_q2 flux_gguf_q4 flux_gguf_q8
