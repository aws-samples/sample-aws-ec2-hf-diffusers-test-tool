# Qwen & Flux Image Editing Benchmark Tool

A Python-based benchmarking and inference framework for testing **AI-powered image editing pipelines** built with [🤗 Hugging Face Diffusers](https://huggingface.co/docs/diffusers).  
Supports both **Qwen Image Edit Plus** and **Flux Kontext** pipelines, including **vanilla** and **GGUF quantized** models.

---

## Overview

This project provides a command-line tool (`main.py`) and a structured Makefile to:
- Run reproducible image editing experiments
- Compare **different models** and **quantization levels**
- Collect **step-level timing metrics**
- Save **output images and performance data**

It is designed for developers, ML researchers, and performance testers who want to benchmark or visually evaluate model variations in **Qwen** and **Flux** pipelines.

---

## Features

- **Supports multiple pipelines**  
  - `QwenImageEditPlusPipeline`  
  - `FluxKontextPipeline`
  
- **Flexible configuration** via CLI arguments
  - Model path, type (`vanilla` / `GGUF`), steps, CFG, seed, etc.

- **Automatic image preprocessing**
  - Loads and pads images to target resolution.

- **Per-step timing metrics**
  - Logs execution time for each inference step to a CSV file.

- **Structured output**
  - Images in `output_images/`
  - Metrics in `output_metrics/`

## Project Structure

```
.
├── main.py              # Core benchmarking script
├── Makefile             # Automation for running multiple tests
├── output_images/       # Generated output images (auto-created)
├── output_metrics/      # CSV performance logs (auto-created)
└── README.md            # Project documentation
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/aws-samples/aws-hf-diffusers-ec2-test-tool
cd aws-hf-diffusers-ec2-test-tool
```

### 2. Create virtual enviroment and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip -r install requirements.txt
```

### 3. (Optional) Authenticate with Hugging Face
Some models require authentication and TOS acceptance:
```bash
hf auth login
```

---

## Usage

### Run directly via Python

#### Qwen (Vanilla)
```bash
python main.py   --pipeline_vendor QwenImageEditPlusPipeline   --model_type vanilla   --steps 10   --cfg 4
```

#### Flux (GGUF)
```bash
python main.py   --pipeline_vendor FluxKontextPipeline   --model_type GGUF   --steps 20   --cfg 1   --model_path "https://huggingface.co/calcuis/kontext-gguf/blob/main/flux-kontext-lite-q4_0.gguf"
```

### Example Output
```
output_images/
  ├── 18c2d3b8-...png
output_metrics/
  ├── 18c2d3b8-...csv
```

Each CSV file includes per-step performance data:
```
run_id,model_type,model_path,step,timestamp,time_sec
b91e...,GGUF,https://huggingface.co/...q4_0.gguf,1,1728954345.123,0.284321
```

---

## Makefile Usage

The included `Makefile` lets you easily run preconfigured tests.

### Run all tests
```bash
make all
```

### Run all Qwen models
```bash
make qwen_all
```

### Run all Flux models
```bash
make flux_all
```

### Run individual tests
```bash
make qwen_vanilla
make flux_gguf_q4
```

---

## Makefile Targets

| Target | Description |
|---------|-------------|
| **`all`** | Runs *all* Qwen and Flux tests sequentially. |
| **`qwen_all`** | Runs all Qwen model tests (vanilla + GGUF variants). |
| **`flux_all`** | Runs all Flux model tests (vanilla + GGUF variants). |
| **`qwen_vanilla`** | Qwen Image Edit with vanilla model. |
| **`qwen_gguf_4bit`** | Qwen GGUF model (`qwen-image-edit-plus-v2-iq4_nl.gguf`). |
| **`qwen_gguf_3bit`** | Qwen GGUF model (`qwen-image-edit-plus-v2-iq3_s.gguf`). |
| **`qwen_gguf_moe`** | Qwen GGUF model (`qwen-image-edit-plus-v2-mxfp4_moe.gguf`). |
| **`flux_vanilla`** | Flux Kontext vanilla model. |
| **`flux_gguf_default`** | Flux Kontext default GGUF test. |
| **`flux_gguf_q2`** | Flux Kontext GGUF model (`flux-kontext-lite-q2_k.gguf`). |
| **`flux_gguf_q4`** | Flux Kontext GGUF model (`flux-kontext-lite-q4_0.gguf`). |
| **`flux_gguf_q8`** | Flux Kontext GGUF model (`flux-kontext-lite-q8_0.gguf`). |

---

## Output Example

**Console Output**
```
Building QwenImageEditPlusPipeline.
Loading GGUF model.
Image saved to: output_images/aa58f...png
Metrics saved to: output_metrics/aa58f...csv
Step timings:
run_id, model_type, model_path, step, timestamp, time_sec
aa58f..., GGUF, qwen-image-edit-plus-v2-iq4_nl.gguf, 1, 1728954345.123, 0.284321
```

**CSV Metrics Example**
| step | timestamp | time_sec |
|------|------------|-----------|
| 1 | 1728954345.123 | 0.284321 |
| 2 | 1728954345.409 | 0.286751 |

---

## Notes
- Some models (e.g., Flux Kontext) require accepting their Hugging Face TOS.
- GGUF quantized models are optimized for efficiency — expect lower memory usage but slightly reduced accuracy.

---

## License

This project is open-source and released under the **MIT License**.  
Feel free to modify and extend it for your own benchmarking needs.

---

## Disclaimer

This repository is provided for research, benchmarking, and educational purposes only. It automates interactions with various open-source diffusion models (e.g., FluxContext, QuenImageEdit, etc.) but does not include or redistribute any model weights. Model downloads are handled automatically through their respective sources.

The models referenced here are each subject to their own licenses and terms of use. You are solely responsible for reviewing, understanding, and complying with the applicable licenses and any related intellectual property or usage restrictions before using these models in production or commercial environments.

This project is not affiliated with or endorsed by the authors, organizations, or license holders of any of the referenced models.

Use at your own discretion and in accordance with all applicable licenses and laws.

---

## Contributing

Pull requests and issues are welcome!  
If you add new pipelines, models, or benchmarking modes, please update both:
- `Makefile`
- `README.md` (targets + usage examples)
