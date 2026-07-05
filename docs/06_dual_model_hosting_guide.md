# Dual-Model Hosting and Integration Guide for Qwen Fine-Tuned Models

This document provides a comprehensive guide to the hardware configuration, deployment process, memory optimization techniques, and client-side usage for hosting the two fine-tuned models (**Qwen Socratic Tutor** and **Qwen Debug Coder**) concurrently on a single GPU.

---

## 1. Host Hardware Configuration

* **GPU**: 1x NVIDIA GeForce RTX 3090 Ti (Total VRAM: **24,564 MiB**)
* **CUDA / Driver**: CUDA Version 13.2 (NVIDIA Driver Version: 595.84)
* **CPU RAM**: ~116 GB RAM
* **Environment**: Cloud Docker Container (Vast.ai)

---

## 2. Hosting and Integration Process

### 2.1. LoRA Weights Merging
Before launching the server, the LoRA adapters were merged directly into the base models to optimize inference speeds and eliminate runtime adapter loading overhead:
* **Socratic Tutor (LoRA 2-module)**: Merged from `Qwen2.5-3B-Instruct` and saved at `/workspace/finetune_gemma/models/finetuned/socratic/qwen-3b-2modules-merged`.
* **Debug Coder (LoRA 7-module)**: Merged from `Qwen2.5-Coder-3B-Instruct` and saved at `/workspace/finetune_gemma/models/finetuned/debugeval/qwen-coder-3b-merged`.

### 2.2. Dual Serving Orchestration
Both models are orchestrated concurrently using the following execution script: [serve_dual_finetuned_qwen.sh](file:///workspace/code-debug-finetunemodel/scripts/serve/serve_dual_finetuned_qwen.sh).

---

## 3. Memory Optimization Techniques

To host two 3B models concurrently on a single 24GB GPU without encountering Out of Memory (OOM) errors, several critical memory constraints were implemented:

1. **VRAM Quota Allocation (`--gpu-memory-utilization 0.38`)**:
   * Restricts each vLLM engine instance to allocate a maximum of 38% of total GPU memory (~**9.6 GB VRAM** each). The two engines combined allocate ~19.2 GB VRAM, leaving a safe **5.3 GB VRAM** margin for execution.
2. **Context Length Constraint (`--max-model-len 16384`)**:
   * The Qwen base models default to a 32,768-token max context, requiring a massive KV Cache allocation on startup. Limiting context length to `16384` ensures a compact KV cache initialization while retaining more than enough length for programming assignments.
3. **Serial Boot with Polling Lock**:
   * During server initialization, vLLM compiles Triton kernels and runs model profiling/warmup, causing a temporary spike in memory usage.
   * **Solution**: The script launches the Socratic server (Port 10100) first, polls its `/v1/models` endpoint in a loop until it is fully ready and its memory consumption stabilizes, and only then starts the Debug Coder server (Port 10200).
4. **Prefix Caching (`--enable-prefix-caching`)**:
   * Caches shared token sequences (such as systemic tutoring prompts) to save memory block storage and accelerate inference.

---

## 4. Client Usage and API Integration

Host Public IP Address: **`171.240.136.207`**

| Model Name | Internal Port | Public Port | API Base URL | Model ID |
| :--- | :---: | :---: | :--- | :--- |
| **Qwen Socratic Tutor** | `10100` | **`46492`** | `http://171.240.136.207:46492/v1` | `qwen-socratic` |
| **Qwen Debug Coder** | `10200` | **`46405`** | `http://171.240.136.207:46405/v1` | `qwen-coder-sft` |

---

### 4.1. Health Check
Verify model status by calling the `/v1/models` endpoint.

* **Windows Command Prompt**:
  ```cmd
  curl.exe http://171.240.136.207:46492/v1/models
  ```
* **Linux / macOS**:
  ```bash
  curl http://171.240.136.207:46492/v1/models
  ```

---

### 4.2. Standard Chat Completion
Used when you need the complete response returned in a single JSON payload (ideal for backend processing).

* **Querying Socratic Tutor via Windows CMD**:
  ```cmd
  curl.exe http://171.240.136.207:46492/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"qwen-socratic\",\"messages\":[{\"role\":\"system\",\"content\":\"You are a Socratic tutor.\"},{\"role\":\"user\",\"content\":\"My code is running in an infinite loop while i <= n: total += i\"}],\"temperature\":0.0}"
  ```

---

### 4.3. Streaming Chat Completion
Enables token-by-token response streaming (ideal for real-time chatbot UIs). Simply add `"stream": true` to the request payload.

* **Streaming call via Windows CMD**:
  ```cmd
  curl.exe http://171.240.136.207:46492/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"qwen-socratic\",\"messages\":[{\"role\":\"user\",\"content\":\"Help me debug a division by zero error in Python\"}],\"stream\":true}"
  ```

* **Streaming call using Python client**:
  ```python
  import openai

  client = openai.OpenAI(base_url="http://171.240.136.207:46492/v1", api_key="none")

  response = client.chat.completions.create(
      model="qwen-socratic",
      messages=[{"role": "user", "content": "Help me fix a recursion depth error."}],
      stream=True
  )

  for chunk in response:
      content = chunk.choices[0].delta.content
      if content:
          print(content, end="", flush=True)
  ```
