# 🚀 Potential Improvements & SOTA Roadmap

**Author:** [Your Name/GitHub]
**Model:** Attention Language Model (Transformer Decoder)

This document outlines the architectural and engineering enhancements planned to transition this project from a "Vanilla" Transformer into a production-ready, Llama-style Large Language Model.

---

## 🏗️ Phase 1: Architectural Parity (The "Llama" Standard)

Current LLMs have moved away from the original 2017 Transformer paper. These changes improve training stability and inference efficiency.

* **[ ] Rotary Positional Embeddings (RoPE):** Replace absolute/learned positional embeddings with RoPE. This allows for better "length extrapolation" (handling sequences longer than the training window).
* *Reference:* [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)


* **[ ] RMSNorm:** Replace `LayerNorm` with `Root Mean Square Layer Normalization`. It is computationally cheaper and provides similar (or better) stability by removing the "mean centering" step.
* **[ ] SwiGLU Activation:** Replace `ReLU` or `GeLU` in the Feed-Forward layers with SwiGLU. This is the gated activation function used in Llama 3 to improve gradient flow.
* **[ ] Weight Tying:** Bind the weights of the input `Embedding` layer and the final `LM_Head` (Output) layer. This reduces parameter count by $\sim 50 \text{M}$ (for 100k vocab) and enforces consistent semantic representation.

---

## ⚡ Phase 2: Compute & Memory Optimization

To train on larger datasets within a limited budget (e.g., €100), efficiency is the primary constraint.

* **[ ] Grouped-Query Attention (GQA):** Reduce the number of Key/Value heads relative to Query heads. This significantly shrinks the **KV Cache** footprint, enabling larger batch sizes and longer context windows.
* *SOTA Milestone:* Used in Llama-3 8B/70B.


* **[ ] FlashAttention-2 / 3 integration:** Use `torch.nn.functional.scaled_dot_product_attention` to leverage kernel-level fusions.
* **Goal:** Reach >70% GPU Model FLOPs Utilization (MFU).


* **[ ] Mixed Precision Training (Bfloat16):** Use `torch.amp` to train in BF16. This halves memory usage and speeds up training on Ampere/Hopper GPUs while maintaining better stability than standard FP16.
* **[ ] Gradient Accumulation:** Simulate large batch sizes (e.g., 512) on consumer hardware by accumulating gradients over multiple small steps.

---

## 🧠 Phase 3: Advanced Inference

Serving the model is as important as training it.

* **[ ] Static KV-Caching:** Implement a pre-allocated, in-place update KV-cache.
* *Benefit:* Enables `torch.compile` to fuse the entire generation loop into a single CUDA graph, reducing "Time-To-First-Token" (TTFT) by up to 4x.


* **[ ] Speculative Decoding:** Use a "Tiny" version of this model (e.g., 10M params) to draft tokens that the "Big" model (e.g., 124M params) verifies in parallel.
* **[ ] 4-bit / 8-bit Quantization:** Showcase post-training quantization (PTQ) to run the model on extremely low-memory devices (mobile/edge).

---

## 📊 Phase 4: Data & Alignment (The "Secret Sauce")

Frontier models win because of their data, not just their code.

* **[ ] Synthetic Data Pipeline:** Use a "Teacher" model (like GPT-4o) to rephrase raw web-scrapes into "Textbook-style" educational content for pretraining.
* **[ ] Curriculum Learning:** Start training on simple, short sequences (e.g., 128 tokens) and progressively "unlock" the full 1024+ context window as the model converges.
* **[ ] Supervised Fine-Tuning (SFT):** Fine-tune the base model on a "Chain-of-Thought" (CoT) dataset to enable basic reasoning and instruction-following.

---

## 📈 Success Metrics for the Showcase

To prove these improvements work, the repo will track:

1. **Tokens Per Second (TPS):** Comparison of Naive vs. KV-Cached generation.
2. **Memory Floor:** VRAM usage with vs. without GQA.
3. **Loss Scaling:** A graph showing how SwiGLU/RoPE reach a lower perplexity faster than the Vanilla architecture.

