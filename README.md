# Stop Overthinking: Unlocking Efficient Listwise Reranking with Minimal Reasoning

Official implementation of the paper: **"Stop Overthinking: Unlocking Efficient Listwise Reranking with Minimal Reasoning"**.

## Authors

- **Danyang Liu** — School of Computer Science and Technology, Beijing Institute of Technology — 3120230887@bit.edu.cn
- **Shaojie Qu** — School of Computer Science and Technology, Beijing Institute of Technology — qushaojie@bit.edu.cn
- **Kan Li\*** — School of Computer Science and Technology, Beijing Institute of Technology — likan@bit.edu.cn

## Abstract

Listwise reranking utilizing Large Language Models (LLMs) has achieved state-of-the-art retrieval effectiveness. Recently, reasoning-enhanced models have further pushed these boundaries by employing Chain-of-Thought (CoT) to perform deep comparative analysis of candidate documents. However, this performance gain comes at a prohibitive computational cost, as models often generate thousands of reasoning tokens before producing a final ranking. In this work, we investigate the relationship between reasoning length and ranking quality, revealing an **overthinking phenomenon** where extended reasoning yields diminishing returns. To address this, we propose a **Length-Regularized Self-Distillation** framework. We synthesize a dataset by sampling diverse reasoning traces from a teacher model (Rank-K) and applying a Pareto-inspired filter to select traces that achieve high ranking performance with minimal token usage. By fine-tuning on these concise, high-quality rationales, the student model learns to internalize efficient reasoning patterns, effectively pruning redundant deliberation. Experiments on TREC Deep Learning and NeuCLIR benchmarks demonstrate that our method maintains the teacher's effectiveness while reducing inference token consumption by **34%–37%** across different retrieval settings, offering a practical solution for deploying reasoning-enhanced rerankers in latency-sensitive applications.

## Keywords

Large language models, Retrieval-augmented generation, Listwise reranking, Self-distillation


## Dependencies

See [requirements.txt](requirements.txt) for the full list of dependencies.

Key dependencies:
- Python >= 3.10
- vLLM (for model inference)
- PyTorch
- Transformers
- Datasets (HuggingFace)
- Matplotlib / Seaborn (for visualization)
- NumPy / Pandas

## Citation

If you find this work helpful, please consider citing:

```bibtex
@inproceedings{liu2025stopoverthinking,
  title={Stop Overthinking: Unlocking Efficient Listwise Reranking with Minimal Reasoning},
  author={Liu, Danyang and Qu, Shaojie and Li, Kan},
  booktitle={...},
  year={2025}
}
```

## License

This project is licensed under the MIT License.
