# Effective Rank as a Diagnostic for Representation Bottlenecks and Compression in Language Models

An empirical study of layerwise representation geometry in transformer language models, connecting **effective rank** and **Gaussian mutual information** to the **Information Bottleneck** (IB) framework. We analyze how individual MLP layers compress and transmit information across four pretrained LLMs, and use these diagnostics to guide structured low-rank compression.

> NYU DS-GA 3001: Information-Theoretic Perspectives on Cognition (Spring 2026)

## Key Findings

### 1. Effective Rank Reveals Layer-Level Compression Structure

Each layer's MLP output has a well-defined **effective rank** (exponential of the spectral entropy of its covariance), which varies substantially across depth. Early and late layers tend toward lower effective rank, while middle layers maintain higher-dimensional representations. This pattern is consistent across all four models and provides a per-layer proxy for representation complexity, i.e., *I(T; X)* in the IB framework.

### 2. Gaussian MI Shows a Two-Phase Pattern

We measure the Gaussian mutual information *I(MLP_output_l; MLP_output_L)* between each layer's MLP output and the final layer's MLP output via canonical correlation analysis (CCA). Across all models, this exhibits:

- **Flat plateau** (early/middle layers): each MLP's output is only moderately correlated with the final representation.
- **Exponential rise** (final ~30-40% of depth): MI increases steeply as successive MLPs become increasingly aligned with the output.

Sharp MI drops occur at layers with **near-rank-1 MLP output** (effective rank ~ 1), where the MLP contributes an essentially one-dimensional signal to the residual stream.

### 3. Residual Stream MI Is Smooth and Monotonic

The residual stream forms a Markov chain *X -> h_0 -> h_1 -> ... -> h_L*, so by the Data Processing Inequality, *I(h_l; h_L)* must increase with depth. Empirically, residual stream MI increases smoothly even at layers where MLP output MI drops, confirming that **skip connections preserve information past degenerate MLP layers**.

### 4. Empirical Evidence for the IB Tradeoff

Late layers simultaneously show **rising MI** (increasing informativeness about the output) and **declining effective rank** (increasing compression). This is the compression-informativeness tradeoff predicted by the Information Bottleneck principle (Tishby & Zaslavsky, 2015), observed here empirically in trained LLMs without any IB-explicit training objective.

### 5. Rank-1 Layers Are Maximally Compressible

Layers whose MLP output has effective rank ~ 1 are compressible by every measure: lowest representation complexity, lowest MI with the final output, and empirically removable with minimal perplexity impact. These appear in every model tested, typically in the first few layers.

### 6. Layerwise Rank Budgets Outperform Uniform Compression

A compression scheme that allocates rank budgets per layer based on a weighted combination of structural rank and activation importance consistently outperforms uniform SVD compression at matched parameter counts.

## Models

| Model | Parameters | Layers | Hidden Size | HuggingFace ID |
|-------|-----------|--------|-------------|----------------|
| Qwen3-0.6B | 0.6B | 28 | 1024 | `Qwen/Qwen3-0.6B` |
| Qwen2.5-1.5B | 1.5B | 28 | 1536 | `Qwen/Qwen2.5-1.5B` |
| SmolLM2-1.7B | 1.7B | 24 | 2048 | `HuggingFaceTB/SmolLM2-1.7B` |
| Llama-3.2-1B | 1.0B | 16 | 2048 | `meta-llama/Llama-3.2-1B` |

All statistics are computed on 65,536 token activations from the WikiText-2 training set.

## Pipeline Overview

```
Stage 0 ── Download models & data
Stage 1 ── Collect layerwise statistics (effective rank, trace, eigenvalues)
Stage 2 ── Plot per-model compressibility & importance profiles
Stage 3 ── Compute rank budgets (3a) & build compressed models (3b)
Stage 4 ── Benchmark perplexity of compressed variants
Stage 5 ── Cross-model analysis: layer EDA (5a) & compression results (5b)
Stage 6 ── Mutual information: MLP output (6), residual stream (6b), comparison (6c)
Stage 7 ── Information Bottleneck visualization
```

## Setup

### Requirements

- Python 3.10+
- PyTorch 2.0+
- `transformers`, `huggingface_hub`, `datasets`
- `matplotlib`
- `tqdm`

```bash
pip install torch transformers huggingface_hub datasets matplotlib tqdm
```

### Download Models and Data

```bash
# Download dataset
python script/0_download_data.py

# Download models (Llama requires a HuggingFace token with gated access)
python script/0_download_model.py --model-id Qwen/Qwen3-0.6B
python script/0_download_model.py --model-id Qwen/Qwen2.5-1.5B
python script/0_download_model.py --model-id HuggingFaceTB/SmolLM2-1.7B
python script/0_download_model.py --model-id meta-llama/Llama-3.2-1B --token YOUR_HF_TOKEN
```

Models are saved to `artifact/models/<base_model_id>/base/`.

## Running the Full Pipeline

All scripts auto-discover paths via `script/_model_layout.py`. The `--model-path` argument defaults to Qwen3-0.6B; override it for other models.

### Stage 1: Collect Layer Statistics

```bash
python script/1_collect_statistics.py --model-path artifact/models/Qwen3-0.6B/base
python script/1_collect_statistics.py --model-path artifact/models/Qwen2.5-1.5B/base
python script/1_collect_statistics.py --model-path artifact/models/SmolLM2-1.7B/base
python script/1_collect_statistics.py --model-path artifact/models/Llama-3.2-1B/base
```

Outputs per model: `result/<model>/1_statistics/layer_statistics.csv` plus cached covariance tensors.

### Stage 2: Plot Per-Model Profiles

```bash
python script/2_plot_statistics.py --base-model-dir artifact/models/Qwen3-0.6B/base
python script/2_plot_statistics.py --base-model-dir artifact/models/Qwen2.5-1.5B/base
python script/2_plot_statistics.py --base-model-dir artifact/models/SmolLM2-1.7B/base
python script/2_plot_statistics.py --base-model-dir artifact/models/Llama-3.2-1B/base
```

### Stage 3: Compression

```bash
# 3a: Compute rank budgets
python script/3_compute_rank_budgets.py --base-model-dir artifact/models/Qwen3-0.6B/base

# 3b: Build compressed models
python script/3_compress_models.py --base-model-dir artifact/models/Qwen3-0.6B/base
```

Repeat for each model. Compression sweeps over `alpha` (compression level) and `w` (rank vs. importance weight).

### Stage 4: Benchmark Perplexity

```bash
python script/4_benchmark_perplexity.py --base-model-dir artifact/models/Qwen3-0.6B/base
```

### Stage 5: Cross-Model Analysis

```bash
# 5a: Layer-level EDA across models (auto-discovers all statistics CSVs)
python script/5_layer_analysis.py

# 5b: Compression results visualization
python script/5_compression_analysis.py
```

### Stage 6: Mutual Information Analysis

```bash
# 6: MLP output MI (per model)
python script/6_mutual_information.py --model-path artifact/models/Qwen3-0.6B/base
python script/6_mutual_information.py --model-path artifact/models/Qwen2.5-1.5B/base
python script/6_mutual_information.py --model-path artifact/models/SmolLM2-1.7B/base
python script/6_mutual_information.py --model-path artifact/models/Llama-3.2-1B/base

# 6b: Residual stream MI (per model)
python script/6b_mutual_information_residual.py --model-path artifact/models/Qwen3-0.6B/base
python script/6b_mutual_information_residual.py --model-path artifact/models/Qwen2.5-1.5B/base
python script/6b_mutual_information_residual.py --model-path artifact/models/SmolLM2-1.7B/base
python script/6b_mutual_information_residual.py --model-path artifact/models/Llama-3.2-1B/base

# 6c: Comparison plots (auto-discovers)
python script/6c_mi_comparison_plot.py
```

### Stage 7: Information Bottleneck Visualization

```bash
python script/7_information_plane.py
```

Produces the information plane scatter, MI/rank trend plots, and empirical IB tradeoff visualization.

## Project Structure

```
.
├── script/
│   ├── _model_layout.py                  # Shared path configuration
│   ├── 0_download_model.py               # Download base models
│   ├── 0_download_data.py                # Download WikiText-2
│   ├── 1_collect_statistics.py           # Layerwise statistics collection
│   ├── 2_plot_statistics.py              # Per-model visualization
│   ├── 3_compute_rank_budgets.py         # Rank budget computation
│   ├── 3_compress_models.py              # SVD-based model compression
│   ├── 4_benchmark_perplexity.py         # Perplexity evaluation
│   ├── 5_layer_analysis.py               # Cross-model layer EDA
│   ├── 5_compression_analysis.py         # Compression results analysis
│   ├── 6_mutual_information.py           # Gaussian MI (MLP output)
│   ├── 6b_mutual_information_residual.py # Gaussian MI (residual stream)
│   ├── 6c_mi_comparison_plot.py          # MLP vs residual MI comparison
│   └── 7_information_plane.py            # Information Bottleneck plots
├── artifact/models/                      # Downloaded & compressed models
├── data/wikitext2/                       # Calibration data
├── result/
│   ├── <model>/1_statistics/             # Per-model layer statistics
│   ├── <model>/2_rank_analysis/          # Per-model rank/importance plots
│   ├── <model>/3_compression/            # Rank budgets
│   ├── <model>/4_benchmark/              # Perplexity benchmarks
│   ├── <model>/6_mutual_information/     # MLP output MI
│   ├── <model>/6b_mutual_information_residual/  # Residual stream MI
│   ├── 5_layer_analysis/                 # Cross-model layer analysis
│   ├── 5_compression_analysis/           # Cross-model compression plots
│   ├── 6c_mi_comparison/                # MI comparison plots
│   └── 7_information_plane/              # IB analysis plots
└── mi_analysis_findings.tex              # Detailed MI findings document
```

## Methodology

### Effective Rank

For a representation matrix with covariance eigenvalues {lambda_i}, the effective rank is:

```
erank = exp(H(p))    where  p_i = lambda_i / sum(lambda_j)
```

This measures the "effective dimensionality" of the representation, ranging from 1 (all variance in one direction) to the full dimension (uniform spectrum).

### Gaussian Mutual Information

Under the Gaussian approximation, MI between layer l's representation z_l and the final layer's representation z_L is computed via two equivalent methods:

- **CCA method** (primary for MLP output): `I = -(1/2) sum_i log(1 - rho_i^2)`, where rho_i are canonical correlations. Uses truncated whitening to handle near-singular covariances.
- **Log-det method** (primary for residual stream): `I = (1/2)(log det Sigma_l + log det Sigma_L - log det Sigma_joint)`. More stable when the representation has low effective rank.

Gaussian MI provides a **lower bound** on true MI and yields reliable relative ordering between layers even when the Gaussian assumption is approximate.

### Compression

MLP weight matrices are compressed via low-rank approximation:

```
protection_score = w * rank_signal + (1 - w) * importance_signal
```

where `rank_signal` is derived from input/output effective rank ratios and `importance_signal` from MLP output trace magnitude. Higher protection scores receive larger rank budgets. Three methods are compared: plain SVD, activation-aware SVD (uniform), and activation-aware SVD (layerwise budgets).

## References

- Tishby, N. & Zaslavsky, N. (2015). "Deep Learning and the Information Bottleneck Principle." *IEEE Information Theory Workshop*. [arXiv:1503.02406](https://arxiv.org/abs/1503.02406)
- Shwartz-Ziv, R. & Tishby, N. (2017). "Opening the Black Box of Deep Neural Networks via Information." [arXiv:1703.00810](https://arxiv.org/abs/1703.00810)
- Chechik, G. et al. (2005). "Information Bottleneck for Gaussian Variables." *JMLR*, 6, 165-188.
- Roy, O. & Vetterli, M. (2007). "The effective rank: A measure of effective dimensionality." *European Signal Processing Conference*.
