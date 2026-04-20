# Effective Rank as a Diagnostic for Representation Bottlenecks and Compression in Language Models

**Peng Zhao**  
Center for Data Science, New York University  
pz2110@nyu.edu

**Fergie Yang**  
Center for Data Science, New York University  
yy5732@nyu.edu

**Instructor:** Noga Zaslavsky  
**Teaching Assistant:** Moufan Li

**Course:** Spring 2026 | PSYCH-GA 3505; DS-GA 3001 | Special Topics Seminar: Information Theory and Cognition

---

## Abstract

Large language models (LLMs) are commonly trained in highly overparameterized regimes and compressed only after training, often through post-training quantization. Relevant practical baselines include `llama.cpp` quantization and AWQ (Activation-Aware Weight Quantization). A central open question is how much of the model's parameter space is actually used by the trained model, both in its intermediate representations and in its weights.

Recent work suggests that **effective rank**, derived from spectral entropy, can serve as a useful diagnostic of representation structure and model capacity. This project asks:

> **Can the effective rank of each layer's representations and weights reliably diagnose representation bottlenecks and predict compression potential, especially under lower-bit quantization?**

Our plan is to evaluate the effective rank of both the representation spectrum and the weight spectrum in a small language model, beginning with **Qwen 3 0.6B**, and then use these quantities to guide diagnosis and compression. We will compare compressed and uncompressed variants and visualize how effective rank changes across layers. The expected outcome is a set of empirical guidelines for using effective rank as a diagnostic for bottlenecks and compression in language models, along with practical compression heuristics. A further theoretical direction is to investigate conditions under which low effective rank implies good compression performance.

---

## 1. Introduction

### Background

A recurring pattern in modern language modeling is that models are trained with more parameters than may ultimately be necessary, then compressed afterwards. In practice, this appears through post-training quantization and related inference-time optimization methods. In theory, overparameterization is also often associated with smoother optimization and improved training dynamics.

This motivates a basic but important question:

> **How much of the available parameter space is actually utilized by a trained model, both in its weights and in its intermediate activations?**

If trained networks use fewer effective directions than their full architecture suggests, then the spectra of their weights and activations may reveal this hidden lower-dimensional structure. Effective rank provides a compact way to summarize that structure. Layerwise patterns in effective rank may help us understand:

- which layers are capacity-limited,
- which layers are overparameterized,
- where low-rank approximation is plausible, and
- where aggressive quantization is likely to work well.

In this sense, effective dimensionality may connect **interpretability**, **capacity analysis**, and **compression**.

### Inspirations

- Importance-matrix style quantization in `llama.cpp`
- AWQ (Activation-Aware Weight Quantization)
- Spectral entropy and effective rank

---

## 2. Core Research Question

We study whether **layerwise effective rank** can be used as a practical signal for:

1. diagnosing representation bottlenecks,
2. identifying layers that are overparameterized,
3. predicting where low-rank approximation should work well, and
4. guiding quantization-bit allocation across layers.

A useful framing is:

- **Representation effective rank** measures how many directions are meaningfully used by the input to a layer.
- **Weight effective rank** measures how many directions are meaningfully used by the operator itself.

Comparing the two may reveal mismatch between incoming signal complexity and operator capacity.

---

## 3. Mathematical Setup

Consider one layer of a neural network:

$$
y = \phi(Wu), \qquad u = \mathrm{Norm}(x),
$$

where:

- $x$ is the activation from the previous layer,
- $u$ is the normalized activation,
- $\mathrm{Norm}(\cdot)$ is the normalization used by the model,
- $W$ is the layer weight matrix,
- $\phi$ is the nonlinearity or downstream transformation.

### Activation Spectrum

Given normalized activations $u_1, u_2, \dots, u_n$, stack them into a matrix:

$$
U = [u_1, u_2, \dots, u_n]^\top.
$$

Then compute the activation covariance spectrum:

$$
U^\top U = Q \Lambda Q^\top,
$$

where the eigenvalues are $\lambda_1, \lambda_2, \dots$.

### Weight Spectrum

For the weight matrix $W$, compute the singular value decomposition:

$$
W = U \Sigma V^\top,
$$

with singular values $\sigma_1, \sigma_2, \dots$.

### Spectral Entropy and Effective Rank

Define normalized spectral masses:

$$
p_i = \frac{\lambda_i}{\sum_j \lambda_j}
\quad \text{or} \quad
p_i = \frac{\sigma_i}{\sum_j \sigma_j}.
$$

Then define spectral entropy:

$$
H(p) = - \sum_i p_i \log p_i,
$$

and effective rank:

$$
r = e^{H(p)}.
$$

We use:

- $r_u$ for the effective rank of the representation spectrum,
- $r_w$ for the effective rank of the weight spectrum.

---

## 4. Hypotheses

We hypothesize that the relationship between $r_u$ and $r_w$ can guide diagnosis and compression:

1. **Large $r_u$, small $r_w$** may indicate a bottleneck.  
   The incoming representation is high-dimensional relative to the operator's effective capacity.

2. **Small $r_u$, small $r_w$** may indicate a well-matched layer with compression potential.  
   The layer may already be operating on a compact signal and may tolerate aggressive compression.

3. **Small $r_u$, large $r_w$** may indicate overcapacity in the weights.  
   The operator may be more expressive than needed for the incoming signal, suggesting potential for low-rank or low-bit compression.

4. **Layerwise rank profiles** may not be uniform.  
   Some stages of the transformer may exhibit systematically higher or lower effective rank, which could motivate non-uniform compression policies.

---

## 5. Research Plan

### Step 1: Choose a Small Language Model

Use a small language model as the main experimental target, beginning with **Qwen 3 0.6B**.

### Step 2: Collect Activations and Weights

- Run the model on a representative text corpus.
- Collect normalized activations from each layer.
- Stack those activations into matrices for covariance computation.
- Extract the trained weight matrices from relevant submodules.

### Step 3: Compute Spectra

For each selected layer:

- compute the eigenvalue spectrum of the activation covariance,
- compute the singular value spectrum of the weight matrix,
- compute spectral entropy and effective rank for both.

### Step 4: Use Effective Rank to Guide Diagnosis

Interpret $r_u$ and $r_w$ jointly as signals of:

- bottlenecked layers,
- well-matched layers,
- overparameterized layers,
- candidate layers for compression.

### Step 5: Compression Experiments

#### A. Low-Rank Approximation

Test direct low-rank factorization guided by effective rank. For example:

$$
Wu \approx W V_{r_u} V_{r_u}^\top u,
$$

where $V_{r_u}$ contains the top-$r_u$ directions used for approximation.

#### B. Quantization Allocation

Use effective rank as a signal for assigning different quantization levels by layer. For example:

- use lower-bit quantization such as **Q3** for layers with low effective rank,
- use higher-bit quantization such as **Q5** for layers with higher effective rank.

### Step 6: Evaluate the Compressed Model

Compare compressed and uncompressed models using:

- **model size**,
- **tokens per second**,
- **perplexity**,
- optionally qualitative text-generation sanity checks.

Also visualize how $r_u$ and $r_w$ evolve across layers.

---

## 6. Evaluation Plan

### Primary Metrics

- **Perplexity:** measures language-model quality degradation after compression.
- **Model size on disk:** quantifies storage savings.
- **Throughput (tokens/sec):** captures inference-speed gains.
- **Effective rank profile:** provides the main diagnostic signal.

### Secondary Diagnostics

- reconstruction error for low-rank approximations,
- layerwise sensitivity to quantization,
- correlation between effective rank and compression-induced loss.

### Key Comparisons

We will compare:

1. **Uncompressed baseline**
2. **Uniform quantization baseline**
3. **Rank-guided quantization policy**
4. **Low-rank approximation baseline**
5. **Hybrid strategy** (if time permits)

---

## 7. Added Implementation Notes

The original proposal is conceptually strong, but the project will be much easier to execute if the implementation scope is narrowed early.

### Recommended Minimal First Milestone

Before any compression experiments, aim to complete this baseline:

1. load the model,
2. collect layerwise activations on a small evaluation corpus,
3. compute $r_u$ and $r_w$ for every layer,
4. produce one clean layerwise visualization,
5. compute baseline perplexity.

That gives a complete "rank-diagnostic pipeline" before introducing compression.

### Suggested Experimental Order

1. **Baseline introspection**  
   Compute spectra and effective-rank plots with no compression.

2. **Simple low-rank experiments**  
   Replace selected weight matrices with truncated approximations.

3. **Uniform quantization baseline**  
   Apply the same quantization level to all candidate layers.

4. **Rank-guided quantization**  
   Allocate bits by layer based on effective-rank statistics.

5. **Analysis and interpretation**  
   Relate compression outcomes back to the observed rank patterns.

### What to Keep Fixed Early

To make the results interpretable, keep these fixed in early experiments:

- one model,
- one main dataset or corpus,
- one activation-sampling procedure,
- one perplexity-evaluation pipeline,
- one definition of effective rank.

Changing too many variables at once will make it hard to tell whether the diagnostic is actually useful.

---

## 8. Added Deliverables

A strong final project should ideally produce the following deliverables:

### Core Deliverables

- a reproducible code pipeline for collecting layerwise activations,
- a script or notebook for computing effective rank,
- plots of $r_u$ and $r_w$ across layers,
- baseline perplexity and speed measurements,
- compression results for at least one low-rank or quantization strategy.

### Final Report Deliverables

- a clear statement of the main hypothesis,
- a concise method section,
- one table of metrics,
- one or two rank-profile figures,
- a short discussion of what effective rank did or did not predict.

### Stretch Deliverables

- mixed-bit quantization policy based on rank,
- theoretical note on why low effective rank may imply compressibility,
- ablation over activation sample size or corpus choice.

---

## 9. Added Risks and Mitigations

### Risk 1: Effective rank may be noisy

**Issue:** rank estimates may depend heavily on the number of activation samples collected.  
**Mitigation:** run a small stability check by varying sample count.

### Risk 2: Activation choice may be ambiguous

**Issue:** it matters whether activations are taken before or after normalization, residual addition, or nonlinearity.  
**Mitigation:** define the hook points explicitly and keep them fixed throughout the main experiments.

### Risk 3: Compression effects may be small at this scale

**Issue:** on a small model, some compression gains may be modest or hard to interpret.  
**Mitigation:** emphasize the diagnostic relationship between rank and layer behavior rather than only absolute gains.

### Risk 4: Too many baselines may dilute the project

**Issue:** comparing too many compression schemes can consume time without deepening the central story.  
**Mitigation:** prioritize one strong baseline and one rank-guided method.

---

## 10. Added Milestone Checklist

### Milestone 1 - Environment and Baseline

- [ ] Download and load the model
- [ ] Prepare one text corpus for activation collection and perplexity evaluation
- [ ] Verify inference works end to end

### Milestone 2 - Rank Computation

- [ ] Register hooks for target layers
- [ ] Save normalized activations
- [ ] Compute activation covariance spectra
- [ ] Compute weight singular-value spectra
- [ ] Compute $r_u$ and $r_w$

### Milestone 3 - Visualization and Interpretation

- [ ] Plot layerwise effective rank
- [ ] Identify candidate bottleneck and low-rank layers
- [ ] Write a short interpretation memo

### Milestone 4 - Compression

- [ ] Run one low-rank compression experiment
- [ ] Run one quantization baseline
- [ ] Run one rank-guided compression policy

### Milestone 5 - Final Writeup

- [ ] Summarize findings in tables and figures
- [ ] Compare hypothesis with observed results
- [ ] Write limitations and future work

---

## 11. Author Contribution Plan

- **Peng:** implement the experiments; brainstorm improved diagnosis and compression methods; contribute to theoretical discussion.
- **Fergie:** assist with code and experiments; lead writing; contribute to the mathematical proof discussion.

---

## 12. AI Usage Statement

ChatGPT is used for polishing writing and brainstorming ideas.

---

## 13. References

1. ggml-org. *llama.cpp quantization readme*. GitHub repository documentation, accessed 2026-03-06.  
2. Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingcheng Zhang, and Song Han. *AWQ: Activation-aware weight quantization for LLM compression and acceleration*. arXiv preprint arXiv:2306.00978, 2023.  
3. Shibhansh Dohare, J. Francisco Hernandez-Garcia, Qingfeng Lan, et al. *Loss of plasticity in deep continual learning*. Nature, 632:768-774, 2024.  
4. Olivier Roy and Martin Vetterli. *The effective rank: A measure of effective dimensionality*. In *15th European Signal Processing Conference (EUSIPCO)*, 2007.

---

## 14. Optional Appendix: One-Sentence Project Summary

This project studies whether the effective rank of layer representations and weights can serve as a practical diagnostic for identifying bottlenecks and guiding compression decisions in a small language model.
