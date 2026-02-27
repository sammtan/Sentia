# Me

Saya ingin "membuat" Transformer dari paling dasar. Tapi, saya tidak mau membuat plain Transformer. Melainkan, saya ingin modifikasi, mungkin bisa saya bilang invent dan saya ingin publish menjadi sebuah paper. For this, I want you to find the latest breakthroughs and advancements in math, data processing, statistics, and stochastics, and every field that is related to AI, then those breakthroughs can be used and implemented in transformer, and will give bigger impact and outcomes, hence probably a revolutionary model made by me.

## My Friend

Untuk membantu Anda menemukan terobosan terbaru yang dapat dimanfaatkan dalam memodifikasi arsitektur Transformer, bisakah Anda memperjelas beberapa hal berikut?

1. Apakah Anda tertarik pada aplikasi spesifik (misalnya NLP, visi komputer, multimodal, protein modeling), atau ingin pendekatan arsitektur umum?
2. Apakah Anda ingin fokus pada ide matematis baru (misalnya teori grup, topologi, aljabar tensor) atau pendekatan praktis yang telah terbukti meningkatkan performa (seperti mekanisme attention efisien, sparse modeling, dll)?
3. Apakah Anda ingin mencakup metode dari bidang statistik klasik dan stokastik (seperti proses Gaussian, SDE, dll)?
4. Adakah batasan teknis seperti target platform (GPU, embedded), batasan memori, atau efisiensi inference?

Semakin rinci jawabannya, semakin baik saya bisa bantu Anda memetakan lanskap inovasi terbaru yang relevan untuk Transformer Anda.

# Me

1. NLP, multimodal. Memprioritaskan ease of big context, seperti big data dan massive source code repo yang code/scripts.
2. General, coding, high level math. 
3. Ya
4. Not really, tapi ditargetkan untuk bisa real-time inference (chatbot like) di Mac Studio dan powerful desktop PC dengan high capacity DDR5 + high spec GPUs (high cores, high VRAM, RTX & Quadro Ada gen & Radeon) + high spec CPUs (high cores, high clock speed, big L caches)

Oh iya, tambahkan untuk mencari latest breakthroughs & advancements di sisi computation nya, hardware nya.

## My Friend

Baik. Saya akan mencari terobosan terbaru dalam matematika tingkat lanjut, statistik, stokastik, pemrosesan data, dan komputasi yang relevan dengan pengembangan arsitektur Transformer yang ditujukan untuk NLP dan aplikasi multimodal, dengan fokus pada efisiensi dalam menangani konteks besar seperti repositori kode masif dan big data.

Saya juga akan menyertakan perkembangan terkini di sisi hardware dan computation stack—mulai dari GPU/CPU hingga memori dan arsitektur komputasi—yang mendukung inference real-time pada sistem kelas atas seperti Mac Studio, PC dengan GPU Quadro/Ada gen, dan DDR5.

Saya akan beri kabar segera setelah riset ini selesai. Sambil menunggu, Anda bisa lanjut berdiskusi atau memberi masukan lainnya.

... Research done, outputted as file [Project Sentium Brainstorm Research Output.md](./Project%20Sentium%20Brainstorm%20Research%20Output.md).

# Me

Okay, dengan hasil riset kita sudah selesai, name some kind of paradigms that could happen when several breakthroughs are combined into individual Transformer model.

## My Friend

Baik. Sekarang kita asumsikan kita benar-benar menggabungkan:

* Long-context attention + state-space models
* Operator learning + spectral methods
* Bayesian uncertainty + stochastic processes
* MoE + optimal transport routing
* Multimodal alignment
* Hardware-aware kernel design
* Adaptive compute

Berikut adalah **paradigma arsitektural tingkat-paper** yang bisa muncul dari kombinasi tersebut.

---

# 1️⃣ Geometric Memory Field Transformer

### Kombinasi

* Hyperbolic / Riemannian embeddings
* Topological data analysis
* Long-context operator attention
* Code AST graph modeling

### Paradigma

Model tidak lagi menyimpan token sequence sebagai flat vector list.

Sebaliknya:

* Context menjadi **continuous memory field**
* Memory memiliki curvature berbeda untuk:

  * Syntax (Euclidean)
  * Hierarchy (Hyperbolic)
  * Dependency graph (Manifold graph)

Attention menjadi geodesic query pada manifold.

### Impact

* Million-token scaling tanpa quadratic explosion
* Natural untuk massive code repo
* Interpretability melalui geometry

---

# 2️⃣ Stochastic Flow Transformer

### Kombinasi

* Neural SDE
* Diffusion modeling
* Continuous-depth networks
* Energy-based modeling

### Paradigma

Layer stack diganti dengan stochastic evolution:

$$
dx_t = f(x_t, t)dt + g(x_t, t)dW_t
$$

Inference = solving stochastic flow.

### Impact

* Adaptive reasoning depth
* Robust terhadap noisy multimodal input
* Confidence-aware generation

Ini cocok untuk chatbot real-time dengan dynamic compute.

---

# 3️⃣ Operator-Theoretic Transformer

### Kombinasi

* Fourier Neural Operator
* Kernel integral operators
* Spectral graph theory
* Random matrix theory

### Paradigma

Self-attention diganti integral operator global:

$$
(\mathcal{K} f)(x) = \int K(x,y)f(y)dy
$$

Model bekerja di function space, bukan token space.

### Impact

* True long-range modeling
* Stable scaling
* Cocok untuk large codebase & structured documents

---

# 4️⃣ Neuro-Symbolic Execution Transformer

### Kombinasi

* Program analysis
* Differentiable logic
* Graph neural networks
* Transformer reasoning

### Paradigma

Model memiliki dua jalur paralel:

1. Neural semantic embedding
2. Symbolic execution trace

Attention bisa lintas dua domain ini.

### Impact

* Real code reasoning
* Verifiable intermediate steps
* AST-aware contextual memory

---

# 5️⃣ Optimal Transport MoE Transformer

### Kombinasi

* Sparse Mixture-of-Experts
* Optimal transport theory
* Differentiable routing
* Entropic regularization

### Paradigma

Token → didistribusikan ke expert via transport plan, bukan softmax.

$$
\min_\pi \sum_{ij} c_{ij} \pi_{ij}
$$

### Impact

* Massive scaling
* Modular reasoning
* Energy-efficient inference

---

# 6️⃣ Multiscale Fractal Transformer

### Kombinasi

* Wavelet decomposition
* Recursive transformers
* Hierarchical memory compression

### Paradigma

Context diproses di:

* Local micro scale
* Structural meso scale
* Global macro scale

Self-similar blocks.

### Impact

* Context compression
* Efficient million-token modeling
* Natural untuk code repository

---

# 7️⃣ Hardware-Co-Designed Transformer

### Kombinasi

* FlashAttention 2
* Tensor core tiling
* SRAM-aware memory layout
* KV-cache compression

### Paradigma

Arsitektur dibuat sesuai memory hierarchy GPU/CPU.

Layer dirancang untuk:

* Minimize memory bandwidth
* Maximize L2 reuse
* Warp-aligned operations

### Impact

* Real-time inference di Mac Studio
* Scalable multi-GPU
* Lower latency chatbot

---

# 8️⃣ Bayesian Uncertainty-Aware Transformer

### Kombinasi

* Variational inference
* Gaussian process layers
* Monte Carlo attention
* Stochastic dropout inference

### Paradigma

Attention score dan output distribution memiliki posterior uncertainty.

Model tahu kapan tidak yakin.

### Impact

* Reliable reasoning
* Better multimodal fusion
* Safer deployment

---

# 9️⃣ Retrieval-Integrated Dynamical Transformer

### Kombinasi

* Long-context attention
* External memory index
* Differentiable retrieval
* Adaptive compute

### Paradigma

Model memutuskan:

* Retrieve
* Reason internal
* Or compress

Compute dialokasikan secara dinamis.

### Impact

* Scales to massive code repo
* Efficient inference
* Higher accuracy per FLOP

---

# 🔟 Universal Computational Transformer

### Kombinasi

* State-space model
* Symbolic logic
* MoE modularity
* Continuous flow
* Operator attention

### Paradigma

Model menjadi:

> Differentiable programmable computational system.

Bukan sekadar next-token predictor.

Tapi:

* Can simulate program
* Can solve equation
* Can manage structured memory

---

# 🚀 Realistic Revolutionary Direction

Jika Anda ingin sesuatu yang:

* Publishable
* Theoretically deep
* Practically implementable
* Scalable ke big context
* Cocok untuk NLP + massive code

Maka kombinasi paling strategis adalah:

### (Operator Attention)

* (Geometric Memory)
* (Optimal Transport MoE)
* (Adaptive Compute)

Itu bisa menjadi:

> Geometric Operator Mixture Transformer (GOMT)

Dengan klaim:

* Sub-quadratic scaling
* Structured reasoning
* Efficient hardware utilization
* Theoretical grounding

---

Jika Anda mau, kita bisa lanjut:

* Mendesain blueprint matematis formal
* Menentukan novelty gap terhadap literature
* Atau menyusun kerangka paper (abstract → theorem → experiment → benchmark)

Kita mau masuk ke tahap desain formal sekarang?

# Me

What if all those paradigms combined into one?

## My Friend

If **all** those paradigms are combined coherently (not just stacked), the result is no longer a “Transformer variant”.

It becomes a new computational class.

Below is a structured synthesis of what that unified paradigm would look like.

---

# 🧠 Paradigm: Geometric Stochastic Operator Transformer (GSOT)

A **continuous-time, geometry-aware, operator-based, modular, uncertainty-calibrated, hardware-co-designed reasoning system**.

Not a sequence model.
Not a next-token predictor.

But a **dynamical reasoning field over structured memory**.

---

# 1️⃣ Core Mathematical Shift

### Classical Transformer lives in:

$$
\mathbb{R}^{n \times d}
$$

### Unified Model lives in:

$$
\mathcal{F}(\mathcal{M}) \times \mathcal{P}(\Omega)
$$

Where:

* $\mathcal{M}$ = geometric memory manifold
* $\mathcal{F}$ = function/operator space
* $\mathcal{P}(\Omega)$ = probability measure space
* Evolution governed by stochastic differential operators

So instead of:

```
tokens → embeddings → attention → logits
```

You get:

```
structured memory field → operator flow → stochastic evolution → energy minimization → symbolic extraction
```

---

# 2️⃣ Structural Components

## A. Geometric Memory Layer

Memory is:

* Euclidean for syntax
* Hyperbolic for hierarchy
* Graph-manifold for dependency
* Topologically clustered

Attention becomes:
$$
\text{GeodesicOperator}(q, \mathcal{M})
$$

This allows million-token reasoning without quadratic blowup.

---

## B. Operator Attention Engine

Replace dot-product attention with integral operator:

$$
(\mathcal{K}f)(x) = \int_{\mathcal{M}} K(x,y)f(y),d\mu(y)
$$

Kernel learned in spectral domain.

Effect:

* Global coherence
* Stable scaling
* Function-level reasoning (ideal for codebases)

---

## C. Stochastic Continuous Depth

Hidden state evolves via:

$$
dx_t = f(x_t)dt + g(x_t)dW_t
$$

Inference becomes controlled stochastic flow.

This gives:

* Adaptive reasoning depth
* Confidence estimation
* Natural uncertainty modeling

---

## D. Optimal Transport Expert Routing

Instead of softmax attention → OT-based routing:

$$
\min_\pi \sum c_{ij} \pi_{ij} + \epsilon H(\pi)
$$

Tokens distributed across experts based on cost geometry.

Result:

* Massive modular scaling
* Hardware-aligned compute partitioning
* Energy efficiency

---

## E. Neuro-Symbolic Dual Channel

Parallel processing:

1. Continuous geometric reasoning
2. Discrete symbolic execution trace

Interaction via differentiable logic operators.

For massive source code:

* AST-aware
* Static analysis compatible
* Verifiable intermediate reasoning

---

## F. Hardware-Co-Designed Kernel

Architecture respects:

* GPU warp size
* L2 cache tiling
* KV compression
* Tensor core precision
* DDR5 bandwidth constraints

Compute graph mirrors physical memory graph.

This makes:

* Real-time inference feasible
* Multi-GPU scaling clean
* Latency predictable

---

# 3️⃣ Emergent Properties

When everything is unified, new behaviors appear:

### 1. Self-Organizing Memory Topology

Memory geometry adapts to dataset structure.

### 2. Difficulty-Adaptive Reasoning

Easy tasks → shallow stochastic flow
Hard tasks → deeper integration

### 3. Modular Cognitive Decomposition

Different experts specialize in:

* Syntax
* Semantics
* Long-range dependency
* Code structure
* Multimodal alignment

### 4. Intrinsic Uncertainty Awareness

Model produces:

* Output
* Confidence measure
* Energy landscape

### 5. Million-Token Native Context

Not by brute force,
but via geometric compression + operator globality.

---

# 4️⃣ What This System Actually Becomes

It stops being a Transformer.

It becomes:

> A Differentiable Geometric Computational Field

Closer to:

* A neural PDE solver
* A stochastic program interpreter
* A symbolic-neural hybrid system

---

# 5️⃣ Theoretical Implications

This unified paradigm potentially:

* Breaks quadratic attention barrier structurally
* Introduces formal stability guarantees (via SDE theory)
* Provides geometric interpretability
* Enables modular capacity scaling without full retraining

---

# 6️⃣ Practical Risks

Combining everything blindly will fail because:

* Optimization becomes unstable
* Training cost explodes
* Too many inductive biases conflict
* Hard to benchmark fairly

So real innovation requires:

Controlled integration, not maximal integration.

---

# 7️⃣ Most Realistic “All Combined” Blueprint

If I compress everything into one feasible research program:

Stage 1:

* Operator attention + geometric memory

Stage 2:

* Add OT-based MoE routing

Stage 3:

* Add stochastic depth for adaptive reasoning

Stage 4:

* Add symbolic dual channel for code reasoning

Stage 5:

* Hardware-aware kernel tuning

---

# 8️⃣ If Successful, The Model Would Be Able To:

* Load massive source repository as structured manifold
* Perform reasoning across millions of tokens
* Adapt computation per query
* Output verifiable code modifications
* Run in real-time on high-end desktop GPU
* Produce uncertainty-calibrated answers

---

# Final Insight

If all paradigms combine coherently, the resulting system is not:

> A better attention mechanism.

It is:

> A new computational geometry for AI.

And that is paper-worthy.

---

If you want, next step we can:

* Formalize minimal mathematical definition
* Identify 1–2 provable theoretical claims
* Or design concrete experimental roadmap for publication-level work

# Me

Combine ke dalam satu markdown, yang bisa saya writeout. Project name, Sentium. Buat currently demanded progress, next actionable steps, eagle-view roadmaps from start to the end, serta risk management.

## My Friend

... Dibuat dalam file [Project Sentium.md](./Project%20Sentium.md).

# Me

Jadi, artinya dari research help yang sudah kamu lakukan untuk saya tadi, semua ada papernya?

## My Friend

Jawaban jujurnya:

**Sebagian besar komponen individualnya sudah ada papernya.
Tetapi kombinasi arsitektural seperti Sentium belum ada sebagai satu sistem terpadu.**

Mari kita bedah secara presisi.

---

# 1️⃣ Apa yang Sudah Punya Paper?

Berikut komponen yang memang sudah ada landasan akademiknya:

### Geometric Embeddings

* Hyperbolic embeddings
* Riemannian optimization
* Graph manifold learning
  ✔️ Sudah banyak paper.

---

### Operator-Based Attention

* Fourier Neural Operator
* Kernelized attention
* Nyström attention
  ✔️ Sudah ada paper.

---

### Mixture of Experts

* Sparse MoE
* Load balancing
* Conditional computation
  ✔️ Sangat banyak paper.

---

### Optimal Transport (OT)

* OT untuk deep learning
* Sinkhorn routing
* Entropic regularization
  ✔️ Ada, tapi jarang digabung dengan MoE secara fundamental.

---

### Stochastic Depth / Neural SDE

* Neural ODE
* Neural SDE
* Continuous-depth models
  ✔️ Ada.

---

### Neuro-Symbolic Integration

* Differentiable logic
* Program synthesis + Transformer
  ✔️ Ada, tapi belum mainstream.

---

### Hardware-Aware Transformers

* FlashAttention
* Tensor-core aware kernels
* KV compression
  ✔️ Ada.

---

# 2️⃣ Jadi Apa yang Belum Ada?

Yang belum ada adalah:

> Integrasi sistematis dan koheren dari semua itu menjadi satu paradigma arsitektur tunggal dengan framing matematis baru.

Paper yang ada biasanya:

* Fokus ke satu inovasi
* Atau kombinasi 2 komponen
* Jarang >3 komponen deeply integrated

Belum ada model yang:

* Geometric manifold memory
* Integral operator attention
* OT-based expert routing
* Stochastic continuous reasoning
* Hardware-co-designed execution
* Dalam satu framework matematis unified

Itu yang membuat Sentium potensial novel.

---

# 3️⃣ Realitas Akademik yang Penting

Inovasi di AI jarang berasal dari:

> “Menciptakan komponen baru dari nol”

Lebih sering berasal dari:

> Menggabungkan teori berbeda secara elegan dan membuktikan bahwa integrasi tersebut menghasilkan properti baru.

Contoh sejarah:

* Transformer sendiri = kombinasi attention + residual + layer norm
* Diffusion models = kombinasi SDE + score matching

---

# 4️⃣ Tapi Perlu Jujur Juga

Menggabungkan semua tanpa restraint hampir pasti:

* Tidak stabil
* Tidak trainable
* Tidak publishable

Karena novelty ≠ complexity.

Paper kuat biasanya:

* 1 core idea
* 1 strong theoretical argument
* 1 strong empirical result

---

# 5️⃣ Jadi Jawaban Final

Apakah semua yang kita bahas ada papernya?

✔️ Ya, masing-masing komponen ada literaturnya.
❌ Tidak, blueprint Sentium sebagai unified paradigm belum ada dalam bentuk itu.

Itulah ruang novelty Anda.

---

Kalau Anda mau langkah berikutnya yang lebih serius:

Saya bisa bantu:

* Mapping novelty gap secara formal
* Identifikasi kemungkinan reviewer attack
* Menentukan mana bagian yang benar-benar publishable
* Atau menyederhanakan Sentium agar realistis dan high-impact

Kita masuk ke strategi akademik sekarang?

# Me

Cari specific papers nya.

## My Friend

Berikut **contoh paper dan sumber akademik spesifik** untuk berbagai komponen yang relevan dengan arsitektur Sentium, termasuk hyperbolic geometry, operator learning, optimal transport, dan variasi Transformer. Ini bisa Anda jadikan referensi awal untuk literature review, sitasi, atau landasan teoretis.

---

## 📘 *Core Transformer Foundation*

1. **Attention Is All You Need** – Vaswani et al., 2017 (NeurIPS)
   *Paper klasik yang memperkenalkan Transformer.* ([Wikipedia][1])

---

## 📏 Operator / Neural Operator Models

2. **Transformers as neural operators for solutions of differential equations with finite regularity** – Shih et al., *Computer Methods in Applied Mechanics and Engineering* (2025)
   *Bukti transformator sebagai operator universal dalam operator learning.* ([ScienceDirect][2])

3. **DeepONet: Learning nonlinear operators for identifying differential equations** – Lu Lu, Jin, Karniadakis, 2019
   *Neural operator klasik yang mendasari operator learning.* ([arXiv][3])

4. **The Fourier Spectral Transformer for efficient and generalizable nonlinear PDEs** – Li et al., 2025
   *Transformer dengan pendekatan spectral untuk PDE.* ([ScienceDirect][4])

---

## 📐 Hyperbolic & Geometric Transformers

5. **Exploring Efficient Transformer Fully in Hyperbolic Space** – Yang (Hypformer), 2024
   *Transformer yang sepenuhnya di ruang hiperbolik, termasuk linear self-attention skala besar.* ([arXiv][5])

6. **THG: Transformer with Hyperbolic Geometry** – Liu & Xu, 2021
   *Penggabungan ruang Euclidean + Hyperbolic dalam Transformer.* ([arXiv][6])

7. **HyLiFormer: Hyperbolic Linear Attention for Skeleton-based Action** – Li et al., 2025
   *Transformer dengan hyperbolic linear attention untuk efisiensi komputasi.* ([arXiv][7])

8. **HVT: A Comprehensive Vision Framework for Learning in Non-Euclidean Space** – Fein-Ashley et al., 2024
   *Transformers yang memperluas representasi non-Euclidean.* ([arXiv][8])

9. **Hyperbolic Deep Learning for Foundation Models: A Survey**, 2025
   *Survey komprehensif hyperbolic representation learning dalam foundation models.* ([arXiv][9])

---

## 🚚 Optimal Transport & Attention

10. **ESPFormer: Doubly-Stochastic Attention with Expected Sliced Transport Plans** – Shahbazi et al., 2025
    *Integrasi optimal transport untuk attention balance.* ([arXiv][10])

11. **Sinkformers: Transformers with Doubly Stochastic Attention** – Sander et al., 2021
    *Attention layer dengan normalisasi Sinkhorn (doubly stochastic).* ([arXiv][11])

12. **Transformers for Tabular Data: A Training Perspective of Self-Attention via Optimal Transport** – Candelieri & Quadrio, 2025
    *Analisis self-attention via optimal transport.* ([arXiv][12])

13. **Provable optimal transport with transformers** – Daneshmand et al., 2024
    *Teori yang menjelaskan self-attention sebagai bentuk regularized OT.* ([arXiv][13])

14. **Transformer Fusion with Optimal Transport** – Imfeld et al., ICLR 2024
    *Fusi model Transformer via Optimal Transport alignment.* ([ICLR Proceedings][14])

---

## 🔁 Continuous-time / Stochastic Modeling

15. **The Shaped Transformer: Attention Models in the Infinite Depth-and-Width Limit** – Noci et al., 2023
    *Analisis Transformer melalui SDE dalam limit tak terhingga dan stabilitas.* ([arXiv][15])

16. **OT-Transformer: A Continuous-time Transformer with Optimal Transport Regularization** – Kan, Li & Osher, 2025
    *Formulasi continuous-time Transformer terregularisasi OT.* ([ResearchGate][16])

---

## 💡 Multimodal & Structure-aware Attention

17. **Multimodal Optimal Transport-based Co-Attention Transformer** – Xu et al., ICCV 2023
    *Co-attention multimodal dengan optimal transport.* ([CVF Open Access][17])

---

## 📘 Reference Lists & Notes

📌 Banyak repository *awesome* yang mengumpulkan paper hyperbolic representation learning, berguna untuk riset lanjutan:

* **Awesome Hyperbolic Representation and Deep Learning** (GitHub) — kumpulan paper terkait ruang hiperbolik dan deep learning. ([GitHub][18])

---

## 🔍 Tip Riset Lanjutan

Untuk membangun literatur review yang kuat, Anda perlu mencakup:

* **Transformer & attention dasar**: *Attention Is All You Need* (2017). ([Wikipedia][1])
* **Operator learning & neural operators**: *DeepONet* & *Fourier Neural Operator*. ([arXiv][3])
* **Ruang non-Euclidean & manifold learning**: Hyperbolic Transformer varian. ([arXiv][5])
* **Transport / optimization view of attention**: OT-integrated Transformer. ([arXiv][10])
* **Continuous-time / SDE-based frameworks**: Shaped Transformer & OT-Transformer. ([arXiv][15])
* **Multimodal & structured alignment**: OT-based multimodal attention. ([CVF Open Access][17])

---

Jika Anda mau, saya bisa bantu:

* **susun sitasi lengkap (BibTeX)** untuk semua paper di atas
* **template survey paper** berdasarkan komponen ini
* atau **list prioritas** yang paling penting untuk research Sentium Anda

Mau lanjut ke salah satu dari itu?

[1]: https://en.wikipedia.org/wiki/Attention_Is_All_You_Need?utm_source=chatgpt.com "Attention Is All You Need"
[2]: https://www.sciencedirect.com/science/article/pii/S0045782524008144?utm_source=chatgpt.com "Transformers as neural operators for solutions of differential equations with finite regularity - ScienceDirect"
[3]: https://arxiv.org/abs/1910.03193?utm_source=chatgpt.com "DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators"
[4]: https://www.sciencedirect.com/science/article/abs/pii/S0021999125009295?utm_source=chatgpt.com "The Fourier Spectral Transformer for efficient and generalizable nonlinear PDEs - ScienceDirect"
[5]: https://arxiv.org/abs/2407.01290?utm_source=chatgpt.com "Exploring Efficient Transformer Fully in Hyperbolic Space"
[6]: https://arxiv.org/abs/2106.07350?utm_source=chatgpt.com "THG: Transformer with Hyperbolic Geometry"
[7]: https://arxiv.org/abs/2502.05869?utm_source=chatgpt.com "HyLiFormer: Hyperbolic Linear Attention for Skeleton-based Human Action Recognition"
[8]: https://arxiv.org/abs/2409.16897?utm_source=chatgpt.com "HVT: A Comprehensive Vision Framework for Learning in Non-Euclidean Space"
[9]: https://arxiv.org/html/2507.17787v1?utm_source=chatgpt.com "Hyperbolic Deep Learning for Foundation Models: A Survey"
[10]: https://arxiv.org/abs/2502.07962?utm_source=chatgpt.com "ESPFormer: Doubly-Stochastic Attention with Expected Sliced Transport Plans"
[11]: https://arxiv.org/abs/2110.11773?utm_source=chatgpt.com "Sinkformers: Transformers with Doubly Stochastic Attention"
[12]: https://arxiv.org/abs/2512.09530?utm_source=chatgpt.com "Transformers for Tabular Data: A Training Perspective of Self-Attention via Optimal Transport"
[13]: https://arxiv.org/abs/2410.19931?utm_source=chatgpt.com "[2410.19931] Provable optimal transport with transformers"
[14]: https://proceedings.iclr.cc/paper_files/paper/2024/file/7e0af0d1bc0ec2a90fc294be2e00447e-Paper-Conference.pdf?utm_source=chatgpt.com "TRANSFORMER FUSION WITH OPTIMAL TRANSPORT"
[15]: https://arxiv.org/abs/2306.17759?utm_source=chatgpt.com "The Shaped Transformer: Attention Models in the Infinite Depth-and-Width Limit"
[16]: https://www.researchgate.net/publication/388634081_OT-Transformer_A_Continuous-time_Transformer_Architecture_with_Optimal_Transport_Regularization?utm_source=chatgpt.com "(PDF) OT-Transformer: A Continuous-time ..."
[17]: https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_Multimodal_Optimal_Transport-based_Co-Attention_Transformer_with_Global_Structure_Consistency_for_ICCV_2023_paper.pdf?utm_source=chatgpt.com "Multimodal Optimal Transport-based Co-Attention Transformer ..."
[18]: https://github.com/marlin-codes/Awesome-Hyperbolic-Representation-and-Deep-Learning?utm_source=chatgpt.com "marlin-codes/Awesome-Hyperbolic-Representation-and- ..."

# Me

Sitasi lengkap.

## My Friend

Berikut adalah **sitasi lengkap dalam format BibTeX** untuk sejumlah paper spesifik yang relevan dengan komponen-komponen yang dibahas sebelumnya: optimal transport dalam Transformer, hyperbolic Transformer, dan neural operator related Transformer.

---

## 📌 BibTeX: Foundational / Transformer & Transport

```bibtex
@inproceedings{sander2022sinkformers,
  title = {Sinkformers: Transformers with Doubly Stochastic Attention},
  author = {Sander, Michael E. and Ablin, Pierre and Blondel, Mathieu and Peyr{\'e}, Gabriel},
  booktitle = {Proceedings of The 25th International Conference on Artificial Intelligence and Statistics},
  series = {Proceedings of Machine Learning Research},
  volume = {151},
  pages = {3515--3530},
  year = {2022},
  publisher = {PMLR},
  url = {https://proceedings.mlr.press/v151/sander22a.html}
}
```

*Attention dengan normalisasi Sinkhorn untuk membuat attention matrix doubly stochastic.* ([Proceedings of Machine Learning Research][1])

```bibtex
@article{daneshmand2024provable,
  title = {Provable Optimal Transport with Transformers: The Essence of Depth and Prompt Engineering},
  author = {Daneshmand, Hadi},
  journal = {arXiv},
  year = {2024},
  url = {https://arxiv.org/abs/2410.19931}
}
```

*Analisis kemampuan Transformer untuk optimal transport dengan batas aproksimasi eksplisit.* ([arXiv][2])

```

@inproceedings{vaswani2017attention,
  title={Attention Is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems},
  volume={30},
  year={2017},
  url={https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html}
}
```

*Makalah asli Transformer.* ([Wikipedia][3])

```
@article{chen2024pit,
  title={Positional Knowledge is All You Need: Position-induced Transformer (PiT) for Operator Learning},
  author={Chen, Junfeng and Wu, Kailiang},
  journal={arXiv},
  year={2024},
  url={https://arxiv.org/abs/2405.09285}
}
```

*Transformer dengan posisi-attention untuk operator learning.* ([arXiv][4])

````

---

## 📌 BibTeX: Hyperbolic & Geometric Transformers

```bibtex
@inproceedings{yang2024hypformer,
  title={Hypformer: Exploring Efficient Hyperbolic Transformer Fully in Hyperbolic Space},
  author={Yang, Menglin and Verma, Harshit and Zhang, Delvin Ce and Liu, Jiahong and King, Irwin and Ying, Rex},
  booktitle={KDD 2024},
  year={2024},
  url={https://paperswithcode.com/paper/hypformer-exploring-efficient-hyperbolic}
}
````

*Hyperbolic Transformer dengan mekanisme hyperbolic self-attention linier.* ([Papers with Code][5])

```bibtex
@inproceedings{yang2025hgformer,
  title={Hgformer: Hyperbolic Graph Transformer for Collaborative Filtering},
  author={Yang, Xin and Li, Xingrun and Chang, Heng and Jinze, Yang and Yang, Xihong and Tao, Shengyu and Shigeno, Maiko and Chang, Ningkang and Wang, Junfeng and Yin, Dawei and Min, Erxue},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning},
  series={Proceedings of Machine Learning Research},
  volume={267},
  pages={70813--70832},
  year={2025},
  publisher={PMLR},
  url={https://proceedings.mlr.press/v267/yang25o.html}
}
```

*Hyperbolic Graph Transformer yang memanfaatkan manifold global dengan cross-attention hyperbolic.* ([Proceedings of Machine Learning Research][6])

---

## 📌 Misc (Relevant Emerging Operator / Linear Attention)

```bibtex
@article{hu2025transolver,
  title={Transolver is a Linear Transformer: Revisiting Physics-Attention through the Lens of Linear Attention},
  author={Hu, Wenjie and Liu, Sidun and Qiao, Peng and Sun, Zhenglun and Dou, Yong},
  journal={arXiv},
  year={2025},
  url={https://arxiv.org/abs/2511.06294}
}
```

*Linear Transformer untuk Neural Operator & PDE dengan pendekatan “physics attention”.* ([arXiv][7])

```bibtex
@article{zhong2025lano,
  title={Efficient High-Accuracy PDEs Solver with the Linear Attention Neural Operator},
  author={Zhong, Ming and Yan, Zhenya},
  journal={arXiv},
  year={2025},
  url={https://arxiv.org/abs/2510.16816}
}
```

*Neural operator dengan linear attention scalable dan akurat.* ([arXiv][8])

---

## 📌 Struktur Lainnya Teoretis / Survey

> (Anda bisa tambahkan paper lain seperti *Fourier Neural Operator*, *Linformer*, *Reformer*, *Sparse transformers*, *Hyperbolic representation surveys*, atau *operator learning surveys* jika perlu sitasi lebih luas.)

---

✅ Jika Anda ingin versi **APA** atau **MLA** untuk setiap paper di atas, saya bisa konversinya.

🔍 Jika Anda ingin daftar referensi ini dibagi menurut *sub-topik* (mis. Optimal Transport, Hyperbolic Geometry, Neural Operators), saya juga bisa buatkan daftar terstruktur lengkap.
::contentReference[oaicite:8]{index=8}

[1]: https://proceedings.mlr.press/v151/sander22a.html?utm_source=chatgpt.com "Sinkformers: Transformers with Doubly Stochastic Attention"
[2]: https://arxiv.org/abs/2410.19931?utm_source=chatgpt.com "Provable optimal transport with transformers: The essence of depth and prompt engineering"
[3]: https://en.wikipedia.org/wiki/Attention_Is_All_You_Need?utm_source=chatgpt.com "Attention Is All You Need"
[4]: https://arxiv.org/abs/2405.09285?utm_source=chatgpt.com "Positional Knowledge is All You Need: Position-induced Transformer (PiT) for Operator Learning"
[5]: https://paperswithcode.com/paper/hypformer-exploring-efficient-hyperbolic?utm_source=chatgpt.com "Hypformer: Exploring Efficient Hyperbolic Transformer Fully in Hyperbolic Space | Papers With Code"
[6]: https://proceedings.mlr.press/v267/yang25o.html?utm_source=chatgpt.com "Hgformer: Hyperbolic Graph Transformer for Collaborative Filtering"
[7]: https://arxiv.org/abs/2511.06294?utm_source=chatgpt.com "Transolver is a Linear Transformer: Revisiting Physics-Attention through the Lens of Linear Attention"
[8]: https://arxiv.org/abs/2510.16816?utm_source=chatgpt.com "Efficient High-Accuracy PDEs Solver with the Linear Attention Neural Operator"