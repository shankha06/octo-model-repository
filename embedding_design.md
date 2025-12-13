# **Chroma-MoE-3B: Architectural Blueprint for a High-Performance, Domain-Specific Embedding Model**

## **Executive Summary**

This report presents a comprehensive technical blueprint for the design, training, and deployment of **Chroma-MoE-3B**, a specialized text embedding model engineered for the finance and retail sectors. Addressing the request for a high-performance model built from scratch, this architecture leverages the most significant advancements in Transformer efficiency from the 2024-2025 research landscape. Specifically, it integrates **DeepSeek-V2’s Multi-Head Latent Attention (MLA)** and **Fine-Grained Mixture-of-Experts (MoE)** with **NV-Embed’s Latent Attention Pooling** to achieve state-of-the-art (SOTA) performance within a strictly constrained parameter budget of approximately 2.8 billion total parameters.

The prevailing trend in Large Language Models (LLMs) has been to scale parameters to hundreds of billions to achieve general-purpose reasoning. However, for specialized embedding tasks—such as semantic search in e-commerce catalogs or retrieval-augmented generation (RAG) over financial 10-K filings—massive dense models often suffer from high inference latency and inefficient memory usage. Chroma-MoE-3B challenges this paradigm by adopting a **Sparse Fine-Grained MoE** architecture. This design maintains the inference speed of a sub-1-billion parameter model while accessing a total representational capacity of nearly 3 billion parameters, ensuring deep domain specialization without the computational penalty of dense scaling.

Furthermore, the model is designed with an 8,192-token context window to accommodate long-form financial reports and extensive product histories. To support this length efficiently, the architecture replaces standard Grouped Query Attention (GQA) with Multi-Head Latent Attention (MLA), reducing Key-Value (KV) cache memory usage by over 90% and enabling large-batch contrastive training—a critical factor for embedding quality. The training recipe utilizes a rigorous two-stage approach: domain-adaptive pre-training with a bidirectional attention mask, followed by contrastive instruction fine-tuning using In-Batch Negatives and Hard Negative Mining on a curated mixture of Hugging Face datasets including FinMTEB, Amazon ESCI, and Ecom-niverse.

## ---

**1\. Introduction: The Imperative for Domain-Specific Sparse Modeling**

The landscape of Natural Language Processing (NLP) has shifted dramatically from dense, general-purpose encoders like BERT and RoBERTa to massive decoder-only generative models. While generative models excel at text production, their utility as embedding models—converting text into fixed-size vector representations for similarity search—has historically lagged behind dedicated encoders. However, recent advancements in 2024 and 2025 have closed this gap, demonstrating that decoder-only architectures, when trained with bidirectional attention and contrastive objectives, can surpass traditional encoders.

### **1.1 The Domain Gap in Finance and Retail**

General-purpose embedding models (e.g., OpenAI's text-embedding-3, BGE-M3) are trained on broad web corpora. While robust, they often fail to capture the nuanced semantic shifts present in specialized domains:

* **Finance:** Terms like "bullish," "short," or "leverage" have distinct, often mathematical implications that differ from their colloquial usage. Financial retrieval tasks require understanding temporal causality (e.g., "Q3 earnings impact on Q4 guidance") and regulatory syntax (e.g., SEC filings).1  
* **Retail:** E-commerce queries are often ungrammatical keyword strings (e.g., "mens running shoes red size 10 cheap"). Semantic similarity in retail must account for product taxonomy (e.g., distinguishing between "accessory for X" and "X itself") and substitute/complement relationships.2

A model trained specifically on these domains can achieve higher precision with fewer parameters by dedicating its capacity to relevant vocabulary and syntactic structures rather than general world knowledge.

### **1.2 The Efficiency Dilemma and the MoE Solution**

Training a high-performance model "from scratch" presents a tradeoff between capacity (model size) and inference latency. A 3B parameter dense model requires significant compute for every token generated or embedded.

* **The MoE Advantage:** Mixture-of-Experts architectures decouple model size from compute cost. By routing tokens to only a small subset of "expert" feed-forward networks (FFNs), an MoE model can possess the total parameter count of a large model (e.g., 3B) while only activating a fraction (e.g., 0.8B) for any given token.3  
* **Fine-Grained Granularity:** Traditional MoE (like Mixtral 8x7B) uses a few large experts. This report advocates for **Fine-Grained MoE** (as seen in DeepSeek-V2 and Qwen2-MoE), which utilizes many small experts. This granularity allows for finer semantic specialization—one expert might specialize in "retail pricing logic," another in "financial risk disclaimers," and a third in "general English syntax".5

### **1.3 Convergence of SOTA Architectures**

The proposed **Chroma-MoE-3B** integrates three distinct SOTA innovations:

1. **DeepSeek-V2's MLA:** Solves the KV cache bottleneck for long contexts (8k), enabling efficient handling of financial reports.7  
2. **Fine-Grained MoE:** Maximizes parameter efficiency and expert specialization.5  
3. **NV-Embed's Latent Pooling:** Replaces simple mean pooling with a learned attention mechanism for superior sequence representation.9

This synthesis creates a model that is uniquely positioned to dominate retrieval benchmarks in the targeted sectors.

## ---

**2\. Architectural Design Specifications**

The **Chroma-MoE-3B** is a decoder-only Transformer model modified for bidirectional attention and sparse computation. The total parameter count is targeted at \~2.8 billion, with active parameters per token restricted to \~0.9 billion to ensure low-latency inference on standard hardware (e.g., single NVIDIA A10G or RTX 4090).

### **2.1 The Transformer Backbone**

Unlike BERT (Encoder-only) or GPT (Causal Decoder), this architecture uses a **Bidirectional Decoder**.

* **Structure:** It retains the decoder's efficient block layout (Self-Attention $\\rightarrow$ FFN) but removes the causal triangular mask during the attention pass. This allows every token to attend to every other token, crucial for capturing the full context required for high-quality embeddings.10  
* **SwiGLU Activation:** All Feed-Forward Networks (FFNs) and Experts utilize the SwiGLU activation function. SwiGLU ($\\text{Swish}(xW) \\cdot (xV)$) has consistently outperformed GeLU in compute-equivalent setups by allowing more expressive linear gating.12  
* **RMSNorm:** Pre-normalization is applied using Root Mean Square Layer Normalization (RMSNorm) for improved training stability and slight computational efficiency gains over LayerNorm.12

**Table 1: Global Model Hyperparameters**

| Hyperparameter | Value | Rationale |
| :---- | :---- | :---- |
| **Total Parameters** | \~2.8 Billion | Matches user requirement; fits in \~6GB VRAM (FP16). |
| **Active Parameters** | \~0.9 Billion | Ensures high throughput; comparable to \~1B dense models. |
| **Context Window** | 8,192 (8k) | Essential for long financial documents and product histories. |
| **Hidden Dimension ($d\_{model}$)** | 2,048 | Standard for this scale; balances capacity and width. |
| **Layers ($L$)** | 28 | Sufficient depth for reasoning without excessive latency. |
| **Vocabulary Size** | 32,000 | Compact, domain-optimized WordPiece/BPE (see Section 3). |

### **2.2 Mechanism 1: Multi-Head Latent Attention (MLA)**

The user requires support for GQA, RoPE, and Latent Attention. While Grouped Query Attention (GQA) is standard in models like Llama-3, **Multi-Head Latent Attention (MLA)** represents the next evolution, specifically targeting the memory inefficiency of the KV cache in long-context scenarios.7

#### **2.2.1 The KV Cache Bottleneck**

In standard Multi-Head Attention (MHA), the model must store Key ($K$) and Value ($V$) matrices for every token in the context. For an 8k context window and batch size of 64, this cache grows enormously, limiting the maximum batch size for training and inference. GQA mitigates this by sharing KV heads, but MLA solves it through **Low-Rank Compression**.

#### **2.2.2 MLA Implementation Details**

MLA projects the input hidden state into a low-dimensional latent vector $c\_{KV}$, which is significantly smaller than the full $K$ and $V$ matrices.

* **Down-Projection:** The input $h\_t$ is projected to a compressed latent vector $c\_{KV} \\in \\mathbb{R}^{d\_c}$ (where $d\_c \\ll d\_h \\times n\_h$).  
* Up-Projection: During the attention operation, $c\_{KV}$ is up-projected to generate the keys and values on the fly.

  $$c\_{KV} \= W\_{DKV} h\_t$$  
  $$K \= W\_{UK} c\_{KV}, \\quad V \= W\_{UV} c\_{KV}$$  
* **Efficiency:** We only cache the compressed vector $c\_{KV}$. For Chroma-MoE-3B, this reduces KV cache memory usage by approximately **93%** compared to standard MHA 14, enabling massive batch sizes during contrastive training (Phase 2).

#### **2.2.3 Decoupled RoPE**

The user requires native **Rotary Positional Embeddings (RoPE)**. In MLA, applying RoPE directly to the compressed latent vector is mathematically unsound because RoPE is rotation-sensitive and sensitive to the vector's specific dimensions.

* **Strategy:** We adopt the **Decoupled RoPE** strategy utilized in DeepSeek-V2.3  
  1. A separate, small "peeking" head projects the input $h\_t$ to a positional vector $k^R\_t$.  
  2. RoPE is applied *only* to this positional vector $k^R\_t$.  
  3. The attention score computation combines the content-based attention (from the compressed latent vector) and the position-based attention (from the RoPE vector).

     $$q^T k \= (q^C)^T k^C \+ (q^R)^T k^R$$

     This ensures precise positional awareness (crucial for "Q3 vs Q4" in finance) without inflating the cache size.

### **2.3 Mechanism 2: Fine-Grained Mixture-of-Experts (MoE)**

To achieve the target parameter count of 2B-3B total with high efficiency, we employ a **Fine-Grained MoE** architecture inspired by DeepSeekMoE 5 and Qwen2-MoE.15

#### **2.3.1 Expert Granularity**

Standard MoE (e.g., Switch Transformer) uses large, coarse-grained experts. For a domain-specific model, we require experts that can capture specialized micro-features (e.g., "detecting currency symbols" vs. "analyzing sentiment").

* **Shared Experts:** We designate **2 experts** as "Shared Experts" that are always activated for every token. These experts capture common linguistic features (grammar, stopwords) and ensure training stability.5  
* **Routed Experts:** We deploy **64 routed experts**.  
* **Active Experts:** For each token, the router selects the **Top-6** experts from the pool of 64\.  
* **Configuration:**  
  * Intermediate Size of Shared Experts: $4 \\times d\_{model}$ (Standard FFN size).  
  * Intermediate Size of Routed Experts: Scaled down (e.g., $1 \\times d\_{model}$) to keep total parameters in check.

#### **2.3.2 Routing Mechanism**

We use a learned router (gating network) $G(x)$ that computes the probability of assigning token $x$ to expert $E\_i$.

$$G(x) \= \\text{Softmax}(x \\cdot W\_g)$$

To prevent expert collapse (where one expert learns everything and others die), we incorporate an Auxiliary Load Balancing Loss during training, penalizing the model if the distribution of expert selection deviates from uniformity.14

### **2.4 Mechanism 3: Latent Attention Pooling**

The final requirement is the generation of a single high-quality vector representation for the input sequence. Standard approaches use the embedding of the \`\` token or the mean of all token embeddings. However, these methods often dilute the signal, especially in long financial documents where the "alpha" (crucial information) is sparse.

We implement **Latent Attention Pooling**, a technique popularized by NV-Embed.9

* Mechanism: We introduce a trainable "latent query" array $Q\_{latent} \\in \\mathbb{R}^{1 \\times d\_{model}}$ (or multiple latents). This query attends to the final hidden states $H\_{out}$ of the transformer using a standard Cross-Attention mechanism.

  $$\\text{Pooling}(H\_{out}) \= \\text{Attention}(Q\_{latent}, H\_{out}, H\_{out})$$  
* **Benefit:** The model learns to "pay attention" to the most semantically significant parts of the sequence (e.g., the product model number or the net income figure) and ignore irrelevant boilerplate text.  
* **Bi-directional Training Integration:** As established in NV-Embed research, this pooling layer is trained jointly with the backbone during the contrastive phase, allowing the latent query to learn domain-specific importance weights.10

## ---

**3\. Data Curation and Pipeline**

The quality of a domain-specific model is deterministic based on its training data. We define a "Ecom-Fin" corpus strategy, leveraging high-quality datasets available on Hugging Face.

### **3.1 The Financial Corpus**

The financial domain demands high precision in numerical reasoning and temporal logic.

| Dataset | Hugging Face ID | Usage Phase | Description & Strategy |
| :---- | :---- | :---- | :---- |
| **FinMTEB** | TheFinAI/finmteb 1 | Fine-tuning | The cornerstone benchmark/training set. Contains 64 datasets covering STS, Retrieval, and Classification. **Action:** Use the training splits of all sub-tasks (e.g., FinSTS, FinRetrieval). |
| **EDGAR Corpus** | financial-reports-sec 16 | Pre-training | Full text of 10-K and 10-Q filings. **Action:** Filter for "Risk Factors" and "MD\&A" sections which contain the most semantic density. Use for MLM pre-training. |
| **ConvFinQA** | MehdiHosseiniMoghadam/ConvFinQA 17 | Fine-tuning | Q\&A pairs over financial reports. **Action:** Convert to (Query, Positive Document) pairs to teach numerical context matching. |
| **FNSPID** | Brianferrell787/financial-news-multisource 18 | Pre-training | Aggregated financial news. **Action:** Use for learning market sentiment and causality. |

### **3.2 The Retail & E-Commerce Corpus**

Retail embeddings must handle the "vocabulary mismatch" problem (users search "cheap kicks", products are labeled "discounted athletic footwear").

| Dataset | Hugging Face ID | Usage Phase | Description & Strategy |
| :---- | :---- | :---- | :---- |
| **Amazon ESCI** | tasksource/esci 2 | Fine-tuning | **Gold Standard.** Contains query-product pairs labeled "Exact", "Substitute", "Complement", "Irrelevant". **Action:** Use "Exact" as positives. Use "Substitute" and "Complement" as **Hard Negatives** (crucial for precision). |
| **Ecom-niverse** | thebajajra/Ecom-niverse 19 | Pre-training | 350B+ tokens of web-crawled e-commerce text. **Action:** Use this to build the domain-specific vocabulary (brand names, SKUs, attributes). |
| **Amazon Reviews** | McAuley-Lab/Amazon-Reviews-2023 20 | Pre-training | **Action:** Use review text to model informal user language and sentiment. |

### **3.3 General Domain & Regularization**

To prevent "catastrophic forgetting" of general English (which is the substrate of all domain text), we mix in general data.

* **FineWeb-Edu:** Use a 15% mixture of HuggingFaceFW/fineweb-edu during pre-training to maintain grammatical competence.  
* **MTEB (General):** Select MS-MARCO and NQ (Natural Questions) for fine-tuning to ensure the model understands general question-answering structures.21

### **3.4 Data Processing and Tokenization**

* **Vocabulary Construction:** Train a customized BPE tokenizer (vocab size 32,000) on a 50/50 mix of Ecom-niverse and EDGAR data. This ensures that domain-specific terms like "EBITDA", "RoI", "HDMI", and "polyester" are single tokens rather than fragmented sub-words, improving efficiency and semantic representation.  
* **Synthetic Data Generation:** Utilize a strong LLM (e.g., Llama-3-70B or Mixtral) to generate synthetic queries for documents in the pre-training corpus. For example, feed a paragraph from a 10-K filing and ask the LLM to "Write a search query a financial analyst would type to find this paragraph." This creates high-quality (Query, Passage) pairs for contrastive training.10

## ---

**4\. Training Strategy and Optimization**

Building Chroma-MoE-3B from scratch involves a rigorous two-stage pipeline. The model is first taught to *understand* the language (Pre-training) and then taught to *embed* the language (Contrastive Tuning).

### **4.1 Phase 1: Domain-Adaptive MoE Pre-training**

Since we are building from scratch, we initialize the weights randomly (or using specific variance scaling for SwiGLU/MoE).

* **Objective:** **Bidirectional Auto-Regressive Modeling (BAR)**. Unlike standard GPT training (predict next token), we use a bidirectional mask where tokens can attend to both left and right contexts. The loss is calculated on masked spans (similar to T5 or UL2) or via standard MLM objective but applied to the decoder architecture. This aligns with NV-Embed's findings that removing causal masking boosts retrieval performance.10  
* **MoE Upcycling (Optional but Recommended):** To stabilize training, start with a dense model for the first 10% of steps. Then, "upcycle" by replicating the FFN weights to initialize the experts and introduce the router. This prevents early training instability often seen in MoEs.15  
* **Optimization:**  
  * **Optimizer:** AdamW ($\\beta\_1=0.9, \\beta\_2=0.95, \\epsilon=1e-8$).  
  * **Learning Rate:** Peak at $3 \\times 10^{-4}$, cosine decay to $3 \\times 10^{-5}$.  
  * **Batch Size:** Global batch size of 2M tokens (achieved via Gradient Accumulation).  
  * **Load Balancing:** Apply auxiliary loss ($\\alpha=0.01$) to ensure entropy in expert routing.

### **4.2 Phase 2: Contrastive Instruction Tuning**

This phase transforms the linguistic model into a semantic embedding model.

* Loss Function: InfoNCE Loss with In-Batch Negatives.

  $$\\mathcal{L} \= \- \\log \\frac{e^{\\text{sim}(q, d^+) / \\tau}}{e^{\\text{sim}(q, d^+) / \\tau} \+ \\sum\_{j=1}^{K} e^{\\text{sim}(q, d^-\_j) / \\tau}}$$

  where $d^+$ is the positive document and $d^-\_j$ are negatives.  
* **In-Batch Negatives:** We use the other samples in the batch as negatives. For a batch size of $B$, each query has 1 positive and $B-1$ negatives.  
* **Hard Negative Mining:** We explicitly include "Hard Negatives" mined from BM25 (lexical overlap but semantic mismatch) and the "Substitute" labels from Amazon ESCI.  
* **GradCache for Large Batches:** InfoNCE benefits logarithmically from batch size. To fit a target batch size of **16,384** (or larger) into GPU memory, we implement **GradCache**.22 This technique splits the batch into micro-batches for the forward pass, caches the gradients, and accumulates them, mathematically simulating a massive global batch without the memory explosion.  
* **Instruction Format:** Inputs are formatted with task instructions to guide the latent attention pooling.  
  * *Input:* Instruction: Retrieve the relevant financial disclosure. Query: What are the liquidity risks associated with the merger?

### **4.3 Hyperparameters and Dynamics**

* **Training Duration:**  
  * Phase 1: \~1.5 Trillion tokens (approx. 2 weeks on 64x H100s).  
  * Phase 2: \~5 Billion tokens (high-quality pairs).  
* **Temperature ($\\tau$):** Start at 0.05, learnable during training.  
* **Max Sequence Length:** 8,192 tokens. (Short queries are padded or packed; long documents utilize the full window).

### **4.4 Hardware Efficiency: Expert Parallelism**

To train the MoE efficiently, we utilize **Expert Parallelism (EP)**.

* The transformer backbone and shared experts are replicated across all GPUs (Data Parallelism).  
* The 64 routed experts are distributed across GPUs. If we have 8 GPUs, each GPU holds 8 experts.  
* An "All-to-All" communication primitive is used to dispatch tokens to the correct GPU for expert processing and gather the results. DeepSeek's "DeepEP" library or Megatron-Core can be used to optimize this communication.23

## ---

**5\. Evaluation Framework**

The model's success will be validated against specific benchmarks that reflect its dual nature (Finance/Retail).

* **Primary Metric:** **NDCG@10** (Normalized Discounted Cumulative Gain) for retrieval tasks.  
* **Secondary Metric:** **Spearman Correlation** for Semantic Textual Similarity (STS) tasks.  
* **Benchmarks:**  
  1. **FinMTEB Leaderboard:** Specifically the Retrieval and STS subsets. Target: Beat bge-m3 and openai-text-embedding-3-large.  
  2. **Amazon ESCI:** Evaluate ranking performance on the "Exact" vs. "Substitute" distinction.  
  3. **MTEB (English):** Standard verification to ensuring general English capabilities remain intact.

## ---

**6\. Conclusion**

**Chroma-MoE-3B** represents a strategic convergence of efficient architecture and domain specialization. By eschewing the "brute force" scaling of dense models, this blueprint utilizes **Fine-Grained MoE** and **MLA** to create a model that is:

1. **Deep:** Capable of understanding complex financial reasoning and retail taxonomy.  
2. **Fast:** Inference latency comparable to a \<1B model.  
3. **Long-Context:** Natively handling 8k token documents via compressed attention.  
4. **Retrieval-Optimized:** Leveraging Latent Attention Pooling and massive-batch contrastive learning.

This report provides the roadmap—from specific Hugging Face datasets like FinMTEB and Amazon ESCI to the implementation of Decoupled RoPE and GradCache—required to build this asset from the ground up. Implementing this design will yield a proprietary embedding model that significantly outperforms off-the-shelf generalist APIs for high-value finance and retail applications.

### **Strategic Recommendations for Implementation**

* **Start with the Tokenizer:** Do not overlook the vocabulary. A standard tokenizer will fragment "EBITDA" into 3 tokens; a custom one makes it 1\. This efficiency compounds over billions of tokens.  
* **Invest in Hard Negatives:** The difference between a "good" and "SOTA" embedding model is often the quality of the negative samples in Phase 2\. Dedicate compute to mining these from the ESCI and FinMTEB datasets.  
* **Monitor Expert Load:** Use strong auxiliary losses. If 2 experts handle 90% of the traffic, the MoE advantage is lost. Visualize expert routing during Phase 1 to ensure healthy distribution.

#### **Works cited**

1. Paper page \- FinMTEB: Finance Massive Text Embedding Benchmark \- Hugging Face, accessed December 13, 2025, [https://huggingface.co/papers/2502.10990](https://huggingface.co/papers/2502.10990)  
2. tasksource/esci · Datasets at Hugging Face, accessed December 13, 2025, [https://huggingface.co/datasets/tasksource/esci](https://huggingface.co/datasets/tasksource/esci)  
3. DeepSeek Technical Analysis — (2)Multi-Head Latent Attention | by Jinpeng Zhang, accessed December 13, 2025, [https://dataturbo.medium.com/deepseek-technical-analysis-2-mla-74bdb87d4ad2](https://dataturbo.medium.com/deepseek-technical-analysis-2-mla-74bdb87d4ad2)  
4. DeepSeek MoE & V2 \- Creative Strategies, accessed December 13, 2025, [https://creativestrategies.com/deepseek-moe-v2/](https://creativestrategies.com/deepseek-moe-v2/)  
5. DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models \- arXiv, accessed December 13, 2025, [https://arxiv.org/html/2401.06066v1](https://arxiv.org/html/2401.06066v1)  
6. Qwen2 Technical Report \- arXiv, accessed December 13, 2025, [https://arxiv.org/html/2407.10671v1](https://arxiv.org/html/2407.10671v1)  
7. Attention Evolved: How Multi-Head Latent Attention Works | by Karl Weinmeister | Google Cloud \- Community | Medium, accessed December 13, 2025, [https://medium.com/google-cloud/attention-evolved-how-multi-head-latent-attention-works-427a922dd6a1](https://medium.com/google-cloud/attention-evolved-how-multi-head-latent-attention-works-427a922dd6a1)  
8. Implementing Multi-Head Latent Attention from Scratch in Python | by Void | Medium, accessed December 13, 2025, [https://medium.com/@atulit23/implementing-multi-head-latent-attention-from-scratch-in-python-1e14d03fbc91](https://medium.com/@atulit23/implementing-multi-head-latent-attention-from-scratch-in-python-1e14d03fbc91)  
9. NV Embed V1 · Models \- Dataloop, accessed December 13, 2025, [https://dataloop.ai/library/model/nvidia\_nv-embed-v1/](https://dataloop.ai/library/model/nvidia_nv-embed-v1/)  
10. Paper Explained 4: NV-Embed. How Nvidia turns decoders into… | by Shirley Li \- Medium, accessed December 13, 2025, [https://medium.com/@lixue421/paper-explained-4-nv-embed-1d1e13fc02a9](https://medium.com/@lixue421/paper-explained-4-nv-embed-1d1e13fc02a9)  
11. Encoder-Decoder or Decoder-Only? Revisiting Encoder-Decoder Large Language Model \- arXiv, accessed December 13, 2025, [https://arxiv.org/html/2510.26622v1](https://arxiv.org/html/2510.26622v1)  
12. Qwen2 Transformer Architecture \- Emergent Mind, accessed December 13, 2025, [https://www.emergentmind.com/topics/qwen2-transformer-architecture](https://www.emergentmind.com/topics/qwen2-transformer-architecture)  
13. Multi-Head Latent Attention (MLA) | Sebastian Raschka, PhD, accessed December 13, 2025, [https://sebastianraschka.com/llms-from-scratch/ch04/05\_mla/](https://sebastianraschka.com/llms-from-scratch/ch04/05_mla/)  
14. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model, accessed December 13, 2025, [https://arxiv.org/html/2405.04434v2](https://arxiv.org/html/2405.04434v2)  
15. Qwen2MoE \- Hugging Face, accessed December 13, 2025, [https://huggingface.co/docs/transformers/model\_doc/qwen2\_moe](https://huggingface.co/docs/transformers/model_doc/qwen2_moe)  
16. financial datasets \- a Tonic Collection \- Hugging Face, accessed December 13, 2025, [https://huggingface.co/collections/Tonic/financial-datasets](https://huggingface.co/collections/Tonic/financial-datasets)  
17. MehdiHosseiniMoghadam/ConvFinQA · Discussions \- Hugging Face, accessed December 13, 2025, [https://huggingface.co/datasets/MehdiHosseiniMoghadam/ConvFinQA/discussions](https://huggingface.co/datasets/MehdiHosseiniMoghadam/ConvFinQA/discussions)  
18. Brianferrell787/financial-news-multisource · Datasets at Hugging Face, accessed December 13, 2025, [https://huggingface.co/datasets/Brianferrell787/financial-news-multisource](https://huggingface.co/datasets/Brianferrell787/financial-news-multisource)  
19. thebajajra/Ecom-niverse · Datasets at Hugging Face, accessed December 13, 2025, [https://huggingface.co/datasets/thebajajra/Ecom-niverse](https://huggingface.co/datasets/thebajajra/Ecom-niverse)  
20. Datasets \- Hugging Face, accessed December 13, 2025, [https://huggingface.co/datasets?other=amazon](https://huggingface.co/datasets?other=amazon)  
21. nvidia / nv-embed-v1, accessed December 13, 2025, [https://docs.api.nvidia.com/nim/reference/nvidia-nv-embed-v1](https://docs.api.nvidia.com/nim/reference/nvidia-nv-embed-v1)  
22. Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup | Request PDF, accessed December 13, 2025, [https://www.researchgate.net/publication/353492632\_Scaling\_Deep\_Contrastive\_Learning\_Batch\_Size\_under\_Memory\_Limited\_Setup](https://www.researchgate.net/publication/353492632_Scaling_Deep_Contrastive_Learning_Batch_Size_under_Memory_Limited_Setup)  
23. deepseek-ai repositories \- GitHub, accessed December 13, 2025, [https://github.com/orgs/deepseek-ai/repositories](https://github.com/orgs/deepseek-ai/repositories)