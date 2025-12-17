# Comprehensive Guide to Advanced Retrieval Systems: State-of-the-Art Methods for Complex Query Handling, Embedding Optimization, and Dynamic Thresholding

## Executive Summary

The development of high-precision Retrieval-Augmented Generation (RAG) systems faces significant challenges when user queries diverge lexically from indexed content or when embedding models fail to prioritize semantically critical terms. This report addresses the specific architectural constraints and data schema provided—comprising `chunk_content`, `chunk_context`, `chunk_tags`, and `chunk_questions`—to propose a state-of-the-art (SOTA) retrieval framework.

To resolve the vocabulary mismatch and incorrect keyword prioritization, we recommend a transition from simple dense retrieval to a **Hybrid Search architecture** augmented by **Contextual Retrieval** and **Late-Interaction Reranking**. Specifically, the integration of **Anthropic’s Contextual Retrieval** method will leverage your existing `chunk_context` to disambiguate chunks before embedding. To address the "dynamic k" requirement, we propose **Adaptive-k Retrieval** and **Cross-Encoder Score Thresholding**, which statistically determine the cutoff point for relevant documents based on score distribution rather than arbitrary fixed limits.

Furthermore, the report details **Matryoshka Representation Learning (MRL)** and **Hard Negative Mining** as primary methods to fine-tune your embedding models, ensuring they learn to distinguish subtle semantic differences. Finally, to improve the quality of synthetic questions, we introduce the **Evol-Instruct** framework, which systematically complicates synthetic queries to mimic the nuance and ambiguity of real-world user input.

---

## 1. State-of-the-Art Retrieval Architectures

The core failure mode described—where models prioritize incorrect words or fail to bridge the vocabulary gap between complex user queries and synthetic questions—suggests that a single-vector dense retrieval approach is insufficient. The following methods represent the current SOTA in handling these specific issues.

### 1.1. Contextual Retrieval (Anthropic Method)

Your dataset already contains a `chunk_context` field ("2 line introduction... and all content that came before"). This is a critical asset. Traditional RAG splits documents into chunks, often severing the semantic link between a chunk and its parent document. This leads to "orphan chunks" that lack the necessary context to be retrieved by a query that references the broader document topic [cite: 1, 2].

**The Mechanism:**
Anthropic’s Contextual Retrieval method involves prepending the context to the chunk content *before* embedding it.
*   **Current Workflow:** You likely embed `chunk_content` and `chunk_context` separately or rely on `chunk_context` only for a secondary match.
*   **Proposed Workflow:**
    1.  Construct a `contextualized_chunk`: `[chunk_context] + " " + [chunk_content]`.
    2.  Embed this combined string.
    3.  Index the embedding.

**Why this solves your problem:**
If a user asks a complex question that relies on background information not explicitly present in the `chunk_content` but present in the `chunk_context`, a standard embedding of just the content will fail. By fusing them, the embedding vector captures the *situated* meaning of the chunk. Research indicates this method can reduce retrieval failure rates by up to 49% [cite: 2].

### 1.2. Hybrid Search: Fusing Dense and Sparse Vectors

The issue of "prioritizing incorrect words" is a classic failure of dense embedding models, which sometimes act like "bag-of-words" models or over-attend to dominant terms while ignoring critical modifiers (e.g., "not," "except," or specific technical identifiers).

**The Solution:**
Implement **Hybrid Search** combining:
1.  **Dense Vector Search:** (Your current embedding models). Captures semantic meaning and synonyms.
2.  **Sparse Vector Search (BM25 or SPLADE):** Captures exact keyword matches and term frequency.

**Implementation Strategy:**
*   **BM25 (Best Matching 25):** This algorithm ranks documents based on the frequency of query terms in the document, penalized by document length. It is highly effective for rare, domain-specific keywords that embedding models might "smooth over" [cite: 3, 4].
*   **Reciprocal Rank Fusion (RRF):** To combine the results, use RRF. RRF takes the rank of a document from the Vector search and the BM25 search and fuses them.
    \[
    RRF\_score(d) = \sum_{r \in R} \frac{1}{k + r(d)}
    \]
    Where \(r(d)\) is the rank of document \(d\) in rank list \(R\), and \(k\) is a constant (usually 60). This ensures that if a document is relevant in *either* method, it bubbles up, but if it is relevant in *both*, it is ranked highly [cite: 3, 5].

This directly addresses your problem: if the embedding model focuses on the wrong word, the BM25 component acts as a safety rail, ensuring that chunks containing the *exact* unique terms from the user's "complicated" query are still retrieved [cite: 6, 7].

### 1.3. Hypothetical Document Embeddings (HyDE)

You mentioned that "questions are complicated meaning they don't use the same words as the chunk_questions." This is a **vocabulary mismatch** problem.

**The Mechanism:**
HyDE does not embed the user's query directly. Instead:
1.  **Generation:** An LLM generates a *hypothetical answer* to the user's query. This answer may be factually hallucinated but will contain the correct *vocabulary* and *semantic structure* of a relevant document.
2.  **Encoding:** This hypothetical answer is embedded.
3.  **Retrieval:** The system searches for chunks similar to the hypothetical answer.

**Why this solves your problem:**
The user's query might be "How do I fix the latency issue in the upstream service?" while the chunk talks about "Reducing lag in the API gateway." A direct match is hard. However, an LLM generating a hypothetical answer to the query will likely use words like "API," "gateway," and "lag," bridging the semantic gap between the query and the document [cite: 8, 9, 10].

### 1.4. Late Interaction Models (ColBERT)

To address the issue where models prioritize incorrect words, **ColBERT (Contextualized Late Interaction over BERT)** is a superior alternative to standard bi-encoders.

**The Mechanism:**
Standard embeddings compress a whole sentence into one vector (e.g., 768 dimensions). This compression loses fine-grained details. ColBERT keeps the embedding of *each token* in the query and document.
*   It computes the similarity between *every* query token and *every* document token (MaxSim).
*   It sums these maximum similarities.

**Why this solves your problem:**
If your query is "return of funds *except* for administrative fees," a standard model might just match "return of funds." ColBERT, by maintaining token-level interactions, allows the model to specifically attend to the "except" and "administrative" tokens against the document tokens, ensuring precise matching of complex constraints [cite: 11, 12, 13].

---

## 2. Improving Embedding Models

You currently have two embedding models (`chunk_questions` and `chunk_content`). To improve them, you must move beyond off-the-shelf weights and perform **Domain-Specific Fine-Tuning**.

### 2.1. Fine-Tuning with Hard Negatives

The most effective way to fix "incorrect chunk matching" is to teach the model what *not* to match. Standard training uses random negatives (random other docs), which are too easy to distinguish.

**Hard Negative Mining:**
You must curate "Hard Negatives"—chunks that are *semantically similar* to the query but are *not* the correct answer.
*   **Process:**
    1.  Use your current model to retrieve the top 10 results for a synthetic question.
    2.  The correct chunk is the Positive.
    3.  The other 9 retrieved chunks (which the model *thought* were good but are wrong) are the **Hard Negatives**.
    4.  Train the model using a **Contrastive Loss** (e.g., InfoNCE or Triplet Loss) to pull the Positive closer and push these specific Hard Negatives away [cite: 14, 15, 16].

This forces the model to learn fine-grained distinctions (e.g., distinguishing "Java version 8" from "Java version 11") rather than just broad topic matching [cite: 17, 18].

### 2.2. Matryoshka Representation Learning (MRL)

To improve efficiency and potentially accuracy, consider **Matryoshka Representation Learning**.
*   **Concept:** MRL trains the model such that the first \(N\) dimensions of the vector (e.g., the first 64) contain the most critical information, the next \(N\) contain the next most important, and so on.
*   **Application:** This allows you to perform a coarse-grained search using only the first 64 dimensions (extremely fast) and then rerank using the full 768 dimensions. This often forces the model to structure information more effectively, prioritizing "core" semantic meaning in the earlier dimensions [cite: 19, 20, 21].

### 2.3. Cross-Encoder Reranking (The "Gold Standard")

If you cannot replace your embedding models immediately, you should add a **Cross-Encoder Reranker** at the end of your pipeline.
*   **Bi-Encoder (Current):** Embeds query and doc separately. Fast but less accurate.
*   **Cross-Encoder (Proposed):** Takes `[Query] + [SEP] + [Document]` as a single input and outputs a relevance score (0 to 1).
*   **Workflow:** Retrieve top 50 docs using your current models. Pass these 50 pairs to a Cross-Encoder. The Cross-Encoder can "see" the interaction between words and will penalize the "incorrect word prioritization" that the bi-encoder made [cite: 22, 23, 24].

---

## 3. Dynamic Retrieval (Dynamic k)

Your requirement is: "if only 2 docs are relevant, then only 2 should be shown." This is known as **Adaptive Retrieval** or **Dynamic Thresholding**. Standard `top-k` (e.g., always return 10) is insufficient.

### 3.1. Adaptive-k via Score Distribution Analysis

Recent research (e.g., "Efficient Context Selection for Long-Context QA") proposes **Adaptive-k**. Instead of a fixed number, the system analyzes the curve of similarity scores returned by the vector search.

**The Algorithm:**
1.  Retrieve a large set (e.g., top 20).
2.  Calculate the **First Derivative** (slope) or the **Largest Gap** between consecutive scores.
    \[
    Gap_i = Score_i - Score_{i+1}
    \]
3.  Identify the "elbow" or the point of steepest drop.
4.  Cut off results at this point.

**Why this works:** Relevant documents typically cluster at the top with high scores. Once the relevance drops, there is usually a significant score gap before the "noise" documents begin. This method is "plug-and-play" and requires no training [cite: 25, 26, 27, 28].

### 3.2. Calibrated Thresholding with Cross-Encoders

Vector similarity scores (Cosine Similarity) are not probabilities; a score of 0.8 in one query might be good, while in another, it might be noise. However, **Cross-Encoders** can be trained to output calibrated probabilities (0 to 1).

**Implementation:**
1.  Train a Cross-Encoder (as mentioned in 2.3) using your synthetic questions and chunks.
2.  Apply a **Sigmoid activation** at the output layer to force a 0-1 range.
3.  Set a **Global Confidence Threshold** (e.g., 0.75).
4.  For every query, retrieve top-k candidates, score them with the Cross-Encoder, and return *only* those with a score > 0.75. If only 2 pass, only 2 are shown. If 0 pass, the system can return "No relevant information found" or fallback to a broader search [cite: 29, 30].

---

## 4. Creating Better Synthetic Questions

You noted that "questions are complicated... hence models are not able to retrieve exact chunk." The synthetic questions used for training or matching are likely too simple or too closely derived from the text (lexical overlap). To fix this, you need **Evol-Instruct** methodologies.

### 4.1. Evol-Instruct (Complexity Evolution)

Originating from the WizardLM research, **Evol-Instruct** is a method to iteratively complicate instructions/questions using an LLM [cite: 31, 32].

**The Process:**
Take a simple synthetic question generated from a chunk (e.g., "What is the revenue of Acme Corp?"). Pass it to an LLM with a specific "Evolution Prompt" to rewrite it.

**Evolution Strategies:**
1.  **Add Constraints:** "What is the revenue of Acme Corp, *excluding* international subsidiaries?"
2.  **Deepening:** "Explain the *implications* of Acme Corp's revenue growth on its stock stability."
3.  **Concretizing:** Replace general terms with specific scenarios.
4.  **Reasoning:** "Considering the market downturn, how did Acme Corp's revenue perform?"

By training your embedding model on these *evolved* questions (mapped to the original chunk), you teach the model to bridge the gap between complex user intent and the source text [cite: 31, 32].

### 4.2. Persona-Based Generation

To mimic "complicated" user queries, generate questions using specific personas.
*   **Prompt:** "You are a frustrated junior engineer who doesn't know the correct terminology. Ask a question about [Chunk Content] describing the symptoms rather than using the technical names."
*   **Result:** Instead of "How to reset the DHCP server," the model might generate "Why is my internet connection not assigning IP addresses automatically?"

This diversifies the vocabulary in your `chunk_questions` field, increasing the likelihood of matching real-world queries [cite: 33].

### 4.3. Generating Hard Negatives for Training

Use the LLM to generate **Distractor Questions**.
*   **Task:** "Generate a question that *looks* like it is about [Chunk A] but is actually about [Chunk B] (a similar but distinct chunk)."
*   Use these distractors during the fine-tuning of your embedding model to sharpen its discrimination capabilities [cite: 34].

---

## 5. Tagging and Filtering Strategy

You have a `tag_model`. The integration of tags should be handled via **Pre-filtering** rather than Post-filtering to ensure efficiency and accuracy.

### 5.1. Pre-filtering vs. Post-filtering

*   **Post-filtering:** Retrieve top 100 vectors -> Filter by tag.
    *   *Risk:* If the relevant docs with the tag are at rank 101, you miss them completely.
*   **Pre-filtering (Recommended):** Filter the database by tag -> Run vector search on the subset.
    *   *Benefit:* Guarantees that the retrieval is performed *only* on the relevant subset. Modern vector databases (like Weaviate, Milvus, Pinecone) support efficient pre-filtering (e.g., using HNSW with ACORN or bitmap indexing) [cite: 35, 36, 37, 38].

### 5.2. LLM-Based Tag Extraction

Since your queries are "complicated," simple keyword matching for tags might fail. Use a small, fast LLM (or a fine-tuned BERT classifier) to extract tags from the query.
*   **Prompt:** "Extract the relevant technical tags from this query. Map them to this allowed list of tags: [List]. Query: ..."
*   This ensures that even if the user uses a synonym, the correct canonical tag is applied for the pre-filtering step [cite: 39, 40].

---

## Summary of Recommendations

| Component | Current State | Recommended SOTA Upgrade | Solves |
| :--- | :--- | :--- | :--- |
| **Retrieval Algo** | Dense Vector Search | **Hybrid Search (Vector + BM25) + RRF** | Incorrect word prioritization; rare keywords. |
| **Context Usage** | Secondary Match | **Anthropic Contextual Retrieval** (Prepend context to content) | Loss of context; orphan chunks. |
| **Embedding Model** | Off-the-shelf | **Fine-tune with Hard Negatives** & **Matryoshka Learning** | Incorrect chunk matching; semantic discrimination. |
| **Query Handling** | Direct Embedding | **HyDE** (Hypothetical Document Embeddings) | Vocabulary mismatch between query and doc. |
| **Dynamic k** | Fixed k | **Adaptive-k** (Score Gap Analysis) or **Cross-Encoder Thresholding** | Showing only relevant docs (2 vs 10). |
| **Synthetic Data** | Simple Generation | **Evol-Instruct** (Iterative Complexity) | Training model on "complicated" queries. |
| **Tagging** | Tag Matching | **Pre-filtering** with LLM Extraction | Efficiently narrowing search space. |

By implementing **Contextual Retrieval** to fix the data representation, **Hybrid Search** to fix the retrieval robustness, **Adaptive-k** to handle the dynamic result count, and **Evol-Instruct** to generate high-quality training data, you will build a retrieval system that is robust to vocabulary mismatches and precise in its selection.

## 1. Introduction

The design of a high-precision retrieval system requires navigating the trade-offs between recall (finding all relevant information) and precision (filtering out irrelevant noise). The user's scenario presents a sophisticated dataset structure—comprising `chunk_content`, `chunk_context`, `chunk_tags`, and synthetic `chunk_questions`—yet faces common but critical failure modes in Information Retrieval (IR): vocabulary mismatch, incorrect keyword prioritization, and the inability to dynamically threshold results (the "Dynamic k" problem).

This report provides a comprehensive analysis of State-of-the-Art (SOTA) methods to address these specific challenges. We move beyond basic Retrieval-Augmented Generation (RAG) implementations to explore advanced architectures such as **Contextual Retrieval**, **Hybrid Search with Reciprocal Rank Fusion**, and **Adaptive Thresholding**. Furthermore, we detail methodologies for fine-tuning embedding models using **Hard Negative Mining** and **Matryoshka Representation Learning**, and we propose the **Evol-Instruct** framework for generating robust synthetic training data.

## 2. Addressing Vocabulary Mismatch and Complex Queries

The user reports that "questions are complicated... hence models are not able to retrieve exact chunk." This is the **Vocabulary Mismatch Problem**, a fundamental challenge in IR where the user's query language diverges from the document's language [cite: 41, 42].

### 2.1. Hybrid Search Architecture (Vector + BM25)

Reliance solely on dense vector embeddings often leads to "semantic drift," where the model retrieves documents that are broadly related to the topic but miss specific constraints or keywords (the "incorrect word prioritization" issue).

**Recommendation:** Implement a **Hybrid Search** system that runs Dense Retrieval and Sparse Retrieval in parallel, fusing the results.

1.  **Sparse Retrieval (BM25):** BM25 (Best Matching 25) is a probabilistic retrieval framework based on Term Frequency-Inverse Document Frequency (TF-IDF). It excels at Exact Keyword Matching. If a user query contains a specific error code, product name, or unique identifier, BM25 ensures documents containing that exact term are retrieved, preventing the embedding model from "hallucinating" a match based on general similarity [cite: 3, 4].
2.  **Dense Retrieval (Vector Search):** This handles the semantic understanding, capturing synonyms and conceptual relationships even when vocabulary differs.
3.  **Reciprocal Rank Fusion (RRF):** To combine these disparate scores (BM25 scores are unbounded; Cosine Similarity is 0-1), use RRF. RRF relies on the *rank* of the document rather than the raw score, making it robust to different score distributions.
    *   **Formula:** \( Score_{RRF}(d) = \sum_{alg \in A} \frac{1}{k + rank_{alg}(d)} \)
    *   This method ensures that a document appearing in the top results of *both* algorithms is prioritized, while a document highly ranked by BM25 (due to a rare keyword match) is still surfaced even if the vector model missed it [cite: 3, 5].

### 2.2. Hypothetical Document Embeddings (HyDE)

To directly address the vocabulary mismatch where queries are "complicated" and distinct from chunk text, **HyDE** is a powerful technique.

*   **Concept:** Instead of embedding the query, use an LLM to generate a *hypothetical answer* to that query.
*   **Process:**
    1.  **User Query:** "How do I mitigate latency in the upstream service?"
    2.  **LLM Generation:** "To reduce lag in the API gateway, you should optimize the connection pool..." (This generated text uses the vocabulary likely found in the documentation).
    3.  **Embedding:** Embed this generated answer.
    4.  **Retrieval:** Search against the `chunk_content` embeddings.
*   **Benefit:** The generated answer acts as a "vocabulary bridge," translating the user's complex intent into the specific terminology used in your dataset [cite: 8, 9, 10].

### 2.3. Anthropic’s Contextual Retrieval

The user's dataset includes `chunk_context` ("2 line introduction... and all content that came before"). This is often underutilized if treated as a separate field. **Contextual Retrieval** is a preprocessing technique that solves the "lost context" problem.

*   **Problem:** A chunk might say, "The revenue increased by 5%." Without context, this is ambiguous. A query asking "What was Apple's revenue growth?" might miss this chunk if "Apple" was mentioned in the previous paragraph (the context).
*   **Solution:** Prepend the `chunk_context` to the `chunk_content` *before* generating the embedding.
    *   **Format:** `[Context String] : [Content String]`
    *   **Result:** The embedding vector now encodes "Apple's revenue increased by 5%," making it retrievable by the specific query.
*   **Impact:** Research by Anthropic suggests this method, combined with Reranking, can improve retrieval performance by up to 67% [cite: 1, 2].

---

## 3. Improving Embedding Models

The user asks for ways to improve embedding models, specifically to handle incorrect word prioritization. Off-the-shelf models (e.g., OpenAI text-embedding-3, BGE-M3) are generalists. To fix specific errors, **Fine-Tuning** is required.

### 3.1. Domain-Specific Fine-Tuning with Hard Negatives

Fine-tuning adjusts the internal weights of the embedding model to align the vector space with your specific domain logic.

*   **The Critical Component: Hard Negatives.**
    *   Standard training uses "In-Batch Negatives" (random other documents). This is too easy.
    *   **Hard Negatives** are documents that are *similar* to the query but *incorrect*.
    *   **Mining Strategy:** Use your current embedding model to retrieve the top 10 results for a `chunk_question`. The top 1 is the Positive. The results at ranks 2-10 (which are wrong but scored highly) are the Hard Negatives.
*   **Training Objective:** Use **Contrastive Loss** (e.g., InfoNCE). This loss function penalizes the model heavily when it places a Hard Negative close to the query in vector space. This forces the model to learn the subtle distinctions (e.g., "v1.0" vs "v2.0") that cause the "incorrect word prioritization" [cite: 14, 15, 16, 17].

### 3.2. Matryoshka Representation Learning (MRL)

MRL is a training technique that creates "nested" embeddings. It forces the model to encode the most important semantic information in the first few dimensions of the vector [cite: 19, 20].

*   **Benefit:** It creates a more robust semantic hierarchy. By forcing the model to compress information, it often improves the quality of the representation, reducing the "noise" that leads to incorrect matches.
*   **Efficiency:** It allows for **Adaptive Retrieval Speed**. You can perform a super-fast initial search using only the first 64 dimensions, then rerank the top 100 using the full 768 dimensions [cite: 21, 43].

### 3.3. Cross-Encoder Reranking

While not an "embedding model" in the strict sense, adding a **Cross-Encoder** is the most effective way to fix prioritization errors without retraining the base model.

*   **Mechanism:** A Cross-Encoder processes the Query and Document *simultaneously* (Full Self-Attention). It can "see" that the word "not" in the query negates the word "available" in the document. Bi-encoders (vector models) often miss this interaction.
*   **Deployment:** Use a Cross-Encoder (e.g., `ms-marco-MiniLM-L-6-v2` or a fine-tuned version) to rescore the top 50 results from your vector search. This provides a massive boost in precision [cite: 22, 23, 24].

---

## 4. Dynamic Retrieval (The "Dynamic k" Problem)

The requirement "if only 2 docs are relevant, then only 2 should be shown" necessitates **Dynamic Thresholding**.

### 4.1. Adaptive-k Retrieval

This method dynamically selects \(k\) based on the **distribution of similarity scores** for a given query, rather than a fixed number.

*   **Algorithm:**
    1.  Retrieve top \(N\) (e.g., 20) documents.
    2.  Analyze the similarity scores \(S_1, S_2, ..., S_N\).
    3.  Calculate the **First Derivative** (rate of change) or look for the **Largest Drop** (Gap) between consecutive scores.
        \[ k = \arg\max_i (S_i - S_{i+1}) \]
    4.  The intuition is that relevant documents cluster at the top with high scores. The first large drop in score signifies the transition from "relevant" to "irrelevant/noise."
*   **Result:** For a precise query, the drop might happen after doc #2. For a broad query, it might happen after doc #15. This satisfies the user's dynamic requirement [cite: 25, 26, 27, 28].

### 4.2. Calibrated Score Thresholding

If using a Cross-Encoder, the output is a relevance score (usually passed through a Sigmoid function to be 0-1).

*   **Calibration:** You can determine a "confidence threshold" (e.g., 0.8) by evaluating on a validation set.
*   **Logic:**
    *   Run Retrieval -> Get Top 20.
    *   Run Cross-Encoder Reranking -> Get 20 scores.
    *   Filter: `Results = [doc for doc in docs if score > 0.8]`.
    *   If only 2 docs have a score > 0.8, only 2 are returned.
*   **Note:** This is more reliable with Cross-Encoders than Bi-Encoders, as Bi-Encoder cosine similarity scores are often not well-calibrated (e.g., a score of 0.75 might be irrelevant in one vector space but relevant in another) [cite: 29, 30].

---

## 5. Creating Better Synthetic Questions

The user notes that current synthetic questions are too simple. To improve retrieval, the synthetic data must mimic the complexity of real user queries.

### 5.1. Evol-Instruct Framework

**Evol-Instruct** is a method for automatically complicating instructions/questions [cite: 31, 32].

*   **In-Depth Evolution:** Prompt the LLM to rewrite a simple question to increase its reasoning difficulty.
    *   *Prompt:* "Rewrite this question to require multi-step reasoning."
    *   *Example:* "What is the error code?" $\rightarrow$ "Given the system failure logs indicating a timeout, what specific error code should be investigated?"
*   **In-Breadth Evolution:** Prompt the LLM to add constraints or rare vocabulary.
    *   *Prompt:* "Rewrite this question to include a negative constraint."
    *   *Example:* "How to restart the server?" $\rightarrow$ "How can I restart the server *without* clearing the cache?"

### 5.2. Persona-Based Generation

Use different "User Personas" to generate diverse questions for the same chunk [cite: 33].
*   **The Novice:** Uses vague terms ("the thing isn't working").
*   **The Expert:** Uses precise, technical jargon ("latency on the ingress controller").
*   **The Manager:** Asks about high-level impact ("business continuity risks").

By training your embedding model on this diverse set of questions (all mapped to the same chunk), you make the model robust to vocabulary variations.

---

## 6. Tagging and Filtering Strategy

The user has a `tag_model`. The integration of tags is crucial for precision.

### 6.1. Pre-filtering vs. Post-filtering

*   **Recommendation: Pre-filtering.**
    *   **Mechanism:** The search engine first selects all documents matching the tags (e.g., `tag="finance"`). Then, it performs the vector search *only* within this subset.
    *   **Why:** Post-filtering (Vector Search first, then filter) is dangerous for dynamic \(k\). If you retrieve top-10 and filter by tag, you might end up with 0 results if the tagged documents were at rank 11-20. Pre-filtering guarantees that if relevant documents exist, they are considered [cite: 35, 36, 37, 38].

### 6.2. LLM-Based Tag Extraction

To handle "complicated" queries, do not rely on regex for tagging. Use an LLM to extract tags.
*   **Workflow:**
    1.  User Query: "I need to fix the payment gateway latency."
    2.  Tag Model (LLM): Extracts `["payments", "performance"]`.
    3.  Retrieval: Pre-filter for `tags IN ["payments", "performance"]` AND Vector Search for query.

---

## 7. Conclusion

To build a retrieval system that handles complex queries, prioritizes words correctly, and dynamically adjusts its output size, the following architecture is recommended:

1.  **Data Processing:** Use **Contextual Retrieval** to fuse `chunk_context` into `chunk_content`.
2.  **Synthetic Data:** Upgrade to **Evol-Instruct** to generate complex, diverse training questions.
3.  **Embedding Model:** **Fine-tune** a base model (e.g., BGE or OpenAI) using **Hard Negatives** derived from the evolved questions.
4.  **Retrieval Engine:** Implement **Hybrid Search** (Vector + BM25) with **Pre-filtering** based on tags.
5.  **Reranking & Thresholding:** Apply a **Cross-Encoder** to rerank results and use **Adaptive-k** (score gap analysis) or **Score Thresholding** to dynamically determine the number of results to present.

This approach leverages the specific strengths of your rich dataset while directly mitigating the weaknesses of standard dense retrieval systems.

## References

[cite: 44] Shawhin. (2025). Fine-Tuning Text Embeddings. *Medium*. [cite: 44]
[cite: 45] Databricks. (2025). Improving Retrieval and RAG Embedding Model Finetuning. [cite: 45]
[cite: 34] Johnson, S. (2025). Fine-tuning embedding models. *YouTube*. [cite: 34]
[cite: 11] Puig, M. (2024). ColBERT and Beyond: Advancing Retrieval Techniques. *Medium*. [cite: 11]
[cite: 1] Analytics Vidhya. (2024). Anthropic's Contextual RAG. [cite: 1]
[cite: 2] Anthropic. (2024). Introducing Contextual Retrieval. [cite: 2]
[cite: 19] Snayan. (2024). Matryoshka Representation Learning. *Medium*. [cite: 19]
[cite: 31] AI Driven. (2025). Generating Realistic Synthetic Training Data. *Medium*. [cite: 31]
[cite: 14] Whyamit. (2024). How to Fine-Tune Embedding Models for RAG. *Medium*. [cite: 14]
[cite: 15] Moreira, G., et al. (2024). NV-Retriever: Improving text embedding models with effective hard-negative mining. *HuggingFace*. [cite: 15]
[cite: 3] Weaviate. (2025). Hybrid Search Explained. [cite: 3]
[cite: 22] CatalyzeX. (2025). Cross Encoder Reranking. [cite: 22]
[cite: 25] Megagon Labs. (2025). Adaptive-k Retrieval. *GitHub*. [cite: 25]
[cite: 8] The Sequence. (2025). Hypothetical Document Embeddings (HyDE). [cite: 8]

**Sources:**
1. [analyticsvidhya.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFLTxbPeCOHMttsv-YtFxKIl8RyDAAUF5MfNk292Bdd1Hg3dvHRCD_vSOCmmE8AX13Y-B5G_4A55hTuG4Jk0Gyuwx4U8xfrZrOMFaPAff_Z8qRcNKCN7wLT6kK_mWakh7eL6nZSV-ptyy-g3_EAbXXhkFyi7H9YZfy1EsdYJw==)
2. [anthropic.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFPBKsLDXeXSqmAfj2Y7fnbh4UGaBf0-IEYNZZRNVJnZsF1ogf2wntj8FGwWObGGGiJbPTcMpTLBvcXimpic1oP3vUtNqKyppfeFz-GP8Z2W3_yfztbe0T-TOvyDkbQAaq0AJZswBl4fO8=)
3. [weaviate.io](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGd4icmUBNDpC5qhMRqAr3roYUbILCi7GRRxT67XgF4CcP8_duav3TT_O5PlhKyIKJWXxqiWKp14vAc2AI-6is0lhWOTWt6k-ROAWr8u6INnjYSXW1yktXN6n7Wo6YOsU42VrXD1T8=)
4. [mongodb.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFHiB0ScRYjSyoTkQlBKk-brQNXwa-AjOppl3lxmD2ORJL106tCBk_o_efsslg4YxVCm79uFrAYEUACdYb1q8NF6md9E-cskrcNYZf9ijZzWMqSRXs7Vj-CRdm8jHnXfgnvacolPjHmW5YyJgKg4zc3B0YYCyOdHRZZqAs=)
5. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEatXjWlVytOUB4ddEAtUxmbiC8HF0iQ7rPQ0FeJkwoD31OSnDLuUn6ZS9hMLpC8sW9MhrZ2CFK_WVmgFdrMLbhUei30fHh5YdZumFtQa2ALGVaB7ropCTkyqpM1bzOl1dnIojI_gz0mYRFFHG4C7op3huDwn96jUdJKHc91DByv2d8v-B5lQERSaTgfT2GvjByHlSGsSpeMuc=)
6. [dev.to](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGse2JKuAaNTBCX1ZuwJxEBQiFT4Kk7vIpLMr1evIefswlhFy654ZB94DMXLna-27kmi7yP3a6go-CmNSiiQz2-2MxtuDd-nb68iDUD9Rjju4R0JQfw03mM63DEJn4AdEDnpaCI7zo_rBG0Yins9qucT53JvRAjdhOrbcWpGnRwpboYpiLwQHYNHlHfKTvsOERBZIjRm8-8d2I6K86UgbtIg1elufkRaBmo)
7. [opensearch.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEkis6800LOIfNpZA483tRtEaz8CmtQ077Ph1zv0oEQ_bp40f_CDWCpknEtqEpSKbme8NixriZ1gfA3wfdrnpXqYNS2iTWojl9Vv_naLfIV46M4dzFDzKoj5Lx3ypm9FFkT_IG4AbMQlmlt8tmI_vpBwEyUY-RyrTUpc5GXtFHbg82wXBqmx8o9g9Kzth7EgcL-qJucM8YgZUVr9PCKtZs=)
8. [substack.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEErH92uUc3hEGOw_GTiZytCUB-ow8rj0ecv8PrrZSaNXrVg41RsuR2K-UhMv-yGBadUC1fT7oowBZTUQVYRHZs70Tp9ky3GpYu1jlUlJwt39s81oUj6m7LOtlUsZP16Sld8En75t10FXiShYs_eHj68ixS2Db_2Anv7g==)
9. [machinelearningplus.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGDxQz0dF2-sJpbzhEjP-Z-vVgEYI4jjrJholJaD6ucd5btC4N8BRm8wj5b9sh7m3ln5CkEW5fIRUvC7ntjQGFO5cs1lioBmer3cFD5P1bFMYIRTV5CvtslspBwQXCL4J8AXSlyc7fpulfGzHwdDZGY8b3eOyRn7o2YP6VzCeTLK7jWICACzeDgmu96TcU12NNPe033OCn3kcBdj0sYgsqS3BNmISzj4KWSZqVbMu_m)
10. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHy5VLGnK5bcMRCBHeNrwqOWtzwxCLTJhNd8PAEt5AyAbEFjLU6UX_7va36s-GdFZVvNYUuVOtdiqWapaL0qGM3qrcTdMsKIjqlbcWTijU_RdaFmPt-ska_Wv3hil27qgk5YoohX5P6fOBLToMyXEaDTopu6c7rK778ANwn5QfiI_lBWvHsrWh2O1bglIfe0v37yndpUcPWjcn0UkKyh10shVscKA9-VF-JcGB0x1oYmFgPklBRnxs202lqx8WXD0kKZ-s=)
11. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEUVpjT_vcm-3lfv8sBPTRHQWKYyzVhyoQeVe3uEepUCm4NY30yuGZAxgjsuenPlj9jJy0AVBztTQl3SEvusx1jiS8wj0QNHaVqCMgJweQBYExvl43qp8EHzExe08QhpoxjoNmDXYUp6Sk89kag_p6Gmc89C_RI6mTqnjivb7ypVp8gXJFISJ9KwfXxYIAD)
12. [youtube.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFBXuZujCK0RDG_O0YInHs4xlc-Ys5TWHYa5YUDa01CFyejo13CkbCkx-flCdcseqCwiQCIwCOH_MkRm9qCg3stN5V7KmCBxO6sv3dxzqWr9Mfm8iq69v-AMxaPs6aQ9Awr)
13. [reddit.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHbNl5U0oicU8WZF09-020eK-EqPoi-Idx6qkSmh-myu4wUGFbcxyoHsOexR4qLxE6muRZ_8ANVgZn6nrwe-m7UVCxnIEZ9Ze19daaSt0RixwSa0kxgfkdgUeLymS_haAPQppJXu6JBWY4IIxWfJ_fGKnxzO7Am6Llg7_oEagvMhQIoCPrVnWa2imePBhCX)
14. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFw3qlF0REdSwU2JDdmjNy8fIkOqorVq1zFQ2lx2xk9S2G1fIO_BNggWP0or1J_5JWwy8ezYmoGzmHrBRw7kpdwi4KKF9YW5EDlZtcdBFjsyNpXRvpkREJpxHbvXaHB0HqbkDUg7mzDsr2nsFBATWAoUjlPK7IZB_n3Ri0tt5Ua2gKic5A45PS20P9lwsjcz5FIEm1lskawlL256KW0YdxttBhNC6nN-HuhDg==)
15. [huggingface.co](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHZ5qi-Deou6bnOwnVUdbisauWU_WQnWVfv7225dIBohZR6mxoDKFnLih-SmKfFBdFme7Q3XMRqKnRbUme1M5kHqAXfdHBykaIDT3QF7DcykU-lX7CxHa00hi5uKMMm)
16. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGxBVGgrwsEUeUkPDCCPtt7FQoV1yLJcvGK1gK_OAzwx-PTYgSIwPjXhQYBtYyPCKUrE4SfZVN0uLJ0eyyeVL77ik99mbTFpBBxwzJmXIDkzs7vwVWMqA==)
17. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG7ER4QkyCP6WgyDCfX520XBV4yhJR_fKukXlbInZcjQJFTxzSonVL_eC9RBmTFGuBJsGbnGRBZnMHsB6ayVuExnACu0DCj-fdMCFBFjv9bswta55u3Ng==)
18. [github.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEaAp_GD8BDA9U2WXGdq2L-VY-ovRYq16Y6MYIKnahtzgsvHGFrLyamAMuWL8NbvAqZgF_FpjljcjXLDRPU-unVpix-uTHcjXE6R5SF6FkZrpKRlac7gp4Wq3AWasLck5lTfXq9ulSJ3mxQwg==)
19. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGl7pRn6e-cLCwXK-hz_Yh11yH965OSJGtV0e0LlF_0CnZ2TnOxzBKZJUxW5ZxVaUiQlGmAzzpmfIjaLfgSW_mOI6HU1vb7eGc7ToepqeIoe7zS28_sbZWICdVyP6vviAyjJyt_FNocRHAQ-GMmFLTboBhAV_9XjK-D0v7EY2jHf6-TJaU8H2IuLGvmhXfOhw==)
20. [mixedbread.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFj_Ze-h3wcBEd2_N9tAX1GcKE8_iZTjT0Fup3yE6V-xd6uip91uytvuJtJfSklf2_1w1KVyLI0p0NGWuFSxRKej0tXv0Z-HO46q0KQtm3FcQ6N9Vg8UIGFu2eynUAkOZv1OmEFeTjTjpIWJjLF)
21. [marqo.ai](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFxAVEAWLQ9PcKKzw45ylYyZd8H1JSa3p3dhnL88KAS5cKCuRKjIp1mQvrTjkxDpqcd3CZaQFk-GpIJNwDA2ky5OVuq39UUN_PBmmRCBoiX5apGppgoyGNIBrZejNeTfCRL63N_eT4u2ktH6M_X2IS9iXkcLYZn5kmag-slWjSTTtMwOxg5Fe-JV5k_QLe1ZnRW2dQf2do4-GZWDa01ccyWJA==)
22. [catalyzex.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGszh3i1bnkqJfXvwpQ5BUG_Yk2gBzlewIOmj2CFaYvHsdJRSvZypbay852hjacuQPoKX3nb4_068QcMRgHNvx3EMSI3UVjetz9bbd3wu6RYvt4c7p0BBX792jJvmyFrZ4W6Py1xJIS9wLq7haF)
23. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF4xMX9JI4zx1weiZYVYoCNaEFkZdlrPFSSNjOn-G-cq1BHrQUM8d-LLo5xew4v1DvbDQoul2JP2NBmYKp4B3zzt1GHGzfq94mbFhVgI0rsnx9t4IjIuGe_ose_pGjl9bNwBUC4IleJM3VPp1Z4u85UcRebWG2Ya_Cfsnd9GQ5sRB0xwNys7U15vnW5NSiC8xEYC88GNez6_TaFP2QbwHp7kSzRhxYYRPQxv6Q=)
24. [sbert.net](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFbchIiSMo3ckb5IIbMWBe4FRwoyQezcV_jS0szHaoHRA3aZ9wbz53l3KMcZiA6L7DN5hJIZWU4yuW2HuXhMm3mSeihaieVk9Dz5HGEKaj4qX3bO0-UL_pBQtx1bhbFpagw1KzOB6X74D36v9L_MIKZWTZMY_ouYIdKqqxMKvPtOVgkwCy1FrYx6SEh5Eun)
25. [github.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE-vzw1grWtP3nBm4NsPyKTEY8N-do7zWNBX4UILYG60L1CnSUhZLdHI6j4aUuYUOuDk0H6DZgwuu-NZosl9Ixp4CXCwTlBp29p9Zx2rQebiy7NQxax97oHtj5IqY1D2EbQNfOCayH35O4=)
26. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGlPIhxYtqckoopfUdPvxZ-Apf1TOnJQYOqO_9KbqHJlO8lAPE27tXchCF68hErvgUHhu_-v78RQQUnYFOVH57Q2R38IzyXWvyEytQrBmCjDdTv4r6tkWMB8acrlf32Hyh9qi2cOTJTtkgVpLz-Gh8MI68PwtejS88VNhWA3rQnMqUQRY6r2_IrTzZPDAUWuCEx-By6AEJe8_-YsAm2j7Nv8MaqkTb-14LH9xpZ1coleSSQyNbPOnsjX7DWkRQEG5KwFbH_Qx3wItU9ahE=)
27. [aclanthology.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEfpk1uuNmEJLK6XPmlVrH1F51xM-axH3deGyh4uoir-hSUjBtFT-HquSLiFKr48q5y7FsId4V3ISyV7NzTZ974LqRp7DASXkuxLANIKAKztz2QLDxn43aakiafSdYYenisbWKd9wk_)
28. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHw7BorUnL2S0H7m3bv04UaSZD0QtN_X28yn8Nfa0xBw0LXzDxQLQ7pIZ0I3XNmcZ4J739xKQxd72pMFBlvF0rKOdusJRTOXnJgBhJ7baEYa9SEEjD0oKPhLsTMMpoJkmRDby5RRBquFJ9-7EgjzwxOPB6eSGlvH74ldmFbOYkaFXVtuQl7LuzeKtL_OEd7_9F_2iXsWb3JGq2Agf-zxtvrSg==)
29. [github.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFh7ah1nDc9u56kk8JzewW8l-HgKWS5zJQafoztY5Dv2NwUIjx4Szyt7WNwvakzXiMzfomwoXO94P7GnPGxcAXV8E8s4K8OiIrzEm0H4p52YY9pB8NQ18f6kGvof4cZeeA1FHyjG61XPhZZl0-eAQUp8BGqjVye)
30. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFj0UyKuYt8bYXQ8mpAOXee0TQS9kBwnQy6eqFvvVygRdjjQFW_x2o5lSLG8VqyyjT2TvHMdAmh510fLoX-jY1JHPfOPlldCI2chQoLZbqDRqjTl65jiNoVQcmpjwc0b1HgRbZQ3oYQqpKqOFSMQilYI5Q1qySZYQb8QG4hyKlwWDlsS-nV03lvb4CFBDxdCg_FvILeri1xedmQ6BNexpMHFQ==)
31. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHrJv8qmKCY4lvYaFH6bfNZa5FaNsbCPVKjbCBiQGPhTlWsHfdrvrr5WSGYJ1WrBr7sLcxU7S4gvDaJWSSbSOqtug8tuHGSELDlDvvoBMb0WmDpgKs-wOoIkiSWFm66Gss7Lwsz3kVQ5_leaLo1QQUjHHFUZE-OZBCLFxFw6licFb9AKRtQh66gDf5qns1fpJSF_rtcpJiJt-jpO1Ew8k6oBZ6QrbdAaYDMcmHas1aYgPXV3HQ_XwruvGn93sd5suxHDV7Nx8RiFtA62f9lgxTXpXLT0xFPR483-rUpkNG8KC4856PM)
32. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGVaExLaeZ0DDalVtXdBA4zRhd0KhHaWz-t1hpsqPlQpqKUfet7ac5aZkoHhtRp8odmviI-1_m71q8LXB_qhLDTTBycLPTaJmpChP5ZjBVoskY-LWUkddRVubw9dR1OTdT71fVXEcTpGCQ7CYP6ALvtTiftc8NMSP9owRROev1CJ8ucbbGMjOxanXO4KTDqjlpY4ThAJ1u240bbovvxkEpTkA==)
33. [github.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHZFJ45-gK6nHUkkTC6vnIC1NElevnHrwgl-d5cWW3RRIz-6mVXPtqOH_esdGv7XvVFX3GqHc75aMWai0wLTLEzIULWll9T91a_kQTxHfhaNhggSwH2NnfI57qUeimO_tZAn0c17RW8J8KAt309_r1THZDly9b-oCesQp9U969o6B9m3AJL5tfN_nw5P0YeEixf9oxLOo00piqcRo8rP2Xu9OhLEiuHN_SVU9VjDCd8yqm0BX3nXhVCJWGjYUrztlkVK72lTkE9AJbRvOI4zAN_aDfcG1cX-4AJpD_hZ3jMr93zJVTybC3hpK06WCqnr-cQ9zWgqCYzlfL-2GLj7xA0HyprLRekZyefXqdrPzVGELRwHvghdQ==)
34. [youtube.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFe4vuM3AFOr8iQeVaf6FAcV2KEhKCfdM9CmgvpeeBrzMI2m3W_HTPlyng7JD8fTvWSW5ZiRrTDdYNvVnJviR7IQA18tt7X_YmHAd8-tlJth1UEzJK7rho_xOep9VVw_bQb)
35. [dev.to](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEl8QF-2ILynTgX8CWlVMMFRcOZN4Uvzcg3dW_i4Y3I-dhF4R_CYnj5Y8-qfsxkS0VN3bARcYvD3b-HjzZ-hm_z_nY37jYdKsWqfbl4vhFHFF97vAylXVwcxxmmVSUBMwo6IeWgC59jrL8z2vg8z6MFt4NI_EJRXQVPXhT-JclG8830rGkbFUNDzoqImpk1mFHeiZ3HtQLlk44=)
36. [apxml.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEANYzIXGTPtJwSThGTnSKSQl1iFLvx5VwiyG_jBgfgN-6Oqjbmmo58BAdJ1NyWLRe9TBVt0whsdJcg5FgCLbN5kaS5QORTqsRk7ChK9TjSoA9y4mJzoGcgcojMEl0oAJeBbbnOaN_l-or2JLRybqV9KquB0Kz5A5QcYasGxQY5fmjlQWFfL3e4g9ZWYSNzOeey1ogZZlBW2k3zmcJw9h09SH40knc-oP32_PNW6d3q7lEh7tuK_TwD9A==)
37. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGxmXx762fSPoI9C3TEijc8MbwvlU3sLBQ4c_oRgl5n6NFHzpEF9Sdg5m29aCcrQiSWcN7VHO_SOBsGgZHyVoWAcwTHDmgivZ_m3SaOv_GyJqsl_RfFRrki4i9uI6EQuivW2DSt-42pYjLZb9XR6aorH1czJa9PM-18lZlj3JSMupEdG1kL8ltheVVMErIpZjAU4DiSjuu3WQ16du9CuGvYfw8q4KJjOG2VNOWz4cFX0OCqwSKFJl9RxYTiqwHgAunIaf4=)
38. [weaviate.io](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFD9R_3Ry9od2gjFhzaHy6bEhpzLNaBxzJYh8boiZMjQT-glblUwCutM6ZR9mLBI0Vy6930DYluXI0oP8Hi3BKyf36R2Ri_tYYia0o_Bf7cuMK4mfQMb4EAWw0wxiJ7GT9aHJk4B6WdbvtR)
39. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFg8Vqev9nQW_6kNpRZy0VkC3CBazGA_AkCIH8y9Tw0rrV65QmD6lO16UJkoxuOd9OEfyIGMpn-7JcF9tUnzrqff7aq3SG_7woJ7cYgDAalDONZ4lz_mT4OgtgUwVle7qsgxhq8fSX6Pw3L73Vso6GtkzYbT7A25qn1x2YAi753R0YYpStGdt1fkFaSoQ2cw7NwyL8eyaedrImGuwwsDNaUD1QGJhM6sfHyp9n6o90rvOVTzvDuB5s=)
40. [youtube.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGpSL4OiRetUFh5qUMI-mP3e7rnMof-hbPbReM25_kkMVZkyJ7uiNgXjgOLAHXmSeSVGer8EBIxkkDLONAW42rxTDsUW8_zc-exgX5rserNWO1mwi77pALIGN68-vKqcCXu)
41. [sease.io](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEPi8l3_Y0VbREvTQXcWmMT0RMai4bxmcm5lNDR-Ca7BxcPPn8M8UkvOBMjejqNJaPOMQGjAW_rR-yHkdNNEl3bFUo23MtYsNC0d5wogTRmVmjAtetdL4GhujiX6_W1mU0y3EbXVUBxVgbB3HtUVMfqQ_5Na_EsuPNOeuyR6Etf91n08_5AQZnK)
42. [researchgate.net](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGDhaR4N5tH55SLx6pn34g2Gtji3QDAY0rtzAiKweBbVSfOoZZyR74QwWkAplFwsrBI9vEW3pDMjeHuTBSN2ImMz1bwsjaAAlW06INwntvLEPnbEijJ4QVFOzGUGOpsetpS1g7AReoPQEWOBJTNHHONpN6Gs9Q5T6LPNzeaw11gRSWLYKFOcoqyx7ApPY-6jUP643ibSRGb7uwjMFjGVYMFAjKMzESeYWdlreW-2NmGUEn49VPmjz9KfaTPItI=)
43. [aclanthology.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG4vfHnkvE72Y0_fYdv9IgmiTyDp-rxA3gSrzb7dQyolvw-PJO-HMUdbE1e0wwo6FBb67jhNwDFYA_vHlJDq0KS1vIC-BdN9blZIqhPVal78EyercI9J5FO9RcsAB2fwr0q7BVtZnX1)
44. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHoJzXEHVwOna5DQ_NFzRufA8gix_l2TqVF2gR8UxGIyiq5TZ36UxO9--hZLze_JmVz30pe-io6ziZy9Yh2vf_Jn7QpfFBwYDMbInkPP2_CCmGruSeP2vUspuQZH_kc7QCy-W0tCBZFNkkXkfR6KMovniLVAl2h7KK7)
45. [databricks.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFMLjyP1GuR_aPsJsc4yuWHEP_LL91W2qdRFV0zPIswlWZHKJLb5HeQcyJc9S04U8Ah9NrgiNvfcfXSErq23Qb3w-BrLeJEEIabxPMOdDhi0XNgAIy-xfe3Nsp7aFUywGvPdk9zeBJGlA67V_KAiCHy_tEuzQwTp6H4XcxvLUTS_mZWochkQo8ERgPacw==)
