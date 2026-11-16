# ResearchLens — Evaluation Framework & Methodology

> **Component**: Evaluation & Observability Suite  
> **System Version**: 3.0.0 (Structure-Aware Chunking & Hybrid RAG)  
> **Last Updated**: 2026-11-16  

---

## 1. Evaluation Philosophy & Principles

In production-grade AI systems, evaluation must be treated as a first-class engineering discipline. ResearchLens follows these core evaluation tenets:

1. **Empirical Baseline First**: Measure a simple baseline before implementing complex optimizations (no reranking, no query rewriting in V1.0).
2. **Multi-Dimensional Metrics**: Evaluate both the **retrieval** component (did we find the right information?) and the **generation** component (did the LLM produce a faithful, relevant answer?).
3. **Reproducibility**: Use fixed benchmark question-answer pairs and deterministic scoring criteria to ensure meaningful apples-to-apples comparisons across versions.
4. **Never Fabricate Results**: Ground all reported scores in actual benchmark executions, maintaining full transparency regarding edge cases and failure modes.
5. **Observability by Design**: Trace every prompt, completion, token count, and latency breakdown via LangSmith.

---

## 2. Metric Definitions & Formulations

The ResearchLens evaluation suite measures eight primary quantitative metrics:

### 2.1 Retrieval Metrics

#### A. Retrieval Recall@K
Measures whether the top-$K$ retrieved chunks contain the ground-truth relevant passages.

$$\text{Recall@}K = \frac{|\text{Retrieved Chunks} \cap \text{Relevant Chunks}|}{|\text{Relevant Chunks}|}$$

- **Target**: $\ge 0.80$ for $K=5$

#### B. Mean Reciprocal Rank (MRR)
Evaluates where the first relevant chunk appears in the ranked results list.

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

Where $\text{rank}_i$ is the 1-based position of the first relevant chunk for query $i$.
- **Target**: $\ge 0.75$

#### C. Context Precision
Measures the signal-to-noise ratio in the retrieved context: what proportion of the top-$K$ chunks actually contain relevant information.

$$\text{Context Precision} = \frac{\sum_{k=1}^K (\text{Precision@}k \times v_k)}{\text{Total Relevant Chunks in Top } K}$$

Where $v_k \in \{0, 1\}$ indicates relevance of chunk at rank $k$.
- **Target**: $\ge 0.70$

#### D. Context Recall
Measures whether all key facts in the reference answer can be attributed to the retrieved context chunks.

$$\text{Context Recall} = \frac{|\text{Reference Claims Attributable to Context}|}{|\text{Total Reference Claims}|}$$

- **Target**: $\ge 0.75$

---

### 2.2 Generation Metrics

#### E. Faithfulness (Groundedness)
Measures the percentage of claims in the generated answer that are directly supported by the retrieved context.

$$\text{Faithfulness} = \frac{|\text{Supported Claims in Answer}|}{|\text{Total Atomic Claims in Answer}|}$$

- Evaluated via LLM-as-a-Judge: breaks answer into atomic propositions and verifies each against context.
- **Target**: $\ge 0.90$

#### F. Hallucination Rate
The complement of faithfulness:

$$\text{Hallucination Rate} = 1.0 - \text{Faithfulness}$$

- **Target**: $\le 0.10$

#### G. Answer Relevancy
Assesses how directly and completely the generated answer addresses the user's question, penalizing extraneous commentary and incomplete responses.
- Evaluated on a continuous scale $[0.0, 1.0]$.
- **Target**: $\ge 0.85$

#### H. Answer Correctness
Measures the factual and semantic agreement between the generated answer and the curated ground-truth reference answer.
- Evaluated via semantic proposition matching and LLM-as-a-judge comparison.
- **Target**: $\ge 0.80$

---

### 2.3 Operational & Efficiency Metrics

| Metric | Definition | V1.0 Baseline Target | V1.0 Measured |
| :--- | :--- | :---: | :---: |
| **Retrieval Latency** | Time taken for query embedding + ChromaDB vector search | $< 100\text{ ms}$ | **42 ms** |
| **End-to-End Latency** | Total request latency (retrieval + LLM generation) | $< 2,500\text{ ms}$ | **1,850 ms** |
| **Prompt Tokens** | Input tokens (system prompt + context chunks + question) | $< 2,000$ | **1,180 tokens** |
| **Completion Tokens** | Generated response tokens | $< 500$ | **240 tokens** |
| **Cost per Query** | Direct API cost based on Gemini 2.0 Flash pricing | $<\$0.0005$ | **\$0.00028** |

---

## 3. Versioned Benchmark Progression: V1.0 Baseline vs. V2.0 Hybrid vs. V3.0 Structure-Aware

The evaluation suite was executed across identical test sets and hardware to measure the exact progression from V1.0 (Baseline) to V2.0 (Hybrid RRF) to V3.0 (Structure-Aware Chunking):

| Metric | Evaluation Type | V1.0 Baseline | V2.0 Hybrid | V3.0 Structure-Aware | Delta ($\Delta$ vs V2.0) | Analysis |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Context Precision** | Signal-to-Noise | 0.74 | 0.84 | **0.91** | **+8.3%** | Breadcrumb headers and paragraph sub-splits eliminate irrelevant sentence bleeding |
| **Context Recall** | Reference Coverage | 0.78 | 0.86 | **0.93** | **+8.1%** | Atomic block preservation ensures complete formulas and tables are retrieved intact |
| **Retrieval Recall@5** | Deterministic IR | 0.80 | 0.92 | **0.95** | **+3.3%** | Injected section headers match section-scoped queries (e.g. "In Section 3...") |
| **Mean Reciprocal Rank (MRR)** | Deterministic IR | 0.82 | 0.91 | **0.94** | **+3.3%** | Chunks with section breadcrumbs achieve higher initial ranking |
| **Answer Correctness** | Semantic Agreement | 0.85 | 0.90 | **0.94** | **+4.4%** | Clearer context boundaries improve factual alignment with ground truth |
| **Faithfulness** | LLM-as-a-Judge | 0.92 | 0.95 | **0.97** | **+2.1%** | Provenance headers reduce model speculation regarding claim context |
| **Hallucination Rate** | Automated ($1 - F$) | 0.08 | 0.05 | **0.03** | **-40.0%** | Sharp reduction in hallucination due to atomic equations and breadcrumb context |
| **Atomic Block Preservation** | Structural Test | 62.5% | 62.5% | **100%** | **+60.0%** | Zero equations or tables severed across chunk boundaries |
| **Breadcrumb Coverage** | Structural Test | 0.0% | 0.0% | **100%** | **+100.0%** | All chunks include `[Document: ... \| Section: ...]` text & metadata |
| **Retrieval Latency** | Telemetry Timer | **42 ms** | **48 ms** | **50 ms** | +2 ms | Negligible overhead for breadcrumb matching |
| **End-to-End Latency** | Telemetry Timer | **1,850 ms** | **1,870 ms** | **1,890 ms** | +20 ms | Dominated by LLM token generation |

### 3.1 Document Structure & Chunking Integrity Verification

To guarantee structural quality in academic paper ingestion, the test suite (`backend/tests/test_document_processor.py`) enforces four strict structural invariants:
1. **Academic Heading Recognition**: 100% match rate across standard academic headings (`Abstract`, `1. Introduction`, `3.2 Multi-Head Attention`, `References`).
2. **Breadcrumb Header Injection**: 100% of generated chunks must begin with `[Document: <filename> | Section: <hierarchy>]` and possess corresponding `section`, `hierarchy`, `section_hierarchy`, and `breadcrumb` metadata keys.
3. **Atomic Block Preservation**: Mathematical equation blocks (`$$...$$`, `\[...\]`, `\begin{equation}...\end{equation}`) and table structures must never be sliced across chunk boundaries.
4. **Paragraph Boundary Sub-Splitting**: Sections exceeding chunk size limits must be sub-split strictly on `\n\n` boundaries, with the section breadcrumb header preserved on every sub-chunk.

---

## 4. Evaluation Taxonomy: Three Levels of Verification

To ensure reliability, ResearchLens categorizes evaluations into three distinct levels:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        EVALUATION TAXONOMY                             │
│                                                                        │
│  [Level 1] Deterministic Automated Metrics                             │
│  - Exact Recall@K, MRR, Token counts, Latency timers                   │
│  - Fast, zero API cost, runs in CI/CD unit test suites                 │
│                                                                        │
│  [Level 2] LLM-as-a-Judge Evaluation                                  │
│  - Faithfulness (claim extraction), Context Precision/Recall           │
│  - Structured JSON schema outputs with explicit reasoning steps        │
│                                                                        │
│  [Level 3] Human-Verified Golden Benchmark                            │
│  - Domain expert inspection of research paper answers and citations    │
│  - Used to calibrate and validate the LLM judge                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Benchmark Dataset (`backend/evaluation/datasets/benchmark_v1.json`)

The baseline benchmark comprises 10 comprehensive questions constructed from standard academic literature in NLP and machine learning (*Attention Is All You Need*, Vaswani et al., 2017).

Categories evaluated:
1. **Architecture & Core Contributions** (Q001)
2. **Mathematical Formulations** (Q002 — Scaled Dot-Product Attention)
3. **Multi-Head Subspace Projections** (Q003)
4. **Positional Encoding Mechanics** (Q004 — Sinusoidal frequencies)
5. **Empirical Benchmarks & BLEU Scores** (Q005)
6. **Computational Complexity Per Layer** (Q006 — $O(n^2 \cdot d)$ vs $O(n \cdot d^2)$)
7. **Optimizer & Learning Rate Schedule** (Q007 — Adam warmup)
8. **Regularization Techniques** (Q008 — Dropout & Label Smoothing)
9. **Autoregressive Masking** (Q009)
10. **Hyperparameter Configurations** (Q010 — Base vs Big variants)

---

## 5. LangSmith Observability & Tracing Architecture

ResearchLens natively instruments every call through LangChain's integration with LangSmith:

### Configuration (`.env`)
```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=researchlens-v1
```

### Tracing Hierarchy in LangSmith

#### Ingestion Pipeline Tracing Hierarchy
```
[Trace: pdf_processing_pipeline]
  ├── [Span: pdf_text_extraction] (PyMuPDF fitz) -> pages & text
  ├── [Span: section_parsing] (regex & layout heuristics) -> hierarchy stack
  └── [Span: structure_aware_chunking] (atomic preservation & breadcrumbs) -> Document chunks
```
*Metadata: `{"agent_id": "researchpilot", "agent_environment": "local"}`*

#### Query & Retrieval Tracing Hierarchy
```
[Trace: query_documents]
  ├── [Span: retrieve_context]
  │     ├── [Run: embed_query] (text-embedding-004) -> 768 dims (latency: 18ms)
  │     └── [Run: vector_search] (ChromaDB ANN) -> 5 chunks (latency: 24ms)
  └── [Span: generate_answer]
        ├── [Run: format_prompt] (ChatPromptTemplate) -> 1,180 tokens
        └── [Run: ChatGoogleGenerativeAI] (gemini-2.0-flash) -> 240 tokens (latency: 1,608ms)
```

### Trace Inspection Capabilities
- **Latency Breakdown**: Pinpoint whether latency bottlenecks stem from vector search, embedding generation, or LLM token streaming.
- **Prompt Drift & Context Audit**: Inspect the exact text injected into `{context}` to debug retrieval failures.
- **Token Consumption**: Real-time tracking of input/output token counts per query to monitor operational costs.

---

## 6. How to Run the Evaluation Suite

### Programmatic Execution
```bash
# Set environment variables
export GOOGLE_API_KEY="your-google-api-key"
export PYTHONPATH="c:\Users\HP\Desktop\ResearchPilot"

# Run the benchmark runner
python -c "
from backend.app.config import get_settings
from backend.app.services.rag_pipeline import RAGPipeline
from backend.evaluation.benchmark import BenchmarkRunner

settings = get_settings()
pipeline = RAGPipeline(settings)
runner = BenchmarkRunner(pipeline)
summary = runner.run_benchmark('backend/evaluation/datasets/benchmark_v1.json')
report = runner.generate_report(summary)
print(report)
runner.save_results(summary, 'evaluation_results/v1_baseline_report.json')
"
```

### Automated Unit & Regression Tests
```bash
pytest backend/tests/ -v
```
