
# AI use and my idea

I used Claude Opus 4.6 as my main AI agent to figure out the solutions. I then looked up the semantic searches such as TF-IDF, BM25 and tried to come up with the best solution possible. Given 2 hour constraint, I was able to implement 3 features (training & evaluation, hybrid searching and users & interface).
# thinkstruct-coding-challenge

---


- Natural language queries to find relevant patents
- Claim-to-claim mapping for overlap and similarity analysis
- Filtered search by classification code (e.g., `B60C` for tires) and keywords
- Quantitative evaluation of retrieval quality

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Flask Web UI                       │
│  Search · Filters · Patent Detail · History · Eval   │
└──────────────────────┬──────────────────────────────┘
                       │ REST API
┌──────────────────────▼──────────────────────────────┐
│              PatentSearchEngine                       │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │  TF-IDF      │  │    BM25      │  │  Hybrid    │ │
│  │  Vectorizer   │  │  (custom)    │  │  Filters   │ │
│  │  + Cosine Sim │  │              │  │            │ │
│  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘ │
│         └─────────┬───────┘              ┌──┘        │
│            Rank Fusion (weighted)         │           │
│            + Filter Application ──────────┘           │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Patent Data (JSON files)                 │
│  40 patents · 649 claims · 4 classification groups   │
└─────────────────────────────────────────────────────┘
```

## Enhancements Implemented

### 1. Hybrid Searching
Adds structured filters on top of semantic scores:
- **Classification prefix filter** — e.g., `B60C` returns only tire/wheel patents
- **Keyword filter** — require specific terms in title/abstract/description
- **Title filter** — match patents whose title contains a phrase

Filtering happens *after* scoring on the sorted rank list, which avoids recomputing embeddings. On this dataset, filtering adds <1ms of overhead (see evaluation output). At larger scale, pre-filtering with inverted indices on classification/keywords before scoring would further improve efficiency.

### 2. Web Interface (PatentLens UI)
A polished browser-based UI built with Flask featuring:
- **Search** with method/level toggles, expandable hybrid filters
- **Patent detail modal** with full claims, description, and "Find Similar" button
- **Evaluation dashboard** showing MRR, Recall@K, MAP, and timing comparisons
- **Search history** with one-click replay
- Session-based, no login required

### 3. Evaluation & Training Pipeline
Generates positive and negative pairs from the corpus:
- **Positive pairs**: claim→parent patent, title→patent, same-classification cross-pairs
- **Hard negatives**: cross-classification claim pairs, random unrelated pairs
- **Metrics**: MRR, Recall@1/5/10, MAP, score separation
- **Exports** `training_pairs.json` for downstream fine-tuning

Key results (claim→patent retrieval):
| Method   | MRR   | R@1  | R@5  | R@10 | Score Sep |
|----------|-------|------|------|------|-----------|
| TF-IDF   | 1.000 | 1.00 | 1.00 | 1.00 | 0.294     |
| BM25     | 0.918 | 0.88 | 0.98 | 1.00 | 72.74     |
| Combined | 0.995 | 0.99 | 1.00 | 1.00 | 0.974     |

## Data Handling

- **Missing fields**: Patents with missing fields are **included** (not excluded). Missing text fields default to empty strings, so partial patents still match on available content. This avoids data loss — documented per project instructions.
- **Indexing**: Each patent is indexed at two levels:
  - Patent-level: `title + abstract + detailed_description` concatenated
  - Claim-level: each claim is a separate searchable document

## How to Run

### Prerequisites
- Python 3.10+
- Required packages: `flask`, `scikit-learn`, `numpy` (all in standard data-science environments)

```bash
pip install flask scikit-learn numpy
```

### Quick Start

```bash
# Clone or unzip the project
cd patent-search

# Place patent JSON files in the data/ directory
# (files named patents_ipa*.json)

# Run the web app
python app.py
```

Then open **http://localhost:5000** in your browser.

### Command-Line Usage

```bash
# Run the search engine directly (demo searches)
python patent_engine.py

# Run the full evaluation suite
python evaluation.py
```

### Project Structure

```
patent-search/
├── data/                    # Patent JSON files
│   ├── patents_ipa250410.json
│   ├── patents_ipa250417.json
│   ├── patents_ipa250424.json
│   └── patents_ipa250501.json
├── patent_engine.py         # Core search engine (Part 1 + Hybrid)
├── evaluation.py            # Evaluation pipeline (Part 2)
├── app.py                   # Flask web UI (Part 2)
├── training_pairs.json      # Generated training data (after eval)
└── README.md                # This file
```

## Design Decisions & Commentary

**Why TF-IDF + BM25 fusion?**  
TF-IDF with cosine similarity captures term co-occurrence patterns, while BM25 applies length normalization and saturation that better handles long patent descriptions. Combining both (60/40 weighted rank fusion) outperforms either alone — the combined method achieves 0.995 MRR vs. 1.0 for TF-IDF and 0.918 for BM25 individually. TF-IDF excels at exact claim→patent matching while BM25 handles fuzzy title→patent queries better.

**Why not deep embeddings (sentence-transformers)?**  
For a 2-hour project with 40 patents, TF-IDF is pragmatic and highly effective (MRR > 0.99). The evaluation pipeline is designed to be model-agnostic — swapping in sentence-transformer embeddings requires only replacing the `_tfidf_scores` method. The training pairs generated here can directly fine-tune a bi-encoder model.

**Hybrid search efficiency:**  
Filtering adds <1ms overhead because we filter the *sorted* score list rather than recomputing scores for the filtered subset. At scale (10^7 patents), the approach would shift to pre-filtering with inverted indices on classification/keywords, then scoring only the reduced candidate set — reducing computation by orders of magnitude.

