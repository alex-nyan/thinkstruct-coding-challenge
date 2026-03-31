"""
Patent Search Engine — Core Module
===================================
Provides semantic (TF-IDF + cosine), BM25, and hybrid search over patent claims,
abstracts, and detailed descriptions.

Design decisions:
  - Patents with missing fields are INCLUDED. Missing text fields default to "".
    This avoids data loss and lets partial patents still match on available content.
  - Each patent is indexed at the *claim level* (one document per claim) AND at
    the *patent level* (abstract + title). This gives fine-grained claim mapping
    AND high-level patent retrieval.
"""

import json, os, re, math, time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Patent:
    doc_number: str
    title: str = ""
    abstract: str = ""
    detailed_description: List[str] = field(default_factory=list)
    claims: List[str] = field(default_factory=list)
    bibtex: str = ""
    classification: str = ""
    filename: str = ""
    filing_date: str = ""  # extracted from filename

    @staticmethod
    def from_dict(d: dict) -> "Patent":
        p = Patent(
            doc_number=str(d.get("doc_number", "")),
            title=d.get("title", ""),
            abstract=d.get("abstract", ""),
            detailed_description=d.get("detailed_description", []) or [],
            claims=d.get("claims", []) or [],
            bibtex=d.get("bibtex", ""),
            classification=d.get("classification", ""),
            filename=d.get("filename", ""),
        )
        # Try to extract date from filename like US20250135801A1-20250501.XML
        m = re.search(r"-(\d{8})\.", p.filename)
        if m:
            raw = m.group(1)
            p.filing_date = f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
        return p

    @property
    def full_description(self) -> str:
        return " ".join(self.detailed_description)

    @property
    def all_text(self) -> str:
        """Concatenation of all textual fields — used for patent-level indexing."""
        parts = [self.title, self.abstract, self.full_description]
        parts.extend(self.claims)
        return " ".join(parts)

    @property
    def classification_prefix(self) -> str:
        """Extract the alphanumeric classification prefix (e.g. B60B1502)."""
        m = re.match(r"([A-Z]\d+[A-Z]?\d*)", self.classification)
        return m.group(1) if m else self.classification


# ---------------------------------------------------------------------------
# BM25 implementation (from scratch, no external dep needed)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lowercase + split on non-alphanumeric."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """Okapi BM25 scorer."""

    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_tokens: List[List[str]] = [_tokenize(d) for d in corpus]
        self.doc_lens = np.array([len(t) for t in self.doc_tokens], dtype=np.float64)
        self.avgdl = self.doc_lens.mean() if self.corpus_size else 1.0

        # Build inverted index: term -> set of doc indices
        self.df: Dict[str, int] = {}
        for tokens in self.doc_tokens:
            seen = set()
            for t in tokens:
                if t not in seen:
                    self.df[t] = self.df.get(t, 0) + 1
                    seen.add(t)

        # Pre-compute IDF
        self.idf: Dict[str, float] = {}
        for term, df in self.df.items():
            self.idf[term] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str) -> np.ndarray:
        q_tokens = _tokenize(query)
        scores = np.zeros(self.corpus_size, dtype=np.float64)
        for qt in q_tokens:
            if qt not in self.idf:
                continue
            idf_val = self.idf[qt]
            for idx, doc_tokens in enumerate(self.doc_tokens):
                tf = doc_tokens.count(qt)
                if tf == 0:
                    continue
                dl = self.doc_lens[idx]
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[idx] += idf_val * num / den
        return scores


# ---------------------------------------------------------------------------
# Search index
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    patent: Patent
    score: float
    matched_claim_idx: Optional[int] = None   # if claim-level match
    matched_claim_text: Optional[str] = None
    match_level: str = "patent"  # "patent" or "claim"


class PatentSearchEngine:
    """
    Dual-level search engine:
      - Patent-level: searches over (title + abstract + description)
      - Claim-level:  searches over individual claims
    Supports TF-IDF cosine, BM25, and hybrid (filters + combined scoring).
    """

    def __init__(self):
        self.patents: List[Patent] = []
        self._patent_lookup: Dict[str, Patent] = {}

        # Patent-level index
        self._patent_corpus: List[str] = []
        self._patent_tfidf: Optional[TfidfVectorizer] = None
        self._patent_tfidf_matrix = None
        self._patent_bm25: Optional[BM25] = None

        # Claim-level index: (patent_idx, claim_idx, claim_text)
        self._claim_records: List[Tuple[int, int, str]] = []
        self._claim_corpus: List[str] = []
        self._claim_tfidf: Optional[TfidfVectorizer] = None
        self._claim_tfidf_matrix = None
        self._claim_bm25: Optional[BM25] = None

        self._built = False

    # ----- Data loading -----

    def load_json(self, path: str):
        with open(path) as f:
            data = json.load(f)
        for d in data:
            p = Patent.from_dict(d)
            self.patents.append(p)
            self._patent_lookup[p.doc_number] = p

    def load_directory(self, directory: str):
        for fp in sorted(Path(directory).glob("patents_ipa*.json")):
            self.load_json(str(fp))

    # ----- Index building -----

    def build_index(self):
        """Build TF-IDF and BM25 indices for both patent and claim levels."""
        t0 = time.perf_counter()

        # Patent-level corpus
        self._patent_corpus = [
            f"{p.title} {p.abstract} {p.full_description}" for p in self.patents
        ]
        self._patent_tfidf = TfidfVectorizer(
            max_features=20_000, stop_words="english", ngram_range=(1, 2),
            sublinear_tf=True, dtype=np.float32,
        )
        self._patent_tfidf_matrix = self._patent_tfidf.fit_transform(self._patent_corpus)
        self._patent_bm25 = BM25(self._patent_corpus)

        # Claim-level corpus
        self._claim_records = []
        self._claim_corpus = []
        for pi, p in enumerate(self.patents):
            for ci, claim in enumerate(p.claims):
                self._claim_records.append((pi, ci, claim))
                self._claim_corpus.append(claim)

        self._claim_tfidf = TfidfVectorizer(
            max_features=20_000, stop_words="english", ngram_range=(1, 2),
            sublinear_tf=True, dtype=np.float32,
        )
        self._claim_tfidf_matrix = self._claim_tfidf.fit_transform(self._claim_corpus)
        self._claim_bm25 = BM25(self._claim_corpus)

        self._built = True
        elapsed = time.perf_counter() - t0
        return {
            "patents_indexed": len(self.patents),
            "claims_indexed": len(self._claim_corpus),
            "build_time_sec": round(elapsed, 4),
        }

    # ----- Core search helpers -----

    def _tfidf_scores(self, query: str, level: str = "patent") -> np.ndarray:
        vec = self._patent_tfidf if level == "patent" else self._claim_tfidf
        mat = self._patent_tfidf_matrix if level == "patent" else self._claim_tfidf_matrix
        q_vec = vec.transform([query])
        return cosine_similarity(q_vec, mat).flatten()

    def _bm25_scores(self, query: str, level: str = "patent") -> np.ndarray:
        bm = self._patent_bm25 if level == "patent" else self._claim_bm25
        return bm.score(query)

    def _combined_scores(self, query: str, level: str = "patent",
                         tfidf_weight: float = 0.6, bm25_weight: float = 0.4) -> np.ndarray:
        """Rank-fusion of TF-IDF cosine and BM25 (both normalized to [0,1])."""
        tfidf = self._tfidf_scores(query, level)
        bm25 = self._bm25_scores(query, level)
        # Min-max normalise
        def norm(arr):
            mn, mx = arr.min(), arr.max()
            return (arr - mn) / (mx - mn + 1e-9)
        return tfidf_weight * norm(tfidf) + bm25_weight * norm(bm25)

    # ----- Hybrid filter helpers -----

    @staticmethod
    def _matches_classification(patent: Patent, prefix: str) -> bool:
        if not prefix:
            return True
        return patent.classification.upper().startswith(prefix.upper())

    @staticmethod
    def _matches_keywords(patent: Patent, keywords: List[str],
                          fields: List[str] = None) -> bool:
        if not keywords:
            return True
        fields = fields or ["title", "abstract"]
        text = ""
        for f in fields:
            if f == "title":
                text += " " + patent.title
            elif f == "abstract":
                text += " " + patent.abstract
            elif f == "description":
                text += " " + patent.full_description
        text_lower = text.lower()
        return all(kw.lower() in text_lower for kw in keywords)

    @staticmethod
    def _matches_title(patent: Patent, title_query: str) -> bool:
        if not title_query:
            return True
        return title_query.lower() in patent.title.lower()

    # ----- Public search API -----

    def search(
        self,
        query: str,
        level: str = "patent",           # "patent" or "claim"
        top_k: int = 10,
        method: str = "combined",         # "tfidf", "bm25", or "combined"
        # Hybrid filters (Part 2 enhancement)
        classification_prefix: str = "",
        keywords: List[str] = None,
        keyword_fields: List[str] = None,
        title_query: str = "",
    ) -> Tuple[List[SearchResult], Dict]:
        """
        Main search entry point.

        Returns (results, timing_info).
        """
        assert self._built, "Call build_index() first."
        keywords = keywords or []
        keyword_fields = keyword_fields or ["title", "abstract"]
        t0 = time.perf_counter()

        # --- Step 1: score every document ---
        if method == "tfidf":
            scores = self._tfidf_scores(query, level)
        elif method == "bm25":
            scores = self._bm25_scores(query, level)
        else:
            scores = self._combined_scores(query, level)

        t_score = time.perf_counter() - t0

        # --- Step 2: apply hybrid filters ---
        has_filters = bool(classification_prefix or keywords or title_query)
        t_filter_start = time.perf_counter()

        if level == "patent":
            results = []
            order = np.argsort(-scores)
            for idx in order:
                p = self.patents[int(idx)]
                if not self._matches_classification(p, classification_prefix):
                    continue
                if not self._matches_keywords(p, keywords, keyword_fields):
                    continue
                if not self._matches_title(p, title_query):
                    continue
                results.append(SearchResult(
                    patent=p, score=float(scores[idx]), match_level="patent"
                ))
                if len(results) >= top_k:
                    break
        else:  # claim-level
            results = []
            order = np.argsort(-scores)
            for idx in order:
                pi, ci, claim_text = self._claim_records[int(idx)]
                p = self.patents[pi]
                if not self._matches_classification(p, classification_prefix):
                    continue
                if not self._matches_keywords(p, keywords, keyword_fields):
                    continue
                if not self._matches_title(p, title_query):
                    continue
                results.append(SearchResult(
                    patent=p,
                    score=float(scores[idx]),
                    matched_claim_idx=ci,
                    matched_claim_text=claim_text,
                    match_level="claim",
                ))
                if len(results) >= top_k:
                    break

        t_filter = time.perf_counter() - t_filter_start

        timing = {
            "scoring_ms": round(t_score * 1000, 2),
            "filtering_ms": round(t_filter * 1000, 2),
            "total_ms": round((time.perf_counter() - t0) * 1000, 2),
            "method": method,
            "level": level,
            "filters_active": has_filters,
            "results_returned": len(results),
        }
        return results, timing

    def get_patent(self, doc_number: str) -> Optional[Patent]:
        return self._patent_lookup.get(str(doc_number))

    def find_similar_patents(self, doc_number: str, top_k: int = 5) -> Tuple[List[SearchResult], Dict]:
        """Given a patent ID, find the most similar patents."""
        p = self.get_patent(doc_number)
        if not p:
            return [], {"error": f"Patent {doc_number} not found"}
        query = f"{p.title} {p.abstract}"
        results, timing = self.search(query, level="patent", top_k=top_k + 1)
        # Remove self from results
        results = [r for r in results if r.patent.doc_number != doc_number][:top_k]
        timing["query_source"] = f"patent:{doc_number}"
        return results, timing

    def map_claim(self, claim_text: str, top_k: int = 10,
                  classification_prefix: str = "") -> Tuple[List[SearchResult], Dict]:
        """Map a claim to the most similar claims in the database."""
        return self.search(
            claim_text, level="claim", top_k=top_k,
            classification_prefix=classification_prefix,
        )

    # ----- Utility -----

    def get_all_classification_prefixes(self) -> List[str]:
        """Return sorted unique classification prefixes (e.g. B60B, B60C, ...)."""
        prefixes = set()
        for p in self.patents:
            # Take first 4 chars as broad prefix
            if len(p.classification) >= 4:
                prefixes.add(p.classification[:4])
        return sorted(prefixes)

    def stats(self) -> Dict:
        return {
            "total_patents": len(self.patents),
            "total_claims": sum(len(p.claims) for p in self.patents),
            "classification_groups": self.get_all_classification_prefixes(),
            "filing_dates": sorted(set(p.filing_date for p in self.patents if p.filing_date)),
        }


# ---------------------------------------------------------------------------
# Convenience: build engine from data directory
# ---------------------------------------------------------------------------

def create_engine(data_dir: str = "data") -> PatentSearchEngine:
    engine = PatentSearchEngine()
    engine.load_directory(data_dir)
    info = engine.build_index()
    print(f"Engine ready: {info['patents_indexed']} patents, "
          f"{info['claims_indexed']} claims indexed in {info['build_time_sec']}s")
    return engine


if __name__ == "__main__":
    engine = create_engine()
    print("\nStats:", json.dumps(engine.stats(), indent=2))

    # Quick demo search
    results, timing = engine.search("non-pneumatic tire", level="patent", top_k=5)
    print(f"\n--- Search: 'non-pneumatic tire' ({timing['total_ms']}ms) ---")
    for r in results:
        print(f"  [{r.score:.4f}] {r.patent.doc_number}: {r.patent.title}")

    # Claim-level search
    results, timing = engine.search("wheel assembly with metallic tread", level="claim", top_k=5)
    print(f"\n--- Claim search: 'wheel assembly with metallic tread' ({timing['total_ms']}ms) ---")
    for r in results:
        print(f"  [{r.score:.4f}] {r.patent.doc_number} claim {r.matched_claim_idx}: {r.matched_claim_text[:100]}...")

    # Hybrid search with classification filter
    results, timing = engine.search("tire noise reduction", level="patent", top_k=5,
                                     classification_prefix="B60C")
    print(f"\n--- Hybrid (B60C only): 'tire noise reduction' ({timing['total_ms']}ms) ---")
    for r in results:
        print(f"  [{r.score:.4f}] {r.patent.doc_number} [{r.patent.classification}]: {r.patent.title}")
