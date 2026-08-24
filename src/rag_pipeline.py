"""
Lightweight, deterministic retrieval layer for the clinical guidelines.

Uses TF-IDF + cosine similarity (appropriate for a small, curated corpus).
When a user condition is provided, condition-matched guidelines are
preferred and only similarity *ranks* within them — so a CKD user never
gets a generic sodium recommendation instead of the renal-specific one.
"""
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ClinicalRAGPipeline:
    """
    Deterministic vector retrieval over the clinical guideline corpus.
    """

    def __init__(self, guidelines_path="data/public/clinical_guidelines.json"):
        self.guidelines_path = guidelines_path
        self.guidelines = []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = None
        self._load_and_index()

    def _resolve_path(self):
        candidates = [
            self.guidelines_path,
            os.path.join(os.path.dirname(__file__), "..", "data", "public", "clinical_guidelines.json"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return self.guidelines_path

    def _load_and_index(self):
        path = self._resolve_path()
        if not os.path.exists(path):
            return
        with open(path, "r") as f:
            self.guidelines = json.load(f)

        corpus = [
            f"{g['condition']} {g['topic']} {g['recommendation']} {g['source']}"
            for g in self.guidelines
        ]
        if corpus:
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def search_guidelines(self, query, condition=None, top_k=2):
        """
        Returns the top_k most relevant guidelines.

        If ``condition`` matches any guideline, retrieval is restricted to
        that condition first (ranked by similarity); otherwise it falls
        back to the full corpus.
        """
        if not self.guidelines or self.tfidf_matrix is None:
            return []

        # Normalise condition matching: "Type 2 Diabetes" vs "Diabetes".
        def _norm(s):
            return str(s).lower().replace("type 2 ", "").strip()

        if condition:
            filtered = [
                (i, g) for i, g in enumerate(self.guidelines)
                if _norm(g["condition"]) in _norm(condition)
                or _norm(condition) in _norm(g["condition"])
            ]
        else:
            filtered = [(i, g) for i, g in enumerate(self.guidelines)]

        if filtered:
            indices = [i for i, _ in filtered]
        else:
            indices = list(range(len(self.guidelines)))

        sub_matrix = self.tfidf_matrix[indices]
        query_vec = self.vectorizer.transform([f"{condition or ''} {query}"])
        similarities = cosine_similarity(query_vec, sub_matrix).flatten()

        order = similarities.argsort()[::-1][:top_k]
        results = []
        for pos in order:
            idx = indices[pos]
            score = float(similarities[pos])
            g = dict(self.guidelines[idx])
            g["relevance_score"] = round(score, 3)
            results.append(g)
        return results

    def format_retrieved_context(self, query, condition=None, top_k=2):
        matches = self.search_guidelines(query, condition, top_k=top_k)
        if not matches:
            return "No specific medical guidelines retrieved."

        chunks = []
        for i, m in enumerate(matches, 1):
            chunks.append(
                f"[Clinical Guideline {i} - {m['condition']} ({m['source']})]\n"
                f"Topic: {m['topic']}\n"
                f"Recommendation: {m['recommendation']}"
            )
        return "\n\n".join(chunks)


# Global singleton instance for easy import
rag_engine = ClinicalRAGPipeline()
