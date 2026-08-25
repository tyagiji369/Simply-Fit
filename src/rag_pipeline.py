import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ClinicalRAGPipeline:
    """
    Lightweight, deterministic retrieval layer over a small curated set of
    evidence-based medical nutrition therapy guidelines (7 entries in
    data/public/clinical_guidelines.json).

    Retrieval uses TF-IDF term vectors and cosine similarity — deliberately
    simple: with a corpus this small, TF-IDF retrieval is fast, fully
    reproducible and needs no embeddings model or vector database. Retrieved
    guidelines are injected into the coach's prompt/context to ground its
    answers ("retrieval-augmented generation" in the plain sense).
    """

    def __init__(self, guidelines_path="data/public/clinical_guidelines.json"):
        self.guidelines_path = guidelines_path
        self.guidelines = []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = None
        self._load_and_index()

    def _load_and_index(self):
        if not os.path.exists(self.guidelines_path):
            # Fallback path if run from root vs app dir
            alt_path = os.path.join(
                os.path.dirname(__file__), "..", "data", "public", "clinical_guidelines.json"
            )
            if os.path.exists(alt_path):
                self.guidelines_path = alt_path

        if os.path.exists(self.guidelines_path):
            with open(self.guidelines_path, "r") as f:
                self.guidelines = json.load(f)

            # Build text corpus for embedding vectorization
            corpus = [
                f"{g['condition']} {g['topic']} {g['recommendation']} {g['source']}"
                for g in self.guidelines
            ]
            if corpus:
                self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def search_guidelines(self, query, condition=None, top_k=2):
        """
        Retrieves top_k clinical guidelines most relevant to query & condition.
        Returns list of matched guideline dicts with similarity scores.
        """
        if not self.guidelines or self.tfidf_matrix is None:
            return []

        # Enhance query with condition if provided
        search_text = f"{condition or ''} {query}"
        query_vec = self.vectorizer.transform([search_text])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top-k indices
        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            g = self.guidelines[idx].copy()
            g["relevance_score"] = round(score, 3)
            results.append(g)

        return results

    def format_retrieved_context(self, query, condition=None, top_k=2):
        """
        Formats retrieved guidelines into a clean context string for LLM grounding.
        """
        matches = self.search_guidelines(query, condition, top_k=top_k)
        if not matches:
            return "No specific medical guidelines retrieved."

        formatted_chunks = []
        for i, m in enumerate(matches, 1):
            formatted_chunks.append(
                f"[Clinical Guideline {i} - {m['condition']} ({m['source']})]\n"
                f"Topic: {m['topic']}\n"
                f"Recommendation: {m['recommendation']}"
            )

        return "\n\n".join(formatted_chunks)


# Global singleton instance for easy import
rag_engine = ClinicalRAGPipeline()
