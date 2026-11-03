from pathlib import Path
from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_path=r"D:\model\bge-reranker-base",
        top_n=5,
        device="cpu"
    ):

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Reranker model not found: {model_path}"
            )

        self.top_n = top_n

        self.ranker = CrossEncoder(
            str(model_path),
            device=device
        )

    def rerank(self, query, documents, top_n=None):

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not documents:
            return []

        pairs = [
            [query, document["text"]]
            for document in documents
        ]

        scores = self.ranker.predict(
            pairs,
            show_progress_bar=False
        )

        ranked = []

        for document, score in zip(documents, scores):
            result = document.copy()
            result["rerank_score"] = float(score)
            ranked.append(result)

        ranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True
        )

        return ranked[:(top_n or self.top_n)]