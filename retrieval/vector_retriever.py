from storage.chrome_storage import ChromaStorage
from embeddings.embedding_processor import EmbeddingProcessor


class VectorRetriever:

    def __init__(
        self,
        embedding_processor=None,
        chroma_storage=None,
        top_k=20
    ):

        self.embedding_processor = (
            embedding_processor or EmbeddingProcessor()
        )

        self.chroma_storage = (
            chroma_storage or ChromaStorage()
        )

        self.top_k = top_k

    # -----------------------------------------
    # Retrieve relevant video records
    # -----------------------------------------

    def retrieve(
        self,
        query,
        video_id=None,
        top_k=None
    ):

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        retrieval_k = top_k or self.top_k

        # Generate query embedding
        query_embedding = self.embedding_processor.embed_query(
            query
        )

        # Retrieve records from ChromaDB
        results = self.chroma_storage.search(
            query_embedding=query_embedding,
            top_k=retrieval_k,
            video_id=video_id
        )

        return results

    # -----------------------------------------
    # Collection information
    # -----------------------------------------

    def count_records(self):

        return self.chroma_storage.count_records()