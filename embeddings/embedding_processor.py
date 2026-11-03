from langchain_openai import OpenAIEmbeddings


class EmbeddingProcessor:

    def __init__(
        self,
        model="text-embedding-3-large",
        batch_size=32,
        dimensions=None
    ):

        self.model = model
        self.batch_size = batch_size
        self.dimensions = dimensions

        # Initialize LangChain OpenAI Embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.model,
            dimensions=self.dimensions,
            chunk_size=self.batch_size
        )

    # -----------------------------------------
    # Generate embeddings for a batch of texts
    # -----------------------------------------

    def embed_batch(self, texts):

        if not texts:
            return []

        for text in texts:

            if not text or not text.strip():
                raise ValueError(
                    "Cannot generate embedding for empty text"
                )

        vectors = self.embeddings.embed_documents(texts)

        return vectors

    # -----------------------------------------
    # Generate embeddings for all records
    # -----------------------------------------

    def embed_records(self, normalized_records):

        embedded_records = []

        for start in range(
            0,
            len(normalized_records),
            self.batch_size
        ):

            batch = normalized_records[
                start:start + self.batch_size
            ]

            texts = [
                record["text"]
                for record in batch
            ]

            vectors = self.embed_batch(texts)

            for record, vector in zip(batch, vectors):

                embedded_records.append({

                    "record_id": record["record_id"],

                    "video_id": record["video_id"],

                    "scene_id": record["scene_id"],

                    "start_seconds": record["start_seconds"],

                    "end_seconds": record["end_seconds"],

                    "text": record["text"],

                    "embedding": vector,

                    "metadata_source": record["metadata_source"]

                })

        return embedded_records

    # -----------------------------------------
    # Generate embedding for user query
    # -----------------------------------------

    def embed_query(self, query):

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        return self.embeddings.embed_query(query)