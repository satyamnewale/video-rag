import json
import chromadb


class ChromaStorage:

    def __init__(
        self,
        persist_directory="storage_data/chroma",
        collection_name="videorag"
    ):

        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine"
            }
        )

    # --------------------------------
    # Prepare Chroma metadata
    # --------------------------------

    def _prepare_metadata(self, record):

        source = record["metadata_source"]

        return {

            "video_id": record["video_id"],

            "scene_id": int(record["scene_id"]),

            "start_seconds": float(record["start_seconds"]),

            "end_seconds": float(record["end_seconds"]),

            "frame_references": json.dumps(
                source.get("frame_references", [])
            ),

            "speakers": json.dumps(
                source.get("speakers", [])
            ),

            "ocr": json.dumps(
                source.get("ocr", [])
            ),

            "visual_descriptions": json.dumps(
                source.get("visual_descriptions", [])
            ),

            "detected_objects": json.dumps(
                source.get("detected_objects", [])
            )

        }

    # --------------------------------
    # Store embedded records
    # --------------------------------

    def store_records(self, embedded_records):

        if not embedded_records:
            return 0

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for record in embedded_records:

            ids.append(record["record_id"])

            documents.append(record["text"])

            embeddings.append(record["embedding"])

            metadatas.append(
                self._prepare_metadata(record)
            )

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        return len(ids)

    # --------------------------------
    # Similarity search
    # --------------------------------

    def search(
        self,
        query_embedding,
        top_k=5,
        video_id=None
    ):

        where = None

        if not query_embedding:
            return []

        if video_id:

            where = {
                "video_id": video_id
            }

        collection_count = self.collection.count()

        if collection_count == 0:
            return []

        top_k = min(top_k, collection_count)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        output = []

        for index in range(
            len(results["ids"][0])
        ):

            output.append({

                "record_id": results["ids"][0][index],

                "text": results["documents"][0][index],

                "metadata": results["metadatas"][0][index],

                "distance": results["distances"][0][index]

            })

        return output

    # --------------------------------
    # Collection information
    # --------------------------------

    def count_records(self):

        return self.collection.count()