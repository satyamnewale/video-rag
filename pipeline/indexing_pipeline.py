from alignment.temporal_alignment import TemporalAlignmentEngine
from alignment.multimodel_metadata_aggregator import (
    MultimodalMetadataAggregator
)

from records.structured_video_record_builder import (
    StructuredVideoRecordBuilder
)

from storage.frame_storage import FrameStorageProcessor
from storage.chrome_storage import ChromaStorage

from normalization.text_normalization import (
    TextNormalizationProcessor
)

from embeddings.embedding_processor import EmbeddingProcessor


class VideoIndexingPipeline:

    def __init__(self):

        self.alignment_engine = TemporalAlignmentEngine(
            window_size=2.0
        )

        self.metadata_aggregator = (
            MultimodalMetadataAggregator()
        )

        self.frame_storage = FrameStorageProcessor()

        self.record_builder = StructuredVideoRecordBuilder()

        self.text_normalizer = TextNormalizationProcessor()

        self.embedding_processor = EmbeddingProcessor(
            model="text-embedding-3-large",
            batch_size=32
        )

        self.vector_storage = ChromaStorage(
            persist_directory="storage_data/chroma",
            collection_name="videorag"
        )

    def index_video(
        self,
        video_id,
        scenes,
        audio_records,
        unique_frames,
        ocr_results,
        blip_results,
        yolo_results
    ):

        # 1. Temporal alignment

        aligned_windows = self.alignment_engine.align(
            scenes=scenes,
            audio_records=audio_records,
            ocr_results=ocr_results,
            blip_results=blip_results,
            yolo_results=yolo_results
        )

        # 2. Multimodal metadata aggregation

        aggregated_records = self.metadata_aggregator.aggregate(
            aligned_windows
        )

        # 3. Save unique frames

        frame_references = self.frame_storage.save_frames(
            unique_frames,
            video_id
        )

        # 4. Build structured video records

        structured_records = self.record_builder.build_records(
            aggregated_records=aggregated_records,
            frame_references=frame_references,
            video_id=video_id
        )

        # 5. Normalize records into text

        normalized_records = self.text_normalizer.normalize_records(
            structured_records
        )

        # 6. Generate embeddings

        embedded_records = self.embedding_processor.embed_records(
            normalized_records
        )

        # 7. Store in ChromaDB

        stored_count = self.vector_storage.store_records(
            embedded_records
        )

        return {

            "video_id": video_id,

            "aligned_windows": len(aligned_windows),

            "aggregated_records": len(aggregated_records),

            "structured_records": len(structured_records),

            "normalized_records": len(normalized_records),

            "embedded_records": len(embedded_records),

            "stored_records": stored_count,

            "frame_references": frame_references

        }