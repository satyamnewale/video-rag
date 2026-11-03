
import os
from pathlib import Path

# Video processors
from video.scene_detection import SceneDetectionProcessor
from video.frame_extraction import FrameExtractionProcessor
from video.frame_selection import FrameSelectionEngine
from video.perceptual_hashing import PerceptualHashProcessor
from video.ocr_processor import OCRProcessor
from video.blip_processor import BLIPProcessor
from video.yolo_processor import YOLOProcessor

# Audio processors
from audio.audio_extracton import AudioExtractionProcessor
from audio.audio_normalization import AudioNormalizationProcessor
from audio.vad_processor import VADProcessor
from audio.transcription_processor import TranscriptionProcessor
from audio.whisperx_processor import WhisperXProcessor
from audio.audio_record_builder import StructuredAudioRecordBuilder

from silero_vad import read_audio

# Correct folder name: pipeline, not pipelines
from pipeline.indexing_pipeline import VideoIndexingPipeline


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

# CHANGE ONLY THE VIDEO FILENAME HERE
VIDEO_PATH = BASE_DIR / "resources" / "anime.mp4"

VIDEO_ID = VIDEO_PATH.stem

TEMP_DIR = BASE_DIR / "storage_data" / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

EXTRACTED_AUDIO = TEMP_DIR / f"{VIDEO_ID}_extracted.wav"
NORMALIZED_AUDIO = TEMP_DIR / f"{VIDEO_ID}_normalized.wav"


# Keep relative storage paths in other classes
# anchored to the project root.
os.chdir(BASE_DIR)


# ==========================================
# AUDIO PIPELINE
# ==========================================

def process_audio(video_path):

    print("\n========== AUDIO PIPELINE ==========")

    # 1. Extract audio
    extractor = AudioExtractionProcessor()

    try:
        extractor.extract_audio(
            video_path=str(video_path),
            output_path=str(EXTRACTED_AUDIO)
        )
    except ValueError as error:
        if "does not contain an audio stream" in str(error):
            print("Video has no audio stream. Skipping audio.")
            return []
        raise

    print("Audio extracted")

    # 2. Normalize audio
    normalizer = AudioNormalizationProcessor(
        sample_rate=16000,
        channels=1
    )

    normalizer.normalize_audio(
        input_path=str(EXTRACTED_AUDIO),
        output_path=str(NORMALIZED_AUDIO)
    )

    print("Audio normalized")

    # 3. Speech detection
    vad = VADProcessor()

    speech_segments = vad.detect_speech(
        str(NORMALIZED_AUDIO)
    )

    print("Speech segments:", len(speech_segments))

    if not speech_segments:
        return []

    # 4. Load normalized audio
    audio = read_audio(
        str(NORMALIZED_AUDIO),
        sampling_rate=16000
    )

    audio_array = audio.numpy()

    # 5. Transcribe speech intervals
    transcriber = TranscriptionProcessor(
        model_size="small"
    )

    transcription_segments = []
    detected_language = None

    for speech in speech_segments:

        start = speech["start_seconds"]
        end = speech["end_seconds"]

        start_sample = int(start * 16000)
        end_sample = int(end * 16000)

        audio_chunk = audio_array[
            start_sample:end_sample
        ].astype("float32")

        if len(audio_chunk) == 0:
            continue

        transcription = transcriber.transcribe(
            audio_chunk
        )

        if detected_language is None:
            detected_language = transcription["language"]

        # Convert crop-relative timestamps
        # into original video-relative timestamps.
        for segment in transcription["segments"]:

            segment["start"] += start
            segment["end"] += start

            for word in segment["words"]:

                if word["start"] is not None:
                    word["start"] += start

                if word["end"] is not None:
                    word["end"] += start

            transcription_segments.append(segment)

    if not transcription_segments:
        return []

    # 6. WhisperX alignment
    transcription_data = {
        "language": detected_language,
        "segments": transcription_segments
    }

    whisperx_processor = WhisperXProcessor(
        enable_diarization=False
    )

    aligned_result = whisperx_processor.align_words(
        audio_path=str(NORMALIZED_AUDIO),
        transcription=transcription_data
    )

    print("WhisperX alignment completed")

    # 7. Build structured audio records
    audio_builder = StructuredAudioRecordBuilder()

    audio_records = audio_builder.build_records(
        aligned_result=aligned_result,
        video_id=VIDEO_ID
    )

    print("Structured audio records:", len(audio_records))

    return audio_records


# ==========================================
# VIDEO / VISION PIPELINE
# ==========================================

def process_video(video_path):

    print("\n========== VIDEO PIPELINE ==========")

    # 1. Scene detection
    scene_processor = SceneDetectionProcessor()

    scenes = scene_processor.detect_scenes(
        str(video_path)
    )

    print("Scenes detected:", len(scenes))

    if not scenes:
        raise ValueError("No scenes detected in video")

    # 2. Frame extraction
    frame_extractor = FrameExtractionProcessor(
        sample_interval=1.0,
        target_size=(640, 640)
    )

    candidate_frames = frame_extractor.extract_frames(
        str(video_path),
        scenes
    )

    print("Candidate frames:", len(candidate_frames))

    # 3. Frame selection
    frame_selector = FrameSelectionEngine(
        max_frames_per_scene=5,
        temporal_bins=5
    )

    selected_frames = frame_selector.select_keyframes(
        candidate_frames,
        scenes
    )

    print("Selected frames:", len(selected_frames))

    # 4. Perceptual hashing
    hash_processor = PerceptualHashProcessor(
        hash_size=8,
        threshold=8
    )

    unique_frames = hash_processor.remove_duplicates(
        selected_frames,
        video_id=VIDEO_ID
    )

    print("Unique frames:", len(unique_frames))

    if not unique_frames:
        raise ValueError("No unique frames were extracted")

    # 5. OCR
    ocr_processor = OCRProcessor(lang="en")

    ocr_results = ocr_processor.process_frames(
        unique_frames
    )

    print("OCR completed")

    # 6. BLIP captions
    # BLIP class already forces CPU internally.
    blip_processor = BLIPProcessor()

    blip_results = blip_processor.process_frames(
        unique_frames
    )

    print("BLIP completed")

    # 7. YOLO object detection
    # YOLO class already forces CPU internally.
    yolo_processor = YOLOProcessor()

    yolo_results = yolo_processor.process_frames(
        unique_frames
    )

    print("YOLO completed")

    return {
        "scenes": scenes,
        "unique_frames": unique_frames,
        "ocr_results": ocr_results,
        "blip_results": blip_results,
        "yolo_results": yolo_results
    }


# ==========================================
# MAIN VIDEO RAG INDEXING
# ==========================================

def main():

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}\n"
            "Check the filename inside resources/videos/"
        )

    print("Video:", VIDEO_PATH)
    print("Video ID:", VIDEO_ID)
    print("PyTorch device: CPU")

    # 1. Process video
    video_data = process_video(VIDEO_PATH)

    # 2. Process audio
    audio_records = process_audio(VIDEO_PATH)

    # 3. Temporal alignment through ChromaDB indexing
    indexing_pipeline = VideoIndexingPipeline()

    result = indexing_pipeline.index_video(

        video_id=VIDEO_ID,

        scenes=video_data["scenes"],

        audio_records=audio_records,

        unique_frames=video_data["unique_frames"],

        ocr_results=video_data["ocr_results"],

        blip_results=video_data["blip_results"],

        yolo_results=video_data["yolo_results"]
    )

    print("\n========== INDEXING COMPLETED ==========")

    for key, value in result.items():

        if key != "frame_references":
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
