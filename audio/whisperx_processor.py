import os

import whisperx
from dotenv import load_dotenv
from whisperx.diarize import DiarizationPipeline

load_dotenv()


class WhisperXProcessor:

    def __init__(self, enable_diarization=False):

        # Force CPU execution. WhisperX alignment uses this device.
        self.device = "cpu"
        self.enable_diarization = enable_diarization

        self.hf_token = os.getenv("HF_TOKEN")

        if self.enable_diarization and not self.hf_token:
            raise ValueError(
                "HF_TOKEN is required for speaker diarization"
            )

    def align_words(
        self,
        audio_path,
        transcription
    ):

        language = transcription["language"]

        # Load normalized audio. WhisperX will process it on CPU.
        audio = whisperx.load_audio(audio_path)

        # Convert faster-whisper output into WhisperX segment format.
        segments = []

        for item in transcription.get("segments", []):

            # Accept either the original faster-whisper keys
            # (start/end) or globally-offset keys
            # (start_seconds/end_seconds).
            start = item.get("start_seconds", item.get("start"))
            end = item.get("end_seconds", item.get("end"))

            if start is None or end is None:
                continue

            text = item.get("text", "").strip()

            if not text:
                continue

            segments.append({
                "start": start,
                "end": end,
                "text": text
            })

        if not segments:
            return {
                "language": language,
                "segments": []
            }

        align_model, metadata = whisperx.load_align_model(
            language_code=language,
            device=self.device
        )

        aligned_result = whisperx.align(
            segments,
            align_model,
            metadata,
            audio,
            self.device,
            return_char_alignments=False
        )

        # Optional speaker diarization; also runs on CPU.
        if self.enable_diarization:

            diarize_model = DiarizationPipeline(
                token=self.hf_token,
                device=self.device
            )

            diarize_segments = diarize_model(audio)

            aligned_result = whisperx.assign_word_speakers(
                diarize_segments,
                aligned_result
            )

        return aligned_result
