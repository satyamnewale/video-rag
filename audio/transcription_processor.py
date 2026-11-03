from faster_whisper import WhisperModel


class TranscriptionProcessor:

    def __init__(self, model_size: str = "small"):
        # Force CPU execution; use INT8 to reduce CPU memory usage.
        self.device = "cpu"
        self.compute_type = "int8"

        self.model = WhisperModel(
            model_size,
            device=self.device,
            compute_type=self.compute_type
        )

    def transcribe(self, audio_chunk, language=None):

        segments, info = self.model.transcribe(
            audio_chunk,
            language=language,
            beam_size=5,
            vad_filter=False,
            word_timestamps=True,
            condition_on_previous_text=False
        )

        results = []

        for segment in segments:
            results.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "words": [
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability
                    }
                    for word in (segment.words or [])
                ]
            })

        return {
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": results
        }
