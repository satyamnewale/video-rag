import torch
import soundfile as sf

from silero_vad import (
    load_silero_vad,
    get_speech_timestamps
)


class VADProcessor:

    def __init__(
        self,
        sample_rate=16000,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100
    ):

        self.sample_rate = sample_rate
        self.threshold = threshold

        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms

        # Force Silero VAD model to CPU.
        self.device = torch.device("cpu")
        self.model = load_silero_vad()
        self.model.to(self.device)
        self.model.eval()

    def detect_speech(self, audio_path):

        # Load WAV directly without torchaudio or SoX
        audio_data, sample_rate = sf.read(
            audio_path,
            dtype="float32"
        )

        # Convert stereo to mono if necessary
        if audio_data.ndim == 2:
            audio_data = audio_data.mean(axis=1)

        # Ensure the audio has the expected sample rate
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"Expected {self.sample_rate} Hz audio, "
                f"but received {sample_rate} Hz"
            )

        # Convert NumPy array to PyTorch tensor
        audio = torch.from_numpy(audio_data).to(self.device)

        speech_timestamps = get_speech_timestamps(
            audio,
            self.model,
            sampling_rate=self.sample_rate,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            return_seconds=True
        )

        speech_segments = []

        for index, segment in enumerate(speech_timestamps):
            speech_segments.append({
                "speech_id": index + 1,
                "start_seconds": segment["start"],
                "end_seconds": segment["end"]
            })

        return speech_segments
