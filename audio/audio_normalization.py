
import ffmpeg
from pathlib import Path


class AudioNormalizationProcessor:

    def __init__(
        self,
        sample_rate=16000,
        channels=1
    ):

        self.sample_rate = sample_rate
        self.channels = channels

    def normalize_audio(
        self,
        input_path,
        output_path
    ):

        if not Path(input_path).exists():
            raise FileNotFoundError(input_path)

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        try:

            probe = ffmpeg.probe(input_path)

            audio_stream = next(
                (
                    stream for stream in probe["streams"]
                    if stream.get("codec_type") == "audio"
                ),
                None
            )

            if audio_stream is None:
                raise ValueError("No audio stream found")

            (
                ffmpeg
                .input(input_path)
                .output(
                    str(output_path),
                    acodec="pcm_s16le",
                    ac=self.channels,
                    ar=self.sample_rate,
                    vn=None
                )
                .overwrite_output()
                .run(
                    capture_stdout=True,
                    capture_stderr=True
                )
            )

            output_probe = ffmpeg.probe(
                str(output_path)
            )

            output_audio = next(
                stream for stream in output_probe["streams"]
                if stream.get("codec_type") == "audio"
            )

            if int(output_audio["sample_rate"]) != self.sample_rate:
                raise ValueError("Incorrect sample rate")

            if int(output_audio["channels"]) != self.channels:
                raise ValueError("Incorrect channel count")

            duration = float(
                output_probe["format"].get("duration", 0)
            )

            if duration <= 0:
                raise ValueError("Audio contains no duration")

            return {
                "audio_path": str(output_path),
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "duration": duration,
                "codec": output_audio.get("codec_name")
            }

        except ffmpeg.Error as error:

            stderr = error.stderr

            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")

            raise RuntimeError(
                stderr or "Audio normalization failed"
            ) from error
