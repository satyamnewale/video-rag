
import ffmpeg
from pathlib import Path


class AudioExtractionProcessor:

    def extract_audio(
        self,
        video_path,
        output_path
    ):

        if not Path(video_path).exists():
            raise FileNotFoundError(video_path)

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        try:
            probe = ffmpeg.probe(video_path)

            has_audio = any(
                stream.get("codec_type") == "audio"
                for stream in probe.get("streams", [])
            )

            if not has_audio:
                raise ValueError(
                    "Video does not contain an audio stream"
                )

            (
                ffmpeg
                .input(video_path)
                .output(
                    str(output_path),
                    vn=None,
                    acodec="pcm_s16le",
                    ac=1,
                    ar=16000
                )
                .overwrite_output()
                .run(
                    capture_stdout=True,
                    capture_stderr=True
                )
            )

            return str(output_path)

        except ffmpeg.Error as error:

            stderr = error.stderr

            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")

            raise RuntimeError(
                stderr or "Audio extraction failed"
            ) from error
