
import ffmpeg
import os


class FFmpegProcessor:

    # --------------------------------
    # Check whether video has audio
    # --------------------------------
    def _has_audio(self, input_path):

        probe = ffmpeg.probe(input_path)

        return any(
            stream.get("codec_type") == "audio"
            for stream in probe.get("streams", [])
        )

    # --------------------------------
    # Validate video
    # --------------------------------
    def validate_video(self, video_path):

        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)

        try:
            ffmpeg.probe(video_path)
            return True

        except ffmpeg.Error:
            return False

    # --------------------------------
    # Normalize video
    # --------------------------------
    def normalize_video(
        self,
        input_path,
        output_path,
        width=1200,
        fps=None
    ):

        stream = ffmpeg.input(input_path)

        video = stream.video.filter(
            "scale",
            width,
            -2
        )

        if fps is not None:
            video = video.filter("fps", fps=fps)

        output_streams = [video]

        # Include audio only when available
        if self._has_audio(input_path):
            output_streams.append(stream.audio)

        output = ffmpeg.output(
            *output_streams,
            output_path,
            vcodec="libx264",
            pix_fmt="yuv420p",
            preset="medium",
            crf=23,
            **(
                {"acodec": "aac"}
                if len(output_streams) > 1 else {}
            )
        )

        output.overwrite_output().run(
            capture_stdout=True,
            capture_stderr=True
        )

        return output_path

    # --------------------------------
    # Convert video
    # --------------------------------
    def convert_video(
        self,
        input_path,
        output_path
    ):

        stream = ffmpeg.input(input_path)

        output_streams = [stream.video]

        if self._has_audio(input_path):
            output_streams.append(stream.audio)

        (
            ffmpeg
            .output(
                *output_streams,
                output_path,
                vcodec="libx264",
                **(
                    {"acodec": "aac"}
                    if len(output_streams) > 1 else {}
                )
            )
            .overwrite_output()
            .run(capture_stderr=True)
        )

        return output_path

    # --------------------------------
    # Resize video
    # --------------------------------
    def resize_video(
        self,
        input_path,
        output_path,
        width,
        height
    ):

        stream = ffmpeg.input(input_path)

        video = stream.video.filter(
            "scale",
            width,
            height
        )

        output_streams = [video]

        if self._has_audio(input_path):
            output_streams.append(stream.audio)

        (
            ffmpeg
            .output(
                *output_streams,
                output_path,
                vcodec="libx264",
                **(
                    {"acodec": "aac"}
                    if len(output_streams) > 1 else {}
                )
            )
            .overwrite_output()
            .run(capture_stderr=True)
        )

        return output_path

    # --------------------------------
    # Change FPS
    # --------------------------------
    def change_fps(
        self,
        input_path,
        output_path,
        fps=30
    ):

        stream = ffmpeg.input(input_path)

        video = stream.video.filter(
            "fps",
            fps=fps
        )

        output_streams = [video]

        if self._has_audio(input_path):
            output_streams.append(stream.audio)

        (
            ffmpeg
            .output(
                *output_streams,
                output_path,
                vcodec="libx264",
                pix_fmt="yuv420p",
                **(
                    {"acodec": "aac"}
                    if len(output_streams) > 1 else {}
                )
            )
            .overwrite_output()
            .run(capture_stderr=True)
        )

        return output_path

    # --------------------------------
    # Extract audio
    # --------------------------------
    def extract_audio(
        self,
        input_path,
        output_path,
        sample_rate=16000
    ):

        if not self._has_audio(input_path):
            raise ValueError(
                "Cannot extract audio: no audio stream found"
            )

        (
            ffmpeg
            .input(input_path)
            .output(
                output_path,
                vn=None,
                acodec="pcm_s16le",
                ac=1,
                ar=sample_rate
            )
            .overwrite_output()
            .run(capture_stderr=True)
        )

        return output_path

    # --------------------------------
    # Convert audio
    # --------------------------------
    def convert_audio(
        self,
        input_path,
        output_path,
        sample_rate=16000
    ):

        (
            ffmpeg
            .input(input_path)
            .output(
                output_path,
                acodec="pcm_s16le",
                ac=1,
                ar=sample_rate
            )
            .overwrite_output()
            .run(capture_stderr=True)
        )

        return output_path
