import ffmpeg
import os

class FFmpegProcessor :

    def validate_video(self, video_path) :

        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)

        try :
            ffmpeg.probe(video_path)
            return True

        except ffmpeg.Error :
            return False

    def normalize_video(
        self, input_path, output_path, width = 1200, fps = None
    ) :
        stream = ffmpeg.input(input_path)
        video = stream.video.filter(
            "scale",
            width,
            "-2"
        )

        if fps is not None :
            video = video.filter("fps", fps= fps)

        output = ffmpeg.output(
            video,
            stream.audio,
            output_path,
            vcodes = "libx264",
            acodec = "aac",
            pix_fmt = "yuv420p",
            preset = "medium",
            crf = 23
        )

        output.overwrite_output().run(
            capture_stdout = True,
            capture_stderr = True
        )

        return output_path

    def convert_video(
        self, input_path, output_path
    ):
        (
            ffmpeg
            .input(input_path)
            .output(
                output_path,
                vcodec = "libx264",
                acodec = "aac"
            )
            .overwrite_output()
            .run(capture_stderr=True)
        )

        return output_path

    def resize_video(
        self, input_path, output_path, width, height
    ):

        (
            ffmpeg
            .input(input_path)
            .filter("scale", width, height)
            .output(
                output_path,
                vcodec = "libx264"
            )
            .overwrite_output()
            .run(capture_stderr=True)
        )

        return output_path

    def change_fps(
        self, input_path, output_path, fps = 30
    ):

        (
            ffmpeg
            .input(input_path)
            .filter("fps", fps=fps)
            .output(
                output_path,
                vcodec = "libx264",
                pix_fmt = "yuv420p"
            )
            .overwrite_output()
            .run(capture_stderr=True)
        )

        return output_path

    def extract_audio(
        self, input_path, output_path, sample_rate = 16000
    ):

        (
            ffmpeg
            .input(input_path)
            .output(
                output_path,
                vn = None,
                acodec = "pcm_s16le",
                ac = 1,
                ar = sample_rate
            )
            .overwrite_output()
            .run(capture_stderr=True)
        )

        return output_path

    def convert_audio(
        self, input_path, output_path, sample_rate = 16000
    ):

        (
            ffmpeg
            .input(input_path)
            .output(
                output_path,
                acodec = "pcm_s16le",
                ac = 1,
                ar = sample_rate
            )
            .overwrite_output()
            .run(capture_stderr=True)
        )

        return output_path