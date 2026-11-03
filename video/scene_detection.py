from scenedetect import (
    open_video,
    SceneManager,
    ContentDetector,
)

class SceneDetectionProcessor : 

    def __init__(self, threshold = 27.0, min_scene_len = 15) :
        self.threshold = threshold
        self.min_scene_len = min_scene_len

    def detect_scenes(self, video_path) :
        video = open_video(video_path)
        scene_manager = SceneManager()
        detector = ContentDetector(
            threshold = self.threshold,
            min_scene_len = self.min_scene_len
        )

        scene_manager.add_detector(detector)

        scene_manager.detect_scenes(
            video = video,
            show_progress = True
        )

        scene_list = scene_manager.get_scene_list(
            start_in_scene = True
        )

        scenes = []

        for index, (start, end) in enumerate(scene_list):

            scenes.append({
                "scene_id": index + 1,

                "start_timestamp": start.get_timecode(),
                "end_timestamp": end.get_timecode(),

                "start_seconds": start.get_seconds(),
                "end_seconds": end.get_seconds(),

                "start_frame": start.get_frames(),
                "end_frame": end.get_frames()
            })

        return scenes