import cv2
from ultralytics import YOLO


class YOLOProcessor:

    def __init__(
        self,
        model_name="yolo26n.pt",
        confidence=0.35
    ):

        self.model = YOLO(model_name)
        self.confidence = confidence

        # CPU-only execution
        self.device = "cpu"

    def detect_objects(self, frame):

        bgr_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )

        results = self.model.predict(
            source=bgr_frame,
            conf=self.confidence,
            device=self.device,
            verbose=False
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            boxes = result.boxes

            for index in range(len(boxes)):

                class_id = int(
                    boxes.cls[index].item()
                )

                label = result.names[class_id]

                confidence = float(
                    boxes.conf[index].item()
                )

                bbox = (
                    boxes.xyxy[index]
                    .cpu()
                    .tolist()
                )

                detections.append({
                    "label": label,
                    "class_id": class_id,
                    "confidence": confidence,
                    "bbox": bbox
                })

        return detections

    def process_frames(self, frames):

        output = []

        for frame_data in frames:

            detections = self.detect_objects(
                frame_data["frame"]
            )

            output.append({
                "keyframe_id": frame_data["keyframe_id"],
                "scene_id": frame_data["scene_id"],
                "frame_number": frame_data["frame_number"],
                "timestamp_seconds": frame_data["timestamp_seconds"],
                "timestamp": frame_data["timestamp"],
                "detected_objects": detections
            })

        return output