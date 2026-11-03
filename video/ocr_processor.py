import cv2
from paddleocr import PaddleOCR


class OCRProcessor:

    def __init__(self, lang="en"):

        self.ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="en_PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def extract_text(self, frame):

        bgr_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )

        results = self.ocr.predict(bgr_frame)

        extracted_text = []

        for result in results:

            data = result.json

            if callable(data):
                data = data()

            if hasattr(data, "model_dump"):
                data = data.model_dump()

            if not isinstance(data, dict):
                continue

            data = data.get("res", data)

            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])

            boxes = data.get(
                "rec_polys",
                data.get("rec_boxes", [])
            )

            for text, score, box in zip(texts, scores, boxes):

                extracted_text.append({
                    "text": str(text),
                    "confidence": float(score),
                    "bbox": (
                        box.tolist()
                        if hasattr(box, "tolist")
                        else box
                    )
                })

        return extracted_text

    def process_frames(self, frames):

        output = []

        for frame_data in frames:

            ocr_results = self.extract_text(
                frame_data["frame"]
            )

            output.append({
                "keyframe_id": frame_data["keyframe_id"],
                "scene_id": frame_data["scene_id"],
                "frame_number": frame_data["frame_number"],
                "timestamp_seconds": frame_data["timestamp_seconds"],
                "timestamp": frame_data["timestamp"],
                "ocr_results": ocr_results
            })

        return output