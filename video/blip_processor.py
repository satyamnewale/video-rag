import torch

from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration
)


class BLIPProcessor:

    def __init__(
        self,
        model_name="Salesforce/blip-image-captioning-base"
    ):

        # CPU-only execution
        self.device = torch.device("cpu")

        self.processor = BlipProcessor.from_pretrained(
            model_name
        )

        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name
        ).to(self.device)

        self.model.eval()

    def generate_caption(self, frame):

        image = Image.fromarray(frame).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            output = self.model.generate(
                **inputs,
                max_new_tokens=60,
                num_beams=3
            )

        caption = self.processor.decode(
            output[0],
            skip_special_tokens=True
        )

        return caption.strip()

    def process_frames(self, frames):

        output = []

        for frame_data in frames:

            description = self.generate_caption(
                frame_data["frame"]
            )

            output.append({
                "keyframe_id": frame_data["keyframe_id"],
                "scene_id": frame_data["scene_id"],
                "frame_number": frame_data["frame_number"],
                "timestamp_seconds": frame_data["timestamp_seconds"],
                "timestamp": frame_data["timestamp"],
                "visual_description": description
            })

        return output