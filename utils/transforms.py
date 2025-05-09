import open_clip
from transformers import Blip2Processor

def clip_train(model_name: str, pretrained: str):
    _, preprocess_train, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    return preprocess_train

def clip_val(model_name: str, pretrained: str):
    _, _, preprocess_val = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    return preprocess_val




class Blip2TensorTransform:
    """Convert a PIL image → (3,224,224) tensor expected by BLIP-2."""
    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b"):
        self.processor = Blip2Processor.from_pretrained(model_name).image_processor

    def __call__(self, pil_image):
        # processor returns dict {'pixel_values': (1,3,224,224)}
        return self.processor(pil_image, return_tensors="pt")["pixel_values"][0]