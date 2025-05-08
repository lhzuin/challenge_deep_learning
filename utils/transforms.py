import open_clip

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