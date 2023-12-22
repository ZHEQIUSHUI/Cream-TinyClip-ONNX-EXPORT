import torch
from PIL import Image
import src.open_clip as open_clip
import torch.nn as nn
import numpy as np

class ClipDecoder(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image_features, text_features):
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        logits_per_image = 100 * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image.softmax(1), logits_per_text.softmax(1)



# manual inheritance
# arch = 'TinyCLIP-ViT-39M-16-Text-19M'
# model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained='YFCC15M')

arch = 'TinyCLIP-ViT-8M-16-Text-3M'
model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained='YFCC15M')

# arch = 'TinyCLIP-ResNet-30M-Text-29M'
# model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained='LAION400M')

# arch = 'TinyCLIP-ResNet-19M-Text-19M'
# model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained='LAION400M')

# arch = 'TinyCLIP-ViT-61M-32-Text-29M'
# model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained='LAION400M')

# arch = 'TinyCLIP-ViT-40M-32-Text-19M'
# model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained='LAION400M')

tokenizer = open_clip.get_tokenizer(arch)

image_fname = './figure/TinyCLIP.jpg'
image = preprocess(Image.open(image_fname)).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    # text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)
