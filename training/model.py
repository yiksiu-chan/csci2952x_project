import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
from peft import get_peft_model, LoraConfig

# https://www.reddit.com/r/LocalLLaMA/comments/15sgg4m/what_modules_should_i_target_when_training_using/
def find_target_modules(model):
    # Initialize a Set to Store Unique Layers
    unique_layers = set()
    
    # Iterate Over All Named Modules in the Model
    for name, module in model.named_modules():
        # Check if the Module Type Contains 'Linear4bit'
        if "h" in str(type(module)):
        # if True:
            # Extract the Type of the Layer
            layer_type = name.split('.')[-1]
            
            # Add the Layer Type to the Set of Unique Layers
            unique_layers.add(layer_type)

    # Return the Set of Unique Layers Converted to a List

    print(list(unique_layers))

    return list(unique_layers)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.projection(x)


class CustomCLIP(nn.Module):
    def __init__(self, text_model_name, vision_model_name, embedding_dim=768, use_peft=False):
        super().__init__()
        # Load models from Huggingface
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.vision_model = AutoModel.from_pretrained(vision_model_name)


        # Check if PEFT (LoRA) is used
        if use_peft:
            lora_config_text = LoraConfig(r=256, lora_alpha=512, lora_dropout=0.1, target_modules='all-linear')
            lora_config_vision = LoraConfig(r=256, lora_alpha=512, lora_dropout=0.1, target_modules='all-linear')

            # Apply LoRA to both text and vision models
            self.text_model = get_peft_model(self.text_model, lora_config_text)
            self.vision_model = get_peft_model(self.vision_model, lora_config_vision)
        else:
            # Freeze the weights of the encoders if PEFT is not used
            for param in self.text_model.parameters():
                param.requires_grad = False

            for param in self.vision_model.parameters():
                param.requires_grad = False

        self.use_peft = use_peft

        # Get hidden size from both models
        text_hidden_size = self.text_model.config.hidden_size
        vision_hidden_size = self.vision_model.config.hidden_size

        # Define projection heads to map to joint space
        self.text_projection = ProjectionHead(text_hidden_size, embedding_dim)
        self.vision_projection = ProjectionHead(vision_hidden_size, embedding_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, vision_inputs):
        """Encode images and project to joint embedding space."""

        if self.use_peft:
            vision_embeddings = self.vision_model(**vision_inputs).last_hidden_state[:, 0, :]  # CLS token
        else:
            with torch.no_grad():
                vision_embeddings = self.vision_model(**vision_inputs).last_hidden_state[:, 0, :]  # CLS token
        
        vision_projected = self.vision_projection(vision_embeddings)
        vision_projected = F.normalize(vision_projected, dim=-1)
        
        return vision_projected

    def encode_text(self, text_inputs):
        """Encode text and project to joint embedding space."""
        
        if self.use_peft:
            text_embeddings = self.text_model(**text_inputs).last_hidden_state[:, -1, :]  # EOS token for gpt2
        else:
            with torch.no_grad():
                text_embeddings = self.text_model(**text_inputs).last_hidden_state[:, -1, :]  # EOS token for gpt2
        
        text_projected = self.text_projection(text_embeddings)
        text_projected = F.normalize(text_projected, dim=-1)
        
        return text_projected

    def forward(self, text_inputs, vision_inputs):
        """
        Compute the encodings of text and vision inputs

        Compute pairwise cosine similarities between text and image embeddings.
        Returns a matrix where each element [i, j] represents the cosine similarity
        between the i-th text input and the j-th image input.
        """

        image_features = self.encode_image(vision_inputs)
        text_features = self.encode_text(text_inputs)

        # Acknowledgement: the logits computation are borrowed from the official CLIP implementation
        # reference: https://github.com/openai/CLIP/blob/main/clip/model.py 

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text
