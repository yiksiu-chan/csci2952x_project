import os
import json
from PIL import Image

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoProcessor
from datasets import load_dataset


class MSCOCODataset(Dataset):
    def __init__(self, captions_file, image_folder, text_model_name, vision_model_name):
        """
        Args:
            captions_file (str): Path to the captions JSON file.
            image_folder (str): Path to the folder with all the images.
            text_model_name (str): Pretrained model name for the text tokenizer (e.g., BERT for CLIP).
            vision_model_name (str): Pretrained model name for the vision processor (e.g., ViT for CLIP).
            transform (callable, optional): Optional transform to be applied on an image.
        """
        # Load the captions JSON
        with open(captions_file, 'r') as f:
            self.captions_data = json.load(f)

        self.image_folder = image_folder
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)

        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.processor = AutoProcessor.from_pretrained(vision_model_name, use_fast=True)

    def __len__(self):
        return len(self.captions_data["annotations"])

    def __getitem__(self, idx):
        annotation = self.captions_data["annotations"][idx]
        image_id = str(annotation["image_id"]).rjust(12, '0')
        caption = annotation["caption"]

        image_filename = os.path.join(self.image_folder, f"{image_id}.jpg")
        image = Image.open(image_filename).convert("RGB")

        image_input = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        text_input = self.tokenizer(caption, return_tensors="pt")

        return {
            'pixel_values': image_input,
            'input_ids': text_input['input_ids'].squeeze(0),
            'attention_mask': text_input['attention_mask'].squeeze(0),
        }


class CustomCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        # Extract image tensors, input IDs, and attention masks from the batch
        pixel_values = [item['pixel_values'] for item in batch]
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]

        # Stack image tensors into a batch
        pixel_values = torch.stack(pixel_values)

        # Pad the input_ids and attention_masks to the same length
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id, padding_side='left')
        attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0, padding_side='left')

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids_padded,
            'attention_mask': attention_masks_padded,
        }


# DataLoader Class customized for CLIP models with collator
class MSCOCODataLoader:
    def __init__(self, captions_file, image_folder, text_model_name, vision_model_name, batch_size=8):
        """
        Args:
            captions_file (str): Path to the captions JSON file.
            image_folder (str): Path to the folder with all the images.
            text_model_name (str): Name of the pretrained text model for tokenization (e.g., "openai-community/gpt2").
            vision_model_name (str): Name of the pretrained vision model for image processing (e.g., "google/vit-base-patch16-224-in21k").
            batch_size (int): Number of samples per batch.
        """
        self.dataset = MSCOCODataset(captions_file, image_folder, text_model_name, vision_model_name)
        self.batch_size = batch_size

    def load_datasets(self):
        collator = CustomCollator(pad_token_id=self.dataset.tokenizer.pad_token_id)

        data_loader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=collator  # Use custom collator
        )
        return data_loader