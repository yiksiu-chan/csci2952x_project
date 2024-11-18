import os
import json
import pandas as pd
from PIL import Image

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoProcessor
from datasets import load_dataset


class MSCOCODataset(Dataset):
    def __init__(self, captions_file, neg_captions_file, image_folder, text_model_name, vision_model_name):
        """
        Args:
            captions_file (str): Path to the captions JSON file.
            neg_captions_file (str): Path to the CSV file with negative captions.
            image_folder (str): Path to the folder with all the images.
            text_model_name (str): Pretrained model name for the text tokenizer (e.g., BERT for CLIP).
            vision_model_name (str): Pretrained model name for the vision processor (e.g., ViT for CLIP).
        """
        # Load the captions JSON and group by image_id
        with open(captions_file, 'r') as f:
            captions_data = json.load(f)
        
        # Group captions by image_id
        self.image_to_captions = {}
        for annotation in captions_data["annotations"]:
            image_id = annotation["image_id"]
            caption = annotation["caption"]
            if image_id not in self.image_to_captions:
                self.image_to_captions[image_id] = []
            self.image_to_captions[image_id].append(caption)

        # Load the negative captions CSV
        self.negative_captions_data = pd.read_csv(neg_captions_file)
        self.negative_captions_data.set_index("image_id", inplace=True)  # Index by image_id for fast lookup

        self.image_folder = image_folder
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.processor = AutoProcessor.from_pretrained(vision_model_name, use_fast=True)

        # Store unique image IDs
        self.image_ids = list(self.image_to_captions.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # Retrieve image filename from negative captions CSV
        if image_id in self.negative_captions_data.index:
            image_files = self.negative_captions_data.loc[image_id, "image_file"]
            if isinstance(image_files, pd.Series):
                image_file = image_files.iloc[0]
            else:
                image_file = image_files
        else:
            raise FileNotFoundError(f"Image file for image_id {image_id} not found in negative captions CSV.")

        # Construct the full image path
        image_filename = os.path.join(self.image_folder, image_file)

        # Open and process the image
        image = Image.open(image_filename).convert("RGB")
        image_input = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        # Tokenize positive captions
        positive_captions = self.image_to_captions[image_id]
        pos_text_inputs = [
            self.tokenizer(caption, return_tensors="pt")
            for caption in positive_captions
        ]
        positive_input_ids = [pos['input_ids'].squeeze(0) for pos in pos_text_inputs]
        positive_attention_masks = [pos['attention_mask'].squeeze(0) for pos in pos_text_inputs]

        # Retrieve and tokenize negative captions
        negative_captions = []
        if image_id in self.negative_captions_data.index:
            negative_captions = self.negative_captions_data.loc[image_id, "variant"]
            if isinstance(negative_captions, str):
                negative_captions = [negative_captions]
            elif isinstance(negative_captions, pd.Series):
                negative_captions = negative_captions.tolist()

        # Remove empty or invalid negative captions
        negative_captions = [caption for caption in negative_captions if isinstance(caption, str) and caption.strip()]

        # Add a placeholder if no valid negatives exist
        while len(negative_captions) < 5:  # Ensure 5 negatives
            negative_captions.append("A generic image")

        # Tokenize negative captions
        neg_text_inputs = [
            self.tokenizer(neg_caption, return_tensors="pt")
            for neg_caption in negative_captions
        ]
        negative_input_ids = [neg['input_ids'].squeeze(0) for neg in neg_text_inputs]
        negative_attention_masks = [neg['attention_mask'].squeeze(0) for neg in neg_text_inputs]

        return {
            'pixel_values': image_input,
            'positive_input_ids': positive_input_ids,  # List of tensors
            'positive_attention_masks': positive_attention_masks,  # List of tensors
            'negative_input_ids': negative_input_ids,  # List of tensors
            'negative_attention_masks': negative_attention_masks,  # List of tensors
        }

    

class CustomCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        # Extract image tensors
        pixel_values = [item['pixel_values'] for item in batch]

        # Flatten positive and negative input IDs and attention masks
        all_positive_input_ids = [pos for item in batch for pos in item['positive_input_ids']]
        all_positive_attention_masks = [pos for item in batch for pos in item['positive_attention_masks']]
        all_negative_input_ids = [neg for item in batch for neg in item['negative_input_ids']]
        all_negative_attention_masks = [neg for item in batch for neg in item['negative_attention_masks']]

        # Determine the global maximum sequence length across all captions
        max_seq_len = max(
            max([seq.size(0) for seq in all_positive_input_ids], default=0),
            max([seq.size(0) for seq in all_negative_input_ids], default=0)
        )

        # Pad positive input IDs and attention masks
        positive_input_ids_padded = pad_sequence(
            [torch.cat([seq, torch.full((max_seq_len - seq.size(0),), self.pad_token_id)]) for seq in all_positive_input_ids],
            batch_first=True
        )
        positive_attention_masks_padded = pad_sequence(
            [torch.cat([seq, torch.zeros(max_seq_len - seq.size(0))]) for seq in all_positive_attention_masks],
            batch_first=True
        )

        # Pad negative input IDs and attention masks
        negative_input_ids_padded = pad_sequence(
            [torch.cat([seq, torch.full((max_seq_len - seq.size(0),), self.pad_token_id)]) for seq in all_negative_input_ids],
            batch_first=True
        )
        negative_attention_masks_padded = pad_sequence(
            [torch.cat([seq, torch.zeros(max_seq_len - seq.size(0))]) for seq in all_negative_attention_masks],
            batch_first=True
        )

        return {
            'pixel_values': torch.stack(pixel_values),
            'positive_input_ids': positive_input_ids_padded,
            'positive_attention_masks': positive_attention_masks_padded,
            'negative_input_ids': negative_input_ids_padded,
            'negative_attention_masks': negative_attention_masks_padded,
        }



# DataLoader Class customized for CLIP models with collator
class MSCOCODataLoader:
    def __init__(self, captions_file, neg_captions_file, image_folder, text_model_name, vision_model_name, batch_size=8):
        """
        Args:
            captions_file (str): Path to the captions JSON file.
            neg_captions_file (str): Path to the CSV file with negative captions.
            image_folder (str): Path to the folder with all the images.
            text_model_name (str): Name of the pretrained text model for tokenization (e.g., "openai-community/gpt2").
            vision_model_name (str): Name of the pretrained vision model for image processing (e.g., "google/vit-base-patch16-224-in21k").
            batch_size (int): Number of samples per batch.
        """
        self.dataset = MSCOCODataset(captions_file, neg_captions_file, image_folder, text_model_name, vision_model_name)
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