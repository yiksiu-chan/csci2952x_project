import os
import json
import pandas as pd
from PIL import Image

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoProcessor
from datasets import load_dataset

from torch.utils.data.sampler import Sampler
from collections import defaultdict
import random

# For validation: load the original MSCOCO dataset (we do not adjust validation data)
class MSCOCODataset(Dataset):
    def __init__(self, captions_file, image_folder, text_model_name, vision_model_name):
        """
        Args:
            captions_file (str): Path to the captions JSON file.
            image_folder (str): Path to the folder with all the images.
            text_model_name (str): Pretrained model name for the text tokenizer (e.g., BERT for CLIP).
            vision_model_name (str): Pretrained model name for the vision processor (e.g., ViT for CLIP).
        """
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
    

# For training: load the data with negative captions
class MSCOCONegTraining(Dataset):
    def __init__(self, neg_captions_file, image_folder, text_model_name, vision_model_name):
        """
        Args:
            neg_captions_file (str): Path to the CSV file with image IDs, file paths, and captions.
            image_folder (str): Path to the folder with all the images.
            text_model_name (str): Pretrained model name for the text tokenizer.
            vision_model_name (str): Pretrained model name for the vision processor.
        """
        # Load the captions CSV
        self.data = pd.read_csv(neg_captions_file)
        self.data.fillna("", inplace=True) # handle missing data

        self.image_folder = image_folder
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.processor = AutoProcessor.from_pretrained(vision_model_name, use_fast=True)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = row["image_id"]
        image_file = row["image_file"]
        image_path = os.path.join(self.image_folder, image_file)

        image_input = self._process_image(image_path)
        if image_input is None:
            print(f"Skipping image {image_path}: Invalid image input.")
            return None 

        positive_caption = row["original_caption"]
        negative_caption = row["variant"]

        if not positive_caption:
            print(f"Skipping: No positive caption for image ID {image_id}.")
            return None
        if not negative_caption:
            print(f"Skipping: No negative caption for image ID {image_id}.")
            return None

        pos_ids, pos_masks = self._tokenize_captions([positive_caption])
        neg_ids, neg_masks = self._tokenize_captions([negative_caption])

        # print(f"Positive IDs shape: {pos_ids[0].shape}, Negative IDs shape: {neg_ids[0].shape}")

        return {
            'pixel_values': image_input,
            'positive_input_ids': pos_ids[0],
            'positive_attention_masks': pos_masks[0],
            'negative_input_ids': neg_ids[0],
            'negative_attention_masks': neg_masks[0],
        }

    def _process_image(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            return self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        except FileNotFoundError:
            print(f"Image file {image_path} not found.")
            return None

    def _tokenize_captions(self, captions):
        tokenized = [self.tokenizer(caption, return_tensors="pt") for caption in captions]
        input_ids = [t['input_ids'].squeeze(0) for t in tokenized]
        attention_masks = [t['attention_mask'].squeeze(0) for t in tokenized]
        return input_ids, attention_masks


class CustomCollator:
    def __init__(self, pad_token_id, training=True):
        """
        Args:
            pad_token_id (int): The token ID used for padding.
            training (bool): Whether the collator is used for training (with positive and negative captions)
                             or validation (with only positive captions).
        """
        self.pad_token_id = pad_token_id
        self.training = training

    def __call__(self, batch):
        batch = [item for item in batch if item is not None]

        if len(batch) == 0:
            return None 

        pixel_values = [item['pixel_values'] for item in batch]

        # For training runs
        if self.training:
            all_positive_input_ids = [item['positive_input_ids'] for item in batch if 'positive_input_ids' in item]
            all_positive_attention_masks = [item['positive_attention_masks'] for item in batch if 'positive_attention_masks' in item]
            all_negative_input_ids = [item['negative_input_ids'] for item in batch if 'negative_input_ids' in item]
            all_negative_attention_masks = [item['negative_attention_masks'] for item in batch if 'negative_attention_masks' in item]

            # Determine max sequence length across both positive and negative captions
            max_seq_len = max(
                max([seq.size(0) for seq in all_positive_input_ids], default=0),
                max([seq.size(0) for seq in all_negative_input_ids], default=0)
            )

            # Padding
            positive_input_ids_padded = pad_sequence(
                [torch.cat([seq, torch.full((max_seq_len - seq.size(0),), self.pad_token_id)]) for seq in all_positive_input_ids],
                batch_first=True
            )
            positive_attention_masks_padded = pad_sequence(
                [torch.cat([seq, torch.zeros(max_seq_len - seq.size(0))]) for seq in all_positive_attention_masks],
                batch_first=True
            )

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

        # For validation, only have positive captions
        else: 
            all_positive_input_ids = [item['input_ids'] for item in batch if 'input_ids' in item]
            all_positive_attention_masks = [item['attention_mask'] for item in batch if 'attention_mask' in item]
            max_seq_len = max([seq.size(0) for seq in all_positive_input_ids], default=0)

            # print(f"Length of all positive input ids: {max_seq_len}")

            positive_input_ids_padded = pad_sequence(
                [torch.cat([seq, torch.full((max_seq_len - seq.size(0),), self.pad_token_id)]) for seq in all_positive_input_ids],
                batch_first=True
            )
            positive_attention_masks_padded = pad_sequence(
                [torch.cat([seq, torch.zeros(max_seq_len - seq.size(0))]) for seq in all_positive_attention_masks],
                batch_first=True
            )

            return {
                'pixel_values': torch.stack(pixel_values),
                'input_ids': positive_input_ids_padded,
                'attention_mask': positive_attention_masks_padded,
            }


# In training runs, ensure each a unique image only appears in a batch once
class UniqueImageSampler(Sampler):
    def __init__(self, data_source, batch_size):
        """
        Args:
            data_source (Dataset): Dataset object.
            batch_size (int): Number of samples per batch.
        """
        self.data_source = data_source
        self.batch_size = batch_size

        # Group indices by image_id
        self.image_id_to_indices = defaultdict(list)

        if hasattr(data_source, "captions_data"):  # For MSCOCODataset
            annotations = data_source.captions_data["annotations"]
            for idx, annotation in enumerate(annotations):
                image_id = annotation["image_id"]
                self.image_id_to_indices[image_id].append(idx)

        elif hasattr(data_source, "data"):  # For MSCOCONegTraining
            for idx, row in data_source.data.iterrows():
                image_id = row["image_id"]
                self.image_id_to_indices[image_id].append(idx)

        else:
            raise ValueError("Dataset is missing expected data attributes.")

        # List of unique image_ids
        self.unique_image_ids = list(self.image_id_to_indices.keys())

    def __iter__(self):
        # Shuffle image_ids and indices
        random.shuffle(self.unique_image_ids)

        batch = []
        for image_id in self.unique_image_ids:
            indices = self.image_id_to_indices[image_id]
            random.shuffle(indices)  # Shuffle rows for this image_id

            # Add shuffled indices to the batch
            for idx in indices:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

        # Handle the remaining batch
        if batch:
            yield batch

    def __len__(self):
        # Number of unique image_ids determines the effective length
        return len(self.unique_image_ids)



# DataLoader Class customized for CLIP models with collator
class MSCOCODataLoader:
    def __init__(self, captions_file, image_folder, text_model_name, vision_model_name, batch_size=8, training=True):
        """
        Args:
            captions_file (str): Path to the captions JSON file.
            neg_captions_file (str): Path to the CSV file with negative captions.
            image_folder (str): Path to the folder with all the images.
            text_model_name (str): Name of the pretrained text model for tokenization.
            vision_model_name (str): Name of the pretrained vision model for image processing.
            batch_size (int): Number of samples per batch.
            training (bool): Whether this is for training or evaluation.
        """
        self.training = training

        # Initialize dataset based on training or evaluation
        if self.training: 
            # Training with negative samples
            print("Using negative captions file for training")
            self.dataset = MSCOCONegTraining(captions_file, image_folder, text_model_name, vision_model_name)
        else: 
            # Loading evaluation data
            print("Loading validation data")
            self.dataset = MSCOCODataset(captions_file, image_folder, text_model_name, vision_model_name)

        self.batch_size = batch_size

    def load_datasets(self):
        collator = CustomCollator(pad_token_id=self.dataset.tokenizer.pad_token_id, training=self.training)

        if self.training:
            # Use UniqueImageSampler during training
            sampler = UniqueImageSampler(self.dataset, batch_size=self.batch_size)
            data_loader = DataLoader(
                self.dataset, 
                batch_sampler=sampler,
                collate_fn=collator
            )
        else:
            # Use standard DataLoader during evaluation
            data_loader = DataLoader(
                self.dataset, 
                batch_size=self.batch_size,
                shuffle=False, 
                collate_fn=collator
            )

        return data_loader