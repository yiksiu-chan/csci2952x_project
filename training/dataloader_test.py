from dataset import *

# Instantiate the dataset and dataloader
# try: 
#     dataset = MSCOCODataset(
#         captions_file="../data/coco/annotations/captions_train2017.json",
#         neg_captions_file="../augment_annotations/variants_train2017_all.csv",
#         image_folder="../data/coco/images/train2017",
#         text_model_name="openai-community/gpt2",
#         vision_model_name="google/vit-base-patch16-224-in21k"
#     )

# # for idx in range(10):
# #     try:
# #         sample = dataset[idx]
# #         print(f"Sample {idx}:")
# #         print("Pixel Values Shape:", sample['pixel_values'].shape)
# #         print("Positive Input IDs Count:", len(sample['positive_input_ids']))
# #         print("Negative Input IDs Count:", len(sample['negative_input_ids']))
# #     except Exception as e:
# #         print(f"Error with sample {idx}: {e}")


#     collator = CustomCollator(pad_token_id=dataset.tokenizer.pad_token_id)
#     dataloader = DataLoader(dataset, batch_size=4, collate_fn=collator)

#     # Iterate through a batch
#     for batch_idx, batch in enumerate(dataloader):
#         try:
#             print(f"Batch {batch_idx}:")
#             print("Pixel Values Shape:", batch['pixel_values'].shape)
#             print("Positive Input IDs Shape:", batch['positive_input_ids'].shape)
#             print("Negative Input IDs Shape:", batch['negative_input_ids'].shape)
#         except FileNotFoundError as e:
#             print(f"Warning: {e}. Skipping batch {batch_idx}.")
#         break  # Test with the first batch only

# except Exception as e:
#     print(f"Error initializing dataset or dataloader: {e}")


dataset = MSCOCODataset(
    captions_file="../data/coco/annotations/captions_train2017.json",
    neg_captions_file="../augment_annotations/variants_train2017_all.csv",
    image_folder="../data/coco/images/train2017",
    text_model_name="openai-community/gpt2-large",
    vision_model_name="google/vit-base-patch16-224-in21k"
)

# for idx in range(5):  # Inspect first 5 samples
#     sample = dataset[idx]
#     print(f"Sample {idx}:")
#     for key, value in sample.items():
#         print(f"{key}: {type(value)}, {len(value) if isinstance(value, list) else value.shape}")

# collator = CustomCollator(pad_token_id=dataset.tokenizer.pad_token_id)

# # Collect a mini-batch from the dataset
# mini_batch = [dataset[i] for i in range(4)]  # Test with 4 samples
# batched_data = collator(mini_batch)

# print("Batched Keys:", batched_data.keys())
# for key, value in batched_data.items():
#     print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else type(value)}")

# collator = CustomCollator(pad_token_id=dataset.tokenizer.pad_token_id)
# batch = collator([dataset[i] for i in range(4)])

# print("Batched Pixel Values Shape:", batch['pixel_values'].shape)
# print("Positive Input IDs Shape:", batch['positive_input_ids'].shape)
# print("Negative Input IDs Shape:", batch['negative_input_ids'].shape)

# Simulated batch device dictionary
# batch_device = {
#     "positive_input_ids": torch.randn(20, 19),  # Example tensor
#     "positive_attention_masks": torch.ones(20, 19),
#     "negative_input_ids": torch.randn(20, 19),
#     "negative_attention_masks": torch.ones(20, 19)
# }

# # Concatenate
# texts_input_ids = torch.cat(
#     [batch_device["positive_input_ids"], batch_device["negative_input_ids"]],
#     dim=0
# )
# texts_attention_mask = torch.cat(
#     [batch_device["positive_attention_masks"], batch_device["negative_attention_masks"]],
#     dim=0
# )

# # Print shapes
# print("Positive Input IDs Shape:", batch_device["positive_input_ids"].shape)
# print("Negative Input IDs Shape:", batch_device["negative_input_ids"].shape)
# print("Concatenated Texts Input IDs Shape:", texts_input_ids.shape)
# print("Concatenated Texts Attention Mask Shape:", texts_attention_mask.shape)

collator = CustomCollator(pad_token_id=dataset.tokenizer.pad_token_id)

# Simulate a mini-batch from the dataset
mini_batch = [dataset[i] for i in range(4)]
batched_data = collator(mini_batch)

# Debugging outputs
print("Batched Pixel Values Shape:", batched_data['pixel_values'].shape)
print("Positive Input IDs Shape:", batched_data['positive_input_ids'].shape)
print("Negative Input IDs Shape:", batched_data['negative_input_ids'].shape)
