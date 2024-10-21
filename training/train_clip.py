import argparse

import torch
import wandb

from model import CustomCLIP
from dataset import MSCOCODataLoader
from trainer import Trainer

def get_args():
    parser = argparse.ArgumentParser(description="Custom CLIP Model Training Script")

    # Model configuration
    parser.add_argument("--text_model_name", type=str, default="openai-community/gpt2", 
                        help="Pretrained text model name")
    parser.add_argument("--vision_model_name", type=str, default="google/vit-base-patch16-224-in21k", 
                        help="Pretrained vision model name")
    parser.add_argument("--embedding_dim", type=int, default=1024, 
                        help="Dimension of the joint embedding space")
    parser.add_argument("--use_peft", action="store_true", 
                        help="Whether to use Parameter-Efficient Fine-Tuning (PEFT)")

    # Training configuration
    parser.add_argument("--epochs", type=int, default=10, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for training and validation")
    parser.add_argument("--learning_rate", type=float, default=1e-3, 
                        help="Learning rate for optimizer")
    parser.add_argument("--log_interval", type=int, default=100, 
                        help="How many batches to wait before logging training status")
    parser.add_argument("--use_wandb", action="store_true", 
                        help="Whether to log metrics to Weights and Biases")
    parser.add_argument("--checkpoint_dir", type=str, default="/users/thua5/ssl_proj/checkpoints", 
                        help="Directory where model checkpoints will be saved")

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define model and data loader
    clip_model = CustomCLIP(
        args.text_model_name, 
        args.vision_model_name, 
        embedding_dim=args.embedding_dim,
        use_peft=args.use_peft
    ).to(device)

    # Setup optimizer
    optimizer = torch.optim.Adam(clip_model.parameters(), lr=args.learning_rate)

    # Initialize the trainer
    trainer = Trainer(
        clip_model, 
        optimizer, 
        batch_size=args.batch_size, 
        log_interval=args.log_interval, 
        use_wandb=args.use_wandb,
        checkpoint_dir=args.checkpoint_dir
    )

    train_dataloader = MSCOCODataLoader(
        "/gpfs/data/superlab/datasets/coco/annotations/captions_train2017.json", 
        "/gpfs/data/superlab/datasets/coco/train2017", 
        args.text_model_name, 
        args.vision_model_name,
        batch_size = args.batch_size
    ).load_datasets()

    val_dataloader = MSCOCODataLoader(
        "/gpfs/data/superlab/datasets/coco/annotations/captions_val2017.json", 
        "/gpfs/data/superlab/datasets/coco/val2017", 
        args.text_model_name, 
        args.vision_model_name,
        batch_size = args.batch_size
    ).load_datasets()

    if args.use_wandb:
        wandb.init(project="clip_training", name="clip_with_lora_and_mscoco") # TODO: improve the naming of different runs

        wandb.config["args"] = vars(args)

    # Train the model
    trainer.train(
        train_dataloader, 
        val_dataloader, 
        epochs=args.epochs
    )