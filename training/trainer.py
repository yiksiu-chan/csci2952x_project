import os

import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

class Trainer:
    def __init__(self, model, optimizer, batch_size=8, log_interval=100, use_wandb=False, device='cuda', checkpoint_dir="checkpoints"):
        """
        Args:
            model (nn.Module): The CLIP model.
            optimizer (Optimizer): The optimizer for training.
            batch_size (int): Batch size for training and validation.
            log_interval (int): Interval to log training metrics.
            use_wandb (bool): Whether to log metrics to Weights & Biases.
            device (str): The device to train on ('cuda' or 'cpu').
            checkpoint_dir (str): Directory where model checkpoints will be saved.
        """
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.device = device
        self.use_wandb = use_wandb
        self.checkpoint_dir = checkpoint_dir

        # Define loss functions (cross-entropy loss)
        self.criterion = nn.CrossEntropyLoss()

        # Track the best validation loss
        self.best_val_loss = float('inf')

        # Create checkpoint directory if it does not exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch data to the target device
            batch_device = {key: val.to(self.device) for key, val in batch.items()}

            # Forward pass
            logits_per_image, logits_per_text = self.model(
                {
                    "input_ids": batch_device["input_ids"],
                    "attention_mask": batch_device["attention_mask"]
                },
                {"pixel_values": batch_device["pixel_values"]}
            )

            # Create target labels (diagonal 1s for correct matches)
            labels = torch.arange(len(batch["pixel_values"]), device=self.device)

            # Compute losses (image-to-text and text-to-image cross-entropy loss)
            loss_i2t = self.criterion(logits_per_image, labels)
            loss_t2i = self.criterion(logits_per_text, labels)
            loss = (loss_i2t + loss_t2i) / 2.0

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Update progress bar and log metrics
            progress_bar.set_postfix({"loss": total_loss / (batch_idx + 1)})
            if self.use_wandb:
                wandb.log({"train_loss": total_loss / (batch_idx + 1), "epoch": epoch + 1})

    def validate_one_epoch(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0.0
        progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")

        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch data to the target device
                batch_device = {key: val.to(self.device) for key, val in batch.items()}

                # Forward pass
                logits_per_image, logits_per_text = self.model(
                    {
                        "input_ids": batch_device["input_ids"],
                        "attention_mask": batch_device["attention_mask"]
                    },
                    {"pixel_values": batch_device["pixel_values"]}
                )

                # Create target labels (diagonal 1s for correct matches)
                labels = torch.arange(len(batch["pixel_values"]), device=self.device)

                # Compute losses
                loss_i2t = self.criterion(logits_per_image, labels)
                loss_t2i = self.criterion(logits_per_text, labels)
                loss = (loss_i2t + loss_t2i) / 2.0

                total_loss += loss.item()

                # Update progress bar and log metrics
                progress_bar.set_postfix({"val_loss": total_loss / (batch_idx + 1)})
                if self.use_wandb and ((batch_idx + 1) % self.log_interval == 0):
                    wandb.log({"val_loss": total_loss / (batch_idx + 1), "epoch": epoch + 1})

        return total_loss / len(val_loader)

    def save_checkpoint(self, epoch, val_loss): # TODO: improve the naming of the checkpoints
        """
        Saves the model checkpoint if the validation loss improves.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch+1}_val_loss_{val_loss:.4f}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    def train(self, train_loader, val_loader, epochs=10):
        """
        Train the CLIP model and evaluate on the validation set after each epoch.
        Save a checkpoint if the validation loss improves.
        """
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            
            # Train for one epoch
            self.train_one_epoch(train_loader, epoch)

            # Validate after each epoch
            val_loss = self.validate_one_epoch(val_loader, epoch)

            # Log the validation loss
            print(f"Validation Loss after Epoch {epoch + 1}: {val_loss:.4f}")
            if self.use_wandb:
                wandb.log({"val_loss_epoch": val_loss, "epoch": epoch + 1})

            # Checkpoint the model if validation loss improves
            if val_loss < self.best_val_loss:
                print(f"Validation loss improved from {self.best_val_loss:.4f} to {val_loss:.4f}. Saving checkpoint...")
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
            else:
                print(f"Validation loss did not improve. Best val loss: {self.best_val_loss:.4f}.")

        if self.use_wandb:
            wandb.finish()