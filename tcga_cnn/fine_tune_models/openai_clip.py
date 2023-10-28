import clip
from transformers import CLIPProcessor, CLIPModel
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class ClipFineTune:
    def __init__(self, model, train_loader, device="cuda", learning_rate=5e-5):
        self.model = model
        self.train_loader = train_loader
        self.device = device

        self.model = self.model.to(self.device)

        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, total=len(self.train_loader))

        for batch in pbar:
            self.optimizer.zero_grad()

            images, texts = batch 
            images = images.to(self.device)
            texts = texts.to(self.device)

            # Forward pass
            logits_per_image, logits_per_text = self.model(images, texts)

            # Compute loss
            ground_truth = torch.arange(len(images), dtype=torch.long, device=self.device)
            loss = (self.loss_img(logits_per_image, ground_truth) + self.loss_txt(logits_per_text, ground_truth)) / 2

            # Backward pass
            loss.backward()
            if self.device == "cpu":
                self.optimizer.step()
            else:
                self.convert_models_to_fp32(self.model)
                self.optimizer.step()
                clip.model.convert_weights(self.model)

            total_loss += loss.item()
            pbar.set_description(f"Loss: {loss.item():.4f}")

        return total_loss / len(self.train_loader)

    def fine_tune(self, epochs):
        for epoch in range(epochs):
            train_loss = self.train_one_epoch()
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")

    def convert_models_to_fp32(self, model): 
        for p in model.parameters(): 
            p.data = p.data.float() 
            p.grad.data = p.grad.data.float() 
