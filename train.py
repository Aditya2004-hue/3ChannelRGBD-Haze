import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import Encoder, Decoder
from dataset import DenseHazeNPYDataset
import os
from tqdm import tqdm

# -------------------------
# Settings
# -------------------------
EPOCHS = 500
BATCH_SIZE = 1
LEARNING_RATE = 0.0005
SAVE_PATH = "dehaze_autoencoder_rgb.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load dataset (expects RGB .npy arrays)
# -------------------------
print("📂 Loading RGB dataset...")
dataset = DenseHazeNPYDataset(hazy_path="hazy.npy", gt_path="GT.npy")  # RGB .npy data

train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# -------------------------
# Initialize model, loss, optimizer
# -------------------------
encoder = Encoder(input_channels=3).to(DEVICE)
decoder = Decoder(output_channels=3).to(DEVICE)

parameters = list(encoder.parameters()) + list(decoder.parameters())
criterion = nn.MSELoss()
optimizer = optim.Adam(parameters, lr=LEARNING_RATE)

# -------------------------
# Training loop
# -------------------------
print("🚀 Starting training...")
best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    encoder.train()
    decoder.train()
    running_loss = 0.0

    for gt, hazy in tqdm(train_loader, desc=f"Epoch {epoch:03d}/{EPOCHS}", leave=False):
        gt = gt.to(DEVICE)
        hazy = hazy.to(DEVICE)

        optimizer.zero_grad()
        features = encoder(hazy)
        output = decoder(features)

        loss = criterion(output, gt)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # -------------------------
    # Validation
    # -------------------------
    encoder.eval()
    decoder.eval()
    val_loss = 0.0
    with torch.no_grad():
        for gt, hazy in val_loader:
            gt = gt.to(DEVICE)
            hazy = hazy.to(DEVICE)

            output = decoder(encoder(hazy))
            val_loss += criterion(output, gt).item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"📘 Epoch [{epoch}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict()
        }, SAVE_PATH)
        print("✅ Saved new best RGB model!")

print("🏁 RGB training complete.")
