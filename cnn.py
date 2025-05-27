import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np

# --- Settings ---
image_dir = input("Enter the path to the input image directory: ").strip()
output_dir = input("Enter the path to the output directory: ").strip()
os.makedirs(output_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Utils ---
def total_variation_loss(x):
    """
    Calculates the total variation loss for a given tensor.
    This helps to encourage spatial smoothness in the reconstructed images,
    contributing to better visual quality and potentially more efficient compression.
    """
    return torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])) + \
           torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))

# --- Transform ---
class PadToMultiple:
    """
    Pads an image to ensure its dimensions are a multiple of a given number.
    This is useful for convolutional layers that might require specific input dimensions.
    """
    def __init__(self, multiple=1):
        self.multiple = multiple

    def __call__(self, img):
        w, h = img.size
        # Calculate new dimensions that are multiples of 'multiple'
        new_w = ((w + self.multiple - 1) // self.multiple) * self.multiple
        new_h = ((h + self.multiple - 1) // self.multiple) * self.multiple
        # Calculate padding needed
        pad = [(new_w - w) // 2, (new_h - h) // 2]
        # Apply padding (left, top, right, bottom)
        return TF.pad(img, (pad[0], pad[1], new_w - w - pad[0], new_h - h - pad[1]))

# Define the transformation pipeline for the images
transform = transforms.Compose([
    PadToMultiple(16), # Pad to a multiple of 16
    transforms.ToTensor() # Convert PIL Image to PyTorch tensor
])

# --- Dataset ---
class ImageFolderDataset(Dataset):
    """
    Custom Dataset class to load images from a specified directory.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        # List all image files with common extensions
        self.img_files = [f for f in os.listdir(root_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
        self.transform = transform

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.img_files)

    def __getitem__(self, idx):
        """
        Retrieves an image and its name at the given index.
        Applies transformations if provided.
        """
        img_name = self.img_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        # Open image and convert to RGB (important for consistent channel count and color preservation)
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        return img, img_name

# --- Model ---
class LatentRefiner(nn.Module):
    """
    A small convolutional network to refine the latent space representation.
    """
    def __init__(self, channels=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 1), # 1x1 convolution
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1) # Another 1x1 convolution
        )
    def forward(self, z):
        return self.net(z)

class ColorCompressor(nn.Module):
    """
    An autoencoder-like model for image compression and color refinement.
    It consists of an encoder, a latent refiner, and a decoder.
    The model aims to reduce image file size by learning a compact latent representation
    while preserving image color and quality through reconstruction.
    """
    def __init__(self):
        super().__init__()
        # Encoder: Reduces spatial dimensions and increases channel depth
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(), # Output: (N, 16, H/2, W/2)
            nn.Conv2d(16, 8, 3, 2, 1), nn.ReLU(), # Output: (N, 8, H/4, W/4)
            nn.Conv2d(8, 4, 3, 2, 1), nn.ReLU()   # Output: (N, 4, H/8, W/8)
        )
        self.refiner = LatentRefiner(4) # Refines the latent representation
        # Decoder: Increases spatial dimensions and reduces channel depth
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 8, 3, 2, 1, 1), nn.ReLU(),  # Output: (N, 8, H/4, W/4)
            nn.ConvTranspose2d(8, 16, 3, 2, 1, 1), nn.ReLU(), # Output: (N, 16, H/2, W/2)
            nn.ConvTranspose2d(16, 3, 3, 2, 1, 1), nn.Sigmoid() # Output: (N, 3, H, W), values between 0 and 1
        )
    def forward(self, x):
        z = self.encoder(x) # Encode the input image to a latent representation
        z_refined = self.refiner(z) # Refine the latent representation
        out = self.decoder(z_refined) # Decode the refined latent representation back to an image
        return out, z_refined

# --- Training ---
# Initialize the model and move it to the appropriate device (GPU if available, else CPU)
model = ColorCompressor().to(device)
# Define the optimizer (Adam is a good general-purpose optimizer)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
# Define the loss function (Mean Squared Error for reconstruction)
# This criterion compares the original images to the images produced by the CNN.
criterion = nn.MSELoss()
# Create the dataset instance
dataset = ImageFolderDataset(image_dir, transform)
# Create the DataLoader for batching and shuffling data
# Set batch_size to 1 to handle images of varying sizes without explicit resizing.
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Define loss coefficients
alpha, beta = 1e-3, 1e-4

# Training loop
for epoch in range(1, 100): # Iterate for a fixed number of epochs
    model.train() # Set the model to training mode
    total = 0.0 # Initialize total loss for the epoch
    for i, (img, names) in enumerate(loader): # Iterate over batches
        img = img.to(device) # Move image batch to the device

        # Forward pass: get reconstructed image and latent representation
        recon, z = model(img)
        # Calculate total loss: reconstruction loss + L1 regularization on latent + total variation loss on reconstruction
        # The L1 regularization on 'z' encourages sparsity in the latent representation, aiding compression.
        # Total variation loss helps maintain image smoothness and color integrity.
        loss = criterion(recon, img) + alpha * torch.mean(torch.abs(z)) + beta * total_variation_loss(recon)
        
        # Backward pass and optimization
        opt.zero_grad() # Clear gradients
        loss.backward() # Compute gradients
        opt.step() # Update model parameters
        
        total += loss.item() # Accumulate loss

        # Save batch reconstruction results
        for j in range(img.size(0)): # Iterate over images in the current batch (will be 1 since batch_size=1)
            # Convert tensor to numpy array, transpose dimensions, scale to 0-255, and convert to uint8
            npimg = (recon[j].cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            # Save the reconstructed image. The file size of this image will be a result of the compression.
            Image.fromarray(npimg).save(os.path.join(output_dir, f'epoch{epoch}_batch{i}_{names[j]}.jpg'))

    # Print average loss for the epoch
    print(f'Epoch {epoch} Loss: {total / len(loader):.4f}')

# --- Save model ---
# Save the trained model's state dictionary
torch.save(model.state_dict(), 'model.pth')
print("Model saved to model.pth")
