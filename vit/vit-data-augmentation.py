# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import random

# %%
# device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
# set the seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)

# %%
# hyperparameters
BATCH_SIZE = 128
EPOCHS = 100
LR = 3e-4
PATCH_SIZE = 4
NUM_CLASSES = 10
IMAGE_SIZE = 32
CHANNELS = 3
EMBED_DIM = 256
NUM_HEADS = 8
DEPTH = 6
MLP_DIM = 512
DROP_RATE = 0.1

# %%
# transformations
# Training augmentation: RandomCrop, RandomHorizontalFlip, ColorJitter, RandomErasing
train_transform = transforms.Compose([
    transforms.RandomCrop(IMAGE_SIZE, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # CIFAR-10 stats
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
])

# Test transform: only normalization (no augmentation)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # CIFAR-10 stats
])

# %%
# dataset
train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=train_transform)
test_data = datasets.CIFAR10(root='data', train=False, download=True, transform=test_transform)

# %%
# convert dataset into dataloader
train_loader = DataLoader(batch_size=BATCH_SIZE, dataset=train_data, shuffle=True)
test_loader = DataLoader(batch_size=BATCH_SIZE, dataset=test_data, shuffle=False)

# %%
# vit 
class PatchEmbedding(nn.Module):
    def __init__(self, 
                img_size,
                patch_size,
                in_channels,
                embed_dim
    ):
        super().__init__()
        self.batch_size = BATCH_SIZE
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1,1+num_patches, embed_dim))

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        x = self.proj(x) # B x E x H/P x W/P
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(B, -1, -1) # expand method does the follwoing: repeat the tensor along the specified dimensions without actually copying the data in memory.
        x = torch.concat((cls_token, x), dim=1)
        x += self.pos_embed
        return x


# %%
# visualizer
# x = torch.arange(120).reshape(2, 3, 4, 5)
# TODO: concat, (B,-1,-1)

# %%
class MLP(nn.Module):
    def __init__(self, 
                 in_features,
                 hidden_features,
                 drop_rate) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features) 
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=in_features) 
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        x = self.dropout(F.gelu(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x

# %%
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 mlp_dim,
                 drop_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=drop_rate, batch_first=True)   #TODO: from scratch.
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, drop_rate)

    def forward(self, x, return_attn: bool = False):
        # Self-attention (return attention weights when requested)
        q = self.norm1(x)
        k = self.norm1(x)
        v = self.norm1(x)
        attn_out, attn_weights = self.attention(q, k, v, need_weights=True, average_attn_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))     #TODO: switch norm and mlp
        if return_attn:
            return x, attn_weights
        return x

# %%
class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, mlp_dim, drop_rate):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.encoder = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, drop_rate) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        for layer in self.encoder:
            x = layer(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        return self.head(cls_token)

    def forward_with_attentions(self, x):
        """Run a forward pass and return (logits, list_of_attentions).
        Each attention in the list has shape (B, num_heads, N, N) where N=1+num_patches.
        """
        x = self.patch_embed(x)
        attn_list = []
        for layer in self.encoder:
            x, attn = layer(x, return_attn=True)
            attn_list.append(attn)
        x = self.norm(x)
        cls_token = x[:, 0]
        return self.head(cls_token), attn_list

    def get_attentions(self, x):
        """Convenience wrapper: return only the attention list for input x."""
        self.eval()
        with torch.no_grad():
            _, attn_list = self.forward_with_attentions(x)
        return attn_list

# %%
# instantiate the model
model = VisionTransformer(
    img_size=IMAGE_SIZE,
    patch_size=PATCH_SIZE,
    in_channels=CHANNELS,
    num_classes=NUM_CLASSES,
    embed_dim=EMBED_DIM,
    depth=DEPTH,
    num_heads=NUM_HEADS,
    mlp_dim=MLP_DIM,
    drop_rate=DROP_RATE
).to(device)

# %%
# define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# %%
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# %%
from tqdm.auto import tqdm
import json
import os

# Metric tracking
metrics = {
    'train_loss': [],
    'train_acc': [],
    'test_loss': [],
    'test_acc': [],
    'learning_rate': []
}

for epoch in tqdm(range(EPOCHS)):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    
    # Track metrics
    metrics['train_loss'].append(train_loss)
    metrics['train_acc'].append(train_acc)
    metrics['test_loss'].append(test_loss)
    metrics['test_acc'].append(test_acc)
    metrics['learning_rate'].append(optimizer.param_groups[0]['lr'])

    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Step scheduler
    scheduler.step()

# Save metrics to disk
os.makedirs('metrics', exist_ok=True)
with open('metrics/training_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\nMetrics saved to metrics/training_metrics.json")

# %%
def plot_training_metrics(metrics: dict, save_path: str = 'metrics/training_curves.png'):
    """Plot and save training metrics: loss, accuracy, and learning rate."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, metrics['train_loss'], label='Train Loss', marker='o', markersize=4)
    axes[0].plot(epochs, metrics['test_loss'], label='Test Loss', marker='s', markersize=4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Test Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, [acc * 100 for acc in metrics['train_acc']], label='Train Acc', marker='o', markersize=4)
    axes[1].plot(epochs, [acc * 100 for acc in metrics['test_acc']], label='Test Acc', marker='s', markersize=4)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Test Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[2].plot(epochs, metrics['learning_rate'], label='Learning Rate', marker='o', markersize=4, color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.show()


# %%
def print_metrics_summary(metrics: dict):
    """Print a summary of training metrics."""
    print("\n" + "="*60)
    print("TRAINING METRICS SUMMARY")
    print("="*60)
    print(f"Best Train Accuracy: {max(metrics['train_acc'])*100:.2f}% (Epoch {metrics['train_acc'].index(max(metrics['train_acc']))+1})")
    print(f"Best Test Accuracy:  {max(metrics['test_acc'])*100:.2f}% (Epoch {metrics['test_acc'].index(max(metrics['test_acc']))+1})")
    print(f"Final Train Loss:    {metrics['train_loss'][-1]:.4f}")
    print(f"Final Test Loss:     {metrics['test_loss'][-1]:.4f}")
    print(f"Final Train Accuracy: {metrics['train_acc'][-1]*100:.2f}%")
    print(f"Final Test Accuracy:  {metrics['test_acc'][-1]*100:.2f}%")
    print(f"Initial LR:          {metrics['learning_rate'][0]:.6f}")
    print(f"Final LR:            {metrics['learning_rate'][-1]:.6f}")
    print("="*60 + "\n")


# %%
# Visualize training metrics
try:
    plot_training_metrics(metrics)
    print_metrics_summary(metrics)
except Exception as e:
    print(f"Could not plot metrics: {e}")

# %%
def visualize_attention_on_image(model: VisionTransformer, img_tensor: torch.Tensor, patch_size: int, img_size: int, layer_idx: int = -1, head: int = 0, device: str = device):
    """Compute and display attention overlay for a single image tensor (C,H,W).

    - model: VisionTransformer instance (already on `device`).
    - img_tensor: image tensor in normalized form (C,H,W).
    - patch_size: patch size used by patch embedding.
    - img_size: full image size in pixels.
    - layer_idx: which encoder layer to visualize (default last).
    - head: which attention head to visualize (default 0).
    """
    model.eval()
    if img_tensor.dim() == 4:
        img = img_tensor[0]
    else:
        img = img_tensor

    with torch.no_grad():
        attn_list = model.get_attentions(img.unsqueeze(0).to(device))

    # select requested layer and head
    attn = attn_list[layer_idx]  # shape (B, num_heads, N, N)
    attn = attn[0, head]         # shape (N, N) for first batch, chosen head

    # take cls token attention to patches
    cls_attn = attn[0].cpu().numpy()  # length N
    cls_patch_attn = cls_attn[1:]  # drop cls->cls

    n_patches = (img_size // patch_size)
    cls_map = cls_patch_attn.reshape(n_patches, n_patches)

    # upsample patch map to image size
    attn_upsampled = np.kron(cls_map, np.ones((patch_size, patch_size)))

    # unnormalize image using CIFAR-10 mean and std
    img_np = img.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)

    plt.figure(figsize=(4, 4))
    plt.imshow(img_np)
    plt.imshow(attn_upsampled, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.title(f'Layer {layer_idx} head {head} cls-attention')
    plt.show()


# %%
# Example: visualize attention for first image from test set
try:
    images, labels = next(iter(test_loader))
    example_img = images[0]
    visualize_attention_on_image(model, example_img, PATCH_SIZE, IMAGE_SIZE, layer_idx=-1, head=0)
except Exception as e:
    print(f"Could not visualize attention: {e}")

# %%



