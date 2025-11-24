import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import scipy.io
from einops import rearrange
from einops.layers.torch import Rearrange
from utilities3 import MatReader, UnitGaussianNormalizer, LpLoss

# ==================== Helper Functions ====================
def to_2tuple(x):
    """Convert scalar to 2-tuple"""
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x)

# ==================== ViT Components ====================
class PatchEmbedding(nn.Module):
    """Convert image to patches and embed them"""
    def __init__(self, img_size=88, patch_size=4, in_channels=1, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2  # 22*22 = 484
        
        # Patch embedding via convolution
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x):
        # x: (B, 1, 88, 88)
        x = self.proj(x)  # (B, embed_dim, 22, 22)
        x = rearrange(x, 'b c h w -> b (h w) c')  # (B, 484, embed_dim)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Combine heads
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x

class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)
        
    def forward(self, x):
        # Attention with residual
        x = x + self.attn(self.norm1(x))
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x

class DePatchify(nn.Module):
    """Learnable depatchification module"""
    def __init__(self, H_img=88, W_img=88, patch_size=4, embed_dim=1, in_chans=256):
        super(DePatchify, self).__init__()
        assert H_img % patch_size == 0
        assert W_img % patch_size == 0
        
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.H, self.W = H_img // patch_size[0], W_img // patch_size[1]
        self.num_patches = self.H * self.W
        
        # Learnable depatchify weights and bias
        self.WE = nn.Parameter(torch.randn(self.embed_dim, self.in_chans, patch_size[0], patch_size[1]))
        self.bE = nn.Parameter(torch.randn(self.embed_dim))
        self.layer_norm = nn.LayerNorm(self.in_chans)
        
        # Initialize weights
        nn.init.trunc_normal_(self.WE, std=0.02)
        nn.init.zeros_(self.bE)
    
    def forward(self, x):
        # x: (B, num_patches, in_chans)
        B, N, C = x.shape
        
        x = self.layer_norm(x)
        
        # Depatchify using learnable weights
        # x: (B, num_patches, in_chans) -> (B, embed_dim, num_patches, patch_h, patch_w)
        y = torch.einsum('ijk,lkmn->iljmn', x, self.WE)
        
        # Add bias
        y = y + self.bE.reshape(1, -1, 1, 1, 1)  # (B, embed_dim, num_patches, patch_h, patch_w)
        
        # Reshape and permute to reconstruct image
        # (B, embed_dim, num_patches, patch_h, patch_w) -> (B, embed_dim, H, W, patch_h, patch_w)
        y = y.reshape(B, self.embed_dim, self.H, self.W, self.patch_size[0], self.patch_size[1])
        y = y.permute(0, 1, 2, 4, 3, 5)  # (B, embed_dim, H, patch_h, W, patch_w)
        
        # Merge patches: (B, embed_dim, H*patch_h, W*patch_w)
        y = y.reshape(B, self.embed_dim, self.H * self.patch_size[0], self.W * self.patch_size[1])
        
        return y

class ViTDensePredictor(nn.Module):
    """Vision Transformer for Dense Prediction (Darcy Flow)"""
    def __init__(
        self, 
        img_size=88, 
        patch_size=4, 
        in_channels=1, 
        out_channels=1,
        embed_dim=256, 
        depth=6, 
        num_heads=8, 
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Depatchify module
        self.depatchify = DePatchify(
            H_img=img_size,
            W_img=img_size,
            patch_size=patch_size,
            embed_dim=out_channels,
            in_chans=embed_dim
        )
        
        self.out_channels = out_channels
        
    def forward(self, x):
        # x: (B, 1, 85, 85)
        B = x.shape[0]
        
        # Pad to 88x88
        x = F.pad(x, (0, 3, 0, 3), mode='reflect')  # (B, 1, 88, 88)
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, 484, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Depatchify to image
        x = self.depatchify(x)  # (B, out_channels, 88, 88)
        
        # Crop back to 85x85
        x = x[:, :, :85, :85]
        
        return x

# ==================== Training Utilities ====================
class Trainer:
    def __init__(
        self, 
        model, 
        train_loader, 
        test_loader, 
        optimizer, 
        scheduler,
        y_normalizer,
        device='cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.y_normalizer = y_normalizer
        self.device = device
        
        self.train_losses = []
        self.test_losses = []
        self.test_l2_errors = []
        
    def train_epoch(self):
        self.model.train()
        train_loss = 0.0
        myloss = LpLoss(size_average=False)
        
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.y_normalizer.cuda()
            
            self.optimizer.zero_grad()
            pred = self.model(x)
            
            # Denormalize for Lp loss computation
            pred_phys = self.y_normalizer.decode(pred)
            y_phys = self.y_normalizer.decode(y)
            
            # Relative Lp loss
            loss = myloss(pred_phys, y_phys)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
        
        return train_loss / ntrain
    
    def evaluate(self):
        self.model.eval()
        test_loss = 0.0
        test_l2 = 0.0
        myloss = LpLoss(size_average=False)
        
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                pred = self.model(x)
                
                # Denormalize for evaluation
                pred_phys = self.y_normalizer.decode(pred)
                y_phys = self.y_normalizer.decode(y)
                
                # Relative Lp loss
                loss = myloss(pred_phys, y_phys)
                test_loss += loss.item()
                
                # Relative L2 error
                l2_error = myloss(pred_phys, y_phys)
                test_l2 += l2_error.item()
        
        return test_loss / ntest, test_l2 / ntest
    
    def train(self, epochs):
        print("Starting training...")
        best_test_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            test_loss, test_l2 = self.evaluate()
            
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.test_l2_errors.append(test_l2)
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Save best model
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(self.model.state_dict(), 'best_vit_darcy.pt')
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Test Loss: {test_loss:.6f} | "
                      f"Test L2: {test_l2:.6f}")
        
        print(f"\nTraining complete! Best test loss: {best_test_loss:.6f}")
        return self.train_losses, self.test_losses, self.test_l2_errors

def plot_results(trainer, x_test, y_test, y_normalizer, device='cuda', n_samples=3):
    """Plot training curves and sample predictions"""
    trainer.model.eval()
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    axes[0].plot(trainer.train_losses, label='Train Loss', alpha=0.7)
    axes[0].plot(trainer.test_losses, label='Test Loss', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training and Test Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # L2 error curve
    axes[1].plot(trainer.test_l2_errors, label='Test Relative L2 Error', color='green', alpha=0.7)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Relative L2 Error')
    axes[1].set_title('Test Error Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot sample predictions
    with torch.no_grad():
        indices = np.random.choice(len(x_test), n_samples, replace=False)
        
        for idx in indices:
            x_sample = x_test[idx:idx+1].to(device)
            y_sample = y_test[idx:idx+1].to(device)
            
            pred = trainer.model(x_sample)
            
            # Denormalize
            x_phys = x_sample[0, 0].cpu().numpy()
            y_phys = y_normalizer.decode(y_sample)[0, 0].cpu().numpy()
            pred_phys = y_normalizer.decode(pred)[0, 0].cpu().numpy()
            
            # Calculate error
            error = np.abs(pred_phys - y_phys)
            rel_l2 = np.linalg.norm(pred_phys - y_phys) / np.linalg.norm(y_phys)
            
            # Plot
            fig, axes = plt.subplots(1, 4, figsize=(20, 4))
            
            im0 = axes[0].imshow(x_phys, cmap='viridis')
            axes[0].set_title('Input: a(x)')
            axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0], fraction=0.046)
            
            im1 = axes[1].imshow(y_phys, cmap='viridis')
            axes[1].set_title('True: u(x)')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046)
            
            im2 = axes[2].imshow(pred_phys, cmap='viridis')
            axes[2].set_title('Predicted: u(x)')
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046)
            
            im3 = axes[3].imshow(error, cmap='hot')
            axes[3].set_title(f'Error (L2: {rel_l2:.4f})')
            axes[3].axis('off')
            plt.colorbar(im3, ax=axes[3], fraction=0.046)
            
            plt.tight_layout()
            plt.savefig(f'prediction_sample_{idx}.png', dpi=150, bbox_inches='tight')
            plt.show()

# ==================== Main Training Script ====================
if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ==================== Data Loading ====================
    TRAIN_PATH = 'darcy/piececonst_r421_N1024_smooth1.mat'
    TEST_PATH = 'darcy/piececonst_r421_N1024_smooth2.mat'
    
    ntrain = 1000
    ntest = 100
    batch_size = 20
    learning_rate = 0.001
    epochs = 100
    step_size = 100
    gamma = 0.5
    
    r = 5  # Subsampling factor
    h = int(((421 - 1) / r) + 1)  # 85
    s = h  # 85
    
    print(f"Loading data... Image size: {s}x{s}")
    
    reader = MatReader(TRAIN_PATH)
    x_train = reader.read_field('coeff')[:ntrain, ::r, ::r][:, :s, :s]
    y_train = reader.read_field('sol')[:ntrain, ::r, ::r][:, :s, :s]
    
    reader.load_file(TEST_PATH)
    x_test = reader.read_field('coeff')[:ntest, ::r, ::r][:, :s, :s]
    y_test = reader.read_field('sol')[:ntest, ::r, ::r][:, :s, :s]
    
    print(f"Data loaded: x_train {x_train.shape}, y_train {y_train.shape}")
    
    # Normalize
    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    
    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)
    # Note: y_test is NOT normalized for evaluation
    
    # Reshape to (B, C, H, W)
    x_train = x_train.reshape(ntrain, 1, s, s)
    x_test = x_test.reshape(ntest, 1, s, s)
    y_train = y_train.reshape(ntrain, 1, s, s)
    y_test = y_test.reshape(ntest, 1, s, s)
    
    # Create dataloaders
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=batch_size,
        shuffle=False
    )
    
    # ==================== Model Setup ====================
    model = ViTDensePredictor(
        img_size=88,          # Padded size (85 -> 88 for 4x4 patches)
        patch_size=4,         # 4x4 patches
        in_channels=1,
        out_channels=1,
        embed_dim=256,        # Embedding dimension
        depth=6,              # Number of transformer blocks
        num_heads=8,          # Attention heads
        mlp_ratio=4.0,
        dropout=0.1
    )
    
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    # ==================== Training ====================
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        y_normalizer=y_normalizer,
        device=device
    )
    
    train_losses, test_losses, test_l2_errors = trainer.train(epochs)
    
    # ==================== Evaluation & Visualization ====================
    print("\nGenerating visualizations...")
    plot_results(trainer, x_test, y_test, y_normalizer, device=device, n_samples=3)
    
    print("\nAll done! Check the saved figures.")