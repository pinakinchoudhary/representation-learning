import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from timeit import default_timer
from utilities3 import *
from Adam import Adam
from tqdm import tqdm

torch.manual_seed(0)
np.random.seed(0)

################################################################
# Spectral Convolution (from original FNO)
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, 
                            dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


################################################################
# Time Embedding Module
################################################################
class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding"""
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, t):
        # t: (B,) -> (B, dim)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        return self.mlp(emb)


################################################################
# Condition Encoder (encodes a(x) into global conditioning)
################################################################
class ConditionEncoder(nn.Module):
    """Encodes permeability field a(x) into a global conditioning vector"""
    def __init__(self, modes=12, width=32):
        super().__init__()
        
        # Use spectral convolutions to extract features from a(x)
        self.conv0 = SpectralConv2d(1, width, modes, modes)
        self.w0 = nn.Conv2d(1, width, 1)
        
        self.conv1 = SpectralConv2d(width, width, modes, modes)
        self.w1 = nn.Conv2d(width, width, 1)
        
        # Global pooling to get conditioning vector
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, 128)
        )
    
    def forward(self, a):
        # a: (B, 1, H, W)
        x1 = self.conv0(a)
        x2 = self.w0(a)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1).squeeze(-1)  # (B, width)
        cond = self.fc(x)  # (B, 128)
        
        return cond


################################################################
# FiLM Layer (modulates features based on time and condition)
################################################################
class FiLMLayer(nn.Module):
    """Generates scale and shift parameters for FiLM conditioning"""
    def __init__(self, time_dim=128, cond_dim=128, out_dim=32):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(time_dim + cond_dim, out_dim * 4),
            nn.GELU(),
            nn.Linear(out_dim * 4, out_dim * 2)
        )
    
    def forward(self, time_emb, cond_emb):
        # Concatenate time and condition embeddings
        x = torch.cat([time_emb, cond_emb], dim=-1)  # (B, time_dim + cond_dim)
        params = self.mlp(x)  # (B, out_dim * 2)
        
        # Split into scale and shift
        scale, shift = params.chunk(2, dim=-1)  # Each (B, out_dim)
        
        return scale, shift


################################################################
# Flow Matching FNO
################################################################
class FlowMatchingFNO(nn.Module):
    def __init__(self, modes=12, width=32):
        super().__init__()
        
        self.modes = modes
        self.width = width
        self.padding = 9
        
        # Time embedding
        self.time_embed = TimeEmbedding(dim=128)
        
        # Condition encoder (encodes a(x))
        self.cond_encoder = ConditionEncoder(modes=modes, width=width)
        
        # FiLM layers for each FNO layer
        self.film0 = FiLMLayer(128, 128, width)
        self.film1 = FiLMLayer(128, 128, width)
        self.film2 = FiLMLayer(128, 128, width)
        self.film3 = FiLMLayer(128, 128, width)
        
        # Input projection: x_t and a(x) -> width channels
        # Input is [x_t, a(x), grid], so 4 channels total
        self.fc0 = nn.Linear(4, self.width)
        
        # FNO layers
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes, self.modes)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes, self.modes)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes, self.modes)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes, self.modes)
        
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        
        # Output projection
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
    
    def apply_film(self, x, scale, shift):
        # x: (B, C, H, W), scale/shift: (B, C)
        scale = scale.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return scale * x + shift
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    
    def forward(self, x_t, t, a):
        """
        x_t: (B, s, s, 1) - current state at time t
        t: (B,) - time
        a: (B, s, s, 1) - conditioning permeability field
        
        Returns: velocity field v(x_t, t, a)
        """
        
        # Get time embedding
        time_emb = self.time_embed(t)  # (B, 128)
        
        # Get condition embedding from a(x)
        a_channels = a.permute(0, 3, 1, 2)  # (B, 1, s, s)
        cond_emb = self.cond_encoder(a_channels)  # (B, 128)
        
        # Get FiLM parameters for each layer
        scale0, shift0 = self.film0(time_emb, cond_emb)
        scale1, shift1 = self.film1(time_emb, cond_emb)
        scale2, shift2 = self.film2(time_emb, cond_emb)
        scale3, shift3 = self.film3(time_emb, cond_emb)
        
        # Concatenate x_t, a, and grid
        grid = self.get_grid(x_t.shape, x_t.device)
        x = torch.cat((x_t, a, grid), dim=-1)  # (B, s, s, 4)
        
        # Input projection
        x = self.fc0(x)  # (B, s, s, width)
        x = x.permute(0, 3, 1, 2)  # (B, width, s, s)
        x = F.pad(x, [0, self.padding, 0, self.padding])
        
        # FNO Layer 0 with FiLM
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.apply_film(x, scale0, shift0)
        x = F.gelu(x)
        
        # FNO Layer 1 with FiLM
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.apply_film(x, scale1, shift1)
        x = F.gelu(x)
        
        # FNO Layer 2 with FiLM
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.apply_film(x, scale2, shift2)
        x = F.gelu(x)
        
        # FNO Layer 3 with FiLM
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = self.apply_film(x, scale3, shift3)
        
        # Remove padding
        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)  # (B, s, s, width)
        
        # Output projection
        x = self.fc1(x)
        x = F.gelu(x)
        v = self.fc2(x)  # (B, s, s, 1) - velocity field
        
        return v


################################################################
# Flow Matching Loss
################################################################
def flow_matching_loss(model, a, u, device):
    """
    Flow matching loss for interpolation from a(x) to u(x)
    x_t = (1-t)*a + t*u  (linear interpolation)
    v_target = u - a     (constant velocity for linear path)
    """
    B = a.shape[0]
    
    # Sample random time t ~ Uniform(0, 1)
    t = torch.rand(B, device=device)
    
    # Interpolate: x_t = (1-t)*a + t*u
    t_expanded = t.view(B, 1, 1, 1)
    x_t = (1 - t_expanded) * a + t_expanded * u
    
    # Target velocity (derivative of linear interpolation)
    v_target = u - a
    
    # Predict velocity
    v_pred = model(x_t, t, a)
    
    # MSE loss
    loss = F.mse_loss(v_pred, v_target)
    
    return loss


################################################################
# ODE Integration for Inference
################################################################
@torch.no_grad()
def integrate_ode(model, a, num_steps=50, device=None):
    """
    Integrate ODE from t=0 to t=1
    dx/dt = v(x, t, a), x(0) = a
    """
    if device is None:
        device = a.device
    
    B = a.shape[0]
    x = a.clone()
    dt = 1.0 / num_steps
    
    # Euler integration
    for i in range(num_steps):
        t = torch.full((B,), i * dt, device=device)
        v = model(x, t, a)
        x = x + dt * v
    
    return x


################################################################
# Data Loading
################################################################
TRAIN_PATH = 'darcy/piececonst_r421_N1024_smooth1.mat'
TEST_PATH = 'darcy/piececonst_r421_N1024_smooth2.mat'

ntrain = 1000
ntest = 100
batch_size = 20
learning_rate = 0.001
epochs = 500
step_size = 100
gamma = 0.5

modes = 12
width = 32
r = 5
h = int(((421 - 1)/r) + 1)
s = h

reader = MatReader(TRAIN_PATH)
x_train = reader.read_field('coeff')[:ntrain,::r,::r][:,:s,:s]
y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s]

reader.load_file(TEST_PATH)
x_test = reader.read_field('coeff')[:ntest,::r,::r][:,:s,:s]
y_test = reader.read_field('sol')[:ntest,::r,::r][:,:s,:s]

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

x_train = x_train.reshape(ntrain,s,s,1)
x_test = x_test.reshape(ntest,s,s,1)
y_train = y_train.reshape(ntrain,s,s,1)
y_test = y_test.reshape(ntest,s,s,1)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train), 
    batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_test, y_test), 
    batch_size=batch_size, shuffle=False
)


################################################################
# Training
################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FlowMatchingFNO(modes=modes, width=width).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)
y_normalizer.cuda()

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_loss = 0.0
    
    for a, u in train_loader:
        a, u = a.to(device), u.to(device)
        
        optimizer.zero_grad()
        loss = flow_matching_loss(model, a, u, device)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    scheduler.step()
    
    # Validation
    if (ep + 1) % 10 == 0:
        model.eval()
        test_l2 = 0.0
        
        with torch.no_grad():
            for a, u in test_loader:
                a, u = a.to(device), u.to(device)
                
                # Integrate ODE from a to u
                u_pred = integrate_ode(model, a, num_steps=50, device=device)
                
                # Denormalize
                u_pred = y_normalizer.decode(u_pred.reshape(batch_size, s, s))
                u = y_normalizer.decode(u.reshape(batch_size, s, s))
                
                # Compute relative L2 error
                test_l2 += myloss(u_pred.view(batch_size, -1), u.view(batch_size, -1)).item()
        
        train_loss /= len(train_loader)
        test_l2 /= ntest
        t2 = default_timer()
        
        print(f'Epoch {ep+1}/{epochs}, Time: {t2-t1:.2f}s, Train Loss: {train_loss:.6f}, Test L2: {test_l2:.6f}')

print("Training complete!")