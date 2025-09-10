import torch

import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

"""
NOTE: you can add as many functions as you need in this file, and for all the classes you can define extra methods if you need
"""

class VarianceScheduler:
    """
    This class is used to keep track of statistical variables used in the diffusion model
    and also adding noise to the data
    """
    def __init__(self, beta_start: float=0.0001, beta_end: float=0.02, num_steps :int=1000, device: torch.device=torch.device('cuda')):
        self.betas = torch.linspace(beta_start, beta_end, num_steps).to(device) # defining the beta variables
        self.alphas = 1. - self.betas # defining the alpha variables
        self.alpha_bars = torch.cumprod(self.alphas, dim=0) # defining the alpha bar variables
        
        # NOTE:Feel free to add to this or modify it as you wish
        self.device = device
        
    def add_noise(self, x: torch.Tensor, timestep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This method receives the input data and the timestep, generates a noise according to the 
        timestep, perturbs the data with the noise, and returns the noisy version of the data and
        the noise itself
        
        Args:
            x (torch.Tensor): input image [B, 1, 28, 28]
            timestep (torch.Tensor): timesteps [B]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: noisy_x [B, 1, 28, 28], noise [B, 1, 28, 28]
        """
        alpha_t_bar = self.alpha_bars[timestep.to(self.device)]
        alpha_t_bar = alpha_t_bar.view(-1, 1, 1, 1).to(self.device)  
        x = x.to(self.device)  
        noise = torch.randn_like(x).to(self.device)
        noisy_x = torch.sqrt(alpha_t_bar) * x + torch.sqrt(1 - alpha_t_bar) * noise
        return noisy_x, noise

class MyBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(MyBlock, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out
    
def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])

    return embedding

# Copied from DDPM_MNIST.ipynb on eClass
class MyUNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100, class_emb_dim=100, num_classes=10):
        super(MyUNet, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        self.class_embed = nn.Embedding(num_classes, class_emb_dim)

        # First half
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            MyBlock((1, 28, 28), 1, 10),
            MyBlock((10, 28, 28), 10, 10),
            MyBlock((10, 28, 28), 10, 10)
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim, 10)
        self.b2 = nn.Sequential(
            MyBlock((10, 14, 14), 10, 20),
            MyBlock((20, 14, 14), 20, 20),
            MyBlock((20, 14, 14), 20, 20)
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

        self.te3 = self._make_te(time_emb_dim, 20)
        self.b3 = nn.Sequential(
            MyBlock((20, 7, 7), 20, 40),
            MyBlock((40, 7, 7), 40, 40),
            MyBlock((40, 7, 7), 40, 40)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, 2, 1),
            nn.SiLU(),
            nn.Conv2d(40, 40, 4, 2, 1)
        )

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            MyBlock((40, 3, 3), 40, 20),
            MyBlock((20, 3, 3), 20, 20),
            MyBlock((20, 3, 3), 20, 40)
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(40, 40, 2, 1)
        )

        self.te4 = self._make_te(time_emb_dim, 80)
        self.b4 = nn.Sequential(
            MyBlock((80, 7, 7), 80, 40),
            MyBlock((40, 7, 7), 40, 20),
            MyBlock((20, 7, 7), 20, 20)
        )

        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, 40)
        self.b5 = nn.Sequential(
            MyBlock((40, 14, 14), 40, 20),
            MyBlock((20, 14, 14), 20, 10),
            MyBlock((10, 14, 14), 10, 10)
        )

        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 20)
        self.b_out = nn.Sequential(
            MyBlock((20, 28, 28), 20, 10),
            MyBlock((10, 28, 28), 10, 10),
            MyBlock((10, 28, 28), 10, 10, normalize=False)
        )

        self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)

    def forward(self, x, t, y):
        # x is (N, 2, 28, 28) (image with positional embedding stacked on channel dimension)
        t = self.time_embed(t)
        y = self.class_embed(y)
        emb = t + y

        n = len(x)
        out1 = self.b1(x + self.te1(emb).reshape(n, -1, 1, 1))  # (N, 10, 28, 28)
        out2 = self.b2(self.down1(out1) + self.te2(emb).reshape(n, -1, 1, 1))  # (N, 20, 14, 14)
        out3 = self.b3(self.down2(out2) + self.te3(emb).reshape(n, -1, 1, 1))  # (N, 40, 7, 7)

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(emb).reshape(n, -1, 1, 1))  # (N, 40, 3, 3)

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 80, 7, 7)
        out4 = self.b4(out4 + self.te4(emb).reshape(n, -1, 1, 1))  # (N, 20, 7, 7)

        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 40, 14, 14)
        out5 = self.b5(out5 + self.te5(emb).reshape(n, -1, 1, 1))  # (N, 10, 14, 14)

        out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 20, 28, 28)
        out = self.b_out(out + self.te_out(emb).reshape(n, -1, 1, 1))  # (N, 1, 28, 28)

        out = self.conv_out(out)

        return out

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )

class NoiseEstimatingNet(nn.Module):
    """
    The implementation of the noise estimating network for the diffusion model
    """
    # feel free to add as many arguments as you need or change the arguments
    def __init__(self, time_emb_dim: int, class_emb_dim: int, num_classes: int=10):
        super().__init__()
        
        # add your codes here
        self.unet = MyUNet(time_emb_dim=time_emb_dim, class_emb_dim=class_emb_dim, num_classes=num_classes)
        
    def forward(self, x: torch.Tensor, timestep: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Estimate the noise given the input image, timestep, and the label
        
        Args:
            x (torch.Tensor): the input (noisy) image [B, 1, 28, 28]
            timestep (torch.Tensor): timestep [B]
            y (torch.Tensor): the corresponding labels for the images [B]

        Returns:
            torch.Tensor: out (the estimated noise) [B, 1, 28, 28]
        """
        
        out = self.unet(x, timestep, y) # get the output of the network
        return out
        
class DiffusionModel(nn.Module):
    """
    The whole diffusion model put together
    """
    def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler):
        """

        Args:
            network (nn.Module): your noise estimating network
            var_scheduler (VarianceScheduler): variance scheduler for getting 
                                the statistical variables and the noisy images
        """
        
        super().__init__()
        
        self.network = network
        self.var_scheduler = var_scheduler
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.float32:
        """
        The forward method for the diffusion model gets the input images and 
        their corresponding labels
        
        Args:
            x (torch.Tensor): the input image [B, 1, 28, 28]
            y (torch.Tensor): labels [B]

        Returns:
            torch.float32: the loss between the actual noises and the estimated noise
        """
        
        # step1: sample timesteps
        timesteps = torch.randint(0, len(self.var_scheduler.betas), (x.size(0),), device=x.device)

        # step2: compute the noisy versions of the input image according to your timesteps
        noisy_x, true_noise = self.var_scheduler.add_noise(x, timesteps)

        # step3: estimate the noises using your noise estimating network
        estimated_noise = self.network(noisy_x, timesteps, y)

        # step4: compute the loss between the estimated noises and the true noises
        loss = F.mse_loss(estimated_noise, true_noise)

        return loss
    
    @torch.no_grad()
    def generate_sample(self, num_images: int, y, device) -> torch.Tensor:
        """
        This method generates as many samples as specified according to the given labels
        
        Args:
            num_images (int): number of images to generate
            y (_type_): the corresponding expected labels of each image
            device (_type_): computation device (e.g. torch.device('cuda')) 

        Returns:
            torch.Tensor: the generated images [num_images, 1, 28, 28]
        """
        
        # add your code here
        T = self.var_scheduler.betas.shape[0] 
        image_shape = (num_images, 1, 28, 28) 
        samples = torch.randn(image_shape, device=device)

        for t in range(T-1, -1, -1):
            epsilon_t = torch.randn(image_shape, device=device)
            alpha_t = self.var_scheduler.alphas[t]
            alpha_bar_t = self.var_scheduler.alpha_bars[t]
            beta_t = self.var_scheduler.betas[t]
            epsilon_theta = self.network(samples, torch.full((num_images,), t, device=device, dtype=torch.long), y)
            samples = ((1 / torch.sqrt(alpha_t)) * (samples - ((beta_t / torch.sqrt(1 - alpha_bar_t)) * epsilon_theta))) + torch.sqrt(beta_t) * epsilon_t

        return samples

def load_diffusion_and_generate():
    device = torch.device('cuda')
    var_scheduler = VarianceScheduler(device=device)  # define your variance scheduler
    network = NoiseEstimatingNet(time_emb_dim=100, class_emb_dim=100, num_classes=10)  # define your noise estimating network
    diffusion = DiffusionModel(network=network, var_scheduler=var_scheduler) # define your diffusion model
    
    # loading the weights of VAE
    diffusion.load_state_dict(torch.load('diffusion.pt'))
    diffusion = diffusion.to(device)
    
    desired_labels = []
    for i in range(10):
        for _ in range(5):
            desired_labels.append(i)

    desired_labels = torch.tensor(desired_labels).to(device)
    generated_samples = diffusion.generate_sample(50, desired_labels, device)
    
    return generated_samples