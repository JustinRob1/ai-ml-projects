import torch

import torch.nn as nn
import torch.nn.functional as F

"""
NOTE: you can add as many functions as you need in this file, and for all the classes you can define extra methods if you need
"""

class VAE(nn.Module):
  def __init__(self, hidden_dim, latent_dim, class_emb_dim, num_classes=10):
    super().__init__()
    
    # implement your encoder here
    self.encoder = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(7 * 7 * 64, hidden_dim),
        nn.ReLU()
    )

    # defining the network to estimate the mean
    self.mu_net = nn.Linear(hidden_dim + class_emb_dim, latent_dim) # implement your mean estimation module here
    
    # defining the network to estimate the log-variance
    self.logvar_net = nn.Linear(hidden_dim + class_emb_dim, latent_dim) # implement your log-variance estimation here
    
    # defining the class embedding module
    self.class_embedding = nn.Embedding(num_classes, class_emb_dim) # implement your class-embedding module here

    # defining the decoder here
    self.decoder = nn.Sequential(
        nn.Linear(latent_dim + class_emb_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 7 * 7 * 64),
        nn.ReLU(),
        nn.Unflatten(1, (64, 7, 7)),
        nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.Sigmoid()
    ) # implement your decoder here

  def forward(self, x: torch.Tensor, y: torch.Tensor):
    """
    Args:
        x (torch.Tensor): image [B, 1, 28, 28]
        y (torch.Tensor): labels [B]
        
    Returns:
        reconstructed: image [B, 1, 28, 28]
        mu: [B, latent_dim]
        logvar: [B, latent_dim]
    """
    
    # implement your forward function here
    x = self.encoder(x)
    y_emb = self.class_embedding(y)
    x = torch.cat([x, y_emb], dim=1)
    mu = self.mu_net(x)
    logvar = self.logvar_net(x)
    z = self.reparameterize(mu, logvar)
    reconstructed = self.decoder(torch.cat([z, y_emb], dim=1))
    return reconstructed, mu, logvar

  def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
    """
    applies the reparameterization trick
    """
    std = torch.exp(0.5 * logvar) 
    eps = torch.randn_like(std) 
    new_sample = mu + eps * std # using the mu and logvar generate a sample
    return new_sample

  def kl_loss(self, mu, logvar):
    """
    calculates the KL divergence between a normal distribution with mean "mu" and
    log-variance "logvar" and the standard normal distribution (mean=0, var=1)
    """
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # calculate the kl-div using mu and logvar
    return kl_div

  def get_loss(self, x: torch.Tensor, y: torch.Tensor):
    """
    given the image x, and the label y calculates the prior loss and reconstruction loss
    """
    reconstructed, mu, logvar = self.forward(x, y)

    # reconstruction loss
    # compute the reconstruction loss here using the "reconstructed" variable above
    recons_loss = F.binary_cross_entropy(reconstructed, x, reduction='sum')

    # prior matching loss
    prior_loss = self.kl_loss(mu, logvar)

    return recons_loss, prior_loss

  @torch.no_grad()
  def generate_sample(self, num_images: int, y, device):
    """
    generates num_images samples by passing noise to the model's decoder
    if y is not None (e.g., y = torch.tensor([1, 2, 3]).to(device)) the model
    generates samples according to the specified labels
    
    Returns:
        samples: [num_images, 1, 28, 28]
    """
    
    # sample from noise, find the class embedding and use both in the decoder to generate new samples
    z = torch.randn(num_images, self.mu_net.out_features).to(device)
    z = torch.cat([z, self.class_embedding(y)], dim=1)
    samples = self.decoder(z).view(num_images, 1, 28, 28)
    return samples

def load_vae_and_generate():
    device = torch.device('cuda')
    # define your VAE model according to your implementation above
    vae = VAE(hidden_dim=128, latent_dim=64, class_emb_dim=10).to(device)
    
    # loading the weights of VAE
    vae.load_state_dict(torch.load('vae.pt'))
    vae = vae.to(device)
    
    desired_labels = []
    for i in range(10):
        for _ in range(5):
            desired_labels.append(i)

    desired_labels = torch.tensor(desired_labels).to(device)
    generated_samples = vae.generate_sample(50, desired_labels, device)
    
    return generated_samples