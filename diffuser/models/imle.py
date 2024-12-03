import torch
import torch.nn as nn
from collections import namedtuple

Sample = namedtuple("Sample", "trajectories values")

class IMLEModel(nn.Module):
    def __init__(
        self,
        model,
        horizon,
        observation_dim,
        action_dim,
        loss_type="l2",
        sample_factor=10,
        noise_coef=0.1,
        staleness=20,
        z_dim=32,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.generator = model  # IMLE Unet

        # IMLE Properties
        self.sample_factor = sample_factor
        self.noise_coef = noise_coef
        self.staleness = staleness
        self.z_dim = z_dim

        self.loss_fn = nn.MSELoss() if loss_type == "l2" else nn.L1Loss()

    def forward(self, latents, s_t):
        """
        Forward pass through the generator.

        Args:
            latents (torch.Tensor): Latent variables [batch_size, horizon, z_dim].
            s_t (torch.Tensor): Conditioning variable [batch_size, cond_dim].

        Returns:
            torch.Tensor: Generated trajectories [batch_size, horizon, output_dim].
        """
        return self.generator(latents, s_t)

    def loss(self, x, s_t):
        """
        Compute the IMLE loss.

        Args:
            x (torch.Tensor): Ground truth trajectories [batch_size, horizon, output_dim].
            s_t (torch.Tensor): Conditioning states [batch_size, cond_dim].

        Returns:
            torch.Tensor: Computed loss.
        """
        batch_size = x.size(0)
        n_samples = batch_size * self.sample_factor

        # Generate latent samples
        latents = torch.randn(
            n_samples, self.horizon, self.z_dim, device=x.device
        )  # [n_samples, horizon, z_dim]

        # Expand s_t to match n_samples
        s_t_expanded = s_t.unsqueeze(1).expand(-1, self.sample_factor, -1)
        s_t_expanded = s_t_expanded.reshape(
            n_samples, -1
        )  # [n_samples, cond_dim]

        # Generate samples conditioned on s_t
        generated = self.generator(latents, s_t_expanded)  # [n_samples, horizon, output_dim]

        # Reshape x for distance computation
        x_expanded = x.unsqueeze(1).expand(-1, self.sample_factor, -1, -1)
        x_expanded = x_expanded.reshape(
            n_samples, self.horizon, x.size(2)
        )  # [n_samples, horizon, output_dim]

        # Compute distances between x and generated samples
        distances = ((x_expanded - generated) ** 2).sum(dim=[1, 2])  # [n_samples]

        # Reshape distances to [batch_size, sample_factor]
        distances = distances.view(batch_size, self.sample_factor)  # [batch_size, sample_factor]

        # Find nearest neighbors
        nns = distances.argmin(dim=1)  # [batch_size]

        # Calculate indices for nearest latents
        indices = nns + torch.arange(batch_size, device=x.device) * self.sample_factor  # [batch_size]

        # Get nearest latents
        nearest_latents = latents[indices]  # [batch_size, horizon, z_dim]

        # Perturb latents
        noise = torch.randn_like(nearest_latents) * self.noise_coef  # [batch_size, horizon, z_dim]
        perturbed_latents = nearest_latents + noise  # [batch_size, horizon, z_dim]

        # Generate outputs using perturbed latents and s_t
        outputs = self.generator(perturbed_latents, s_t)  # [batch_size, horizon, output_dim]

        # Compute loss
        loss = self.loss_fn(outputs, x)  # [1]

        return loss

class ValueIMLE(IMLEModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def loss(self, x, s_t, target, t):
        """
        Compute the loss for ValueIMLE, potentially incorporating time step `t`.

        Args:
            x (torch.Tensor): Ground truth trajectories [batch_size, horizon, output_dim].
            s_t (torch.Tensor): Conditioning states [batch_size, cond_dim].
            target (torch.Tensor): Target values for loss computation.
            t (torch.Tensor or float): Time step.

        Returns:
            torch.Tensor: Computed loss.
        """
        # Implement your custom loss computation here
        # For example, if you're incorporating time steps or additional target values

        # You might generate outputs using the parent class's methods
        loss = super().loss(x, s_t)
        # Then adjust the loss based on `target` and `t` as needed

        return loss
