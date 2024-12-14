import torch
import torch.nn as nn
from collections import namedtuple

from .helpers import (
    Losses,
)

Sample = namedtuple("Sample", "trajectories values")

def find_nn(data_point, generated):
    
    data_point = data_point.reshape(-1)
    generated = generated.reshape(generated.shape[0], -1)

    dists = torch.sum((generated - data_point)**2, dim=1)
    dists = dists**0.5
    return torch.argmin(dists).item()

class IMLEModel(nn.Module):
    def __init__(
        self,
        model,
        horizon,
        observation_dim,
        action_dim,
        loss_type="IMLE",
        sample_factor=10,
        noise_coef=0.1,
        staleness=20,
        z_dim=32,
        action_weight=1.0,
        loss_discount=1.0,
        loss_weights=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.generator = model
        self.transition_dim = observation_dim + action_dim
        # IMLE Properties
        self.sample_factor = sample_factor
        self.noise_coef = noise_coef
        self.staleness = staleness
        self.z_dim = z_dim
        self.action_dim = action_dim
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    @torch.no_grad()
    def conditional_sample(self, cond, *args, horizon=None, **kwargs):
        """
        Forward pass through the generator.

        Returns:
            torch.Tensor: Generated trajectories [batch_size, horizon, output_dim].
        """
        batch_size = len(cond[0])
        shape = (batch_size, self.horizon, self.transition_dim)
        cond_tensor = torch.stack([cond[key] for key in sorted(cond.keys())], dim=1)
        cond_tensor = cond_tensor.view(cond_tensor.size(0), -1)
        x = torch.randn(shape, device=cond_tensor.device)
        trajectories = self.generator(x, cond_tensor)
        return trajectories

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)

    
    def loss(self, x, cond, epoch=1):
        zs = torch.randn_like(x)
        cond_tensor = torch.stack([cond[key] for key in sorted(cond.keys())], dim=1)
        cond_tensor = cond_tensor.view(cond_tensor.size(0), -1)
        generated = self.generator(zs, cond_tensor).detach()
        nns = torch.tensor([find_nn(d, generated) for d in x], dtype=torch.long, device=x.device)
        imle_nn_z = zs[nns] + torch.randn_like(zs[nns]) * self.noise_coef
        outs = self.generator(imle_nn_z, cond_tensor)
        return self.loss_fn(outs, x)