import numpy as np
import torch
import torch.nn as nn

import transformers
from .helpers import (
    SinusoidalPosEmb,
)

from diffuser.models.trajectory_gpt2 import GPT2Model

class AdaptiveLayerNorm(nn.Module):
    def __init__(self, hidden_size, time_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_size * 2),  # Output scaling and shifting factors
            nn.SiLU(),
        )

    def forward(self, x, timestep_embedding):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, hidden_size)
            timestep_embedding: Timestep embedding of shape (batch_size, time_dim)
        Returns:
            Tensor of shape (batch_size, seq_length, hidden_size) with adaptive normalization.
        """
        # Standard LayerNorm
        normalized_x = self.layer_norm(x)

        # Compute scaling and shifting factors from timestep embedding
        time_factors = self.time_mlp(timestep_embedding)  # (batch_size, hidden_size * 2)
        scaling_factor, shifting_factor = time_factors.chunk(2, dim=-1)  # Split into scale and shift

        # Reshape for broadcasting over the sequence dimension
        scaling_factor = scaling_factor.unsqueeze(1)  # (batch_size, 1, hidden_size)
        shifting_factor = shifting_factor.unsqueeze(1)  # (batch_size, 1, hidden_size)

        # Apply scaling and shifting
        return normalized_x * scaling_factor + shifting_factor

class DecisionTransformer(nn.Module):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            observation_dim,
            action_dim,
            hidden_size,
            horizon,
            past_horizon,
            max_ep_len=4096,
            action_tanh=True,
            time_dim = 32,
            **kwargs
    ):
        super().__init__() 
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.past_horizon = past_horizon

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        time_dim = time_dim
        self.embed_timestep = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.position_embedding = nn.Embedding(horizon, hidden_size)
        self.embed_state = torch.nn.Linear(self.observation_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.action_dim, hidden_size)

        self.embed_ln = AdaptiveLayerNorm(hidden_size, time_dim)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.observation_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.action_dim)] + ([nn.Tanh()] if action_tanh else []))
        )

    # def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
    def forward(self, x, cond, t, attention_mask=None):
        device = x.device
        batch_size, seq_length, _ = x.shape
        # batch_size, seq_length = states.shape[0], states.shape[1]

        # Split concatenated input into actions and states
        actions = x[:, :, :self.action_dim]
        states = x[:, :, self.action_dim:]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(device)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        timestep_embeddings = self.embed_timestep(t)

        # positional embeddings
        position_ids = torch.arange(0, seq_length, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1).to(device)
        position_embeddings = self.position_embedding(position_ids)
        state_embeddings = state_embeddings + position_embeddings
        action_embeddings = action_embeddings + position_embeddings
        # returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 2*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs, timestep_embeddings)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 2*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        # return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        state_preds = self.predict_state(x[:,1])    # predict next state given state and action
        action_preds = self.predict_action(x[:,0])  # predict next action given state
        trajectories_predicted = torch.cat([action_preds, state_preds], dim=-1)
        return trajectories_predicted 

    # def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
    #     # we don't care about the past rewards in this model

    #     states = states.reshape(1, -1, self.observation_dim)
    #     actions = actions.reshape(1, -1, self.action_dim)
    #     returns_to_go = returns_to_go.reshape(1, -1, 1)
    #     timesteps = timesteps.reshape(1, -1)

    #     if self.horizon is not None:
    #         states = states[:,-self.horizon:]
    #         actions = actions[:,-self.horizon:]
    #         returns_to_go = returns_to_go[:,-self.horizon:]
    #         timesteps = timesteps[:,-self.horizon:]

    #         # pad all tokens to sequence length
    #         attention_mask = torch.cat([torch.zeros(self.horizon-states.shape[1]), torch.ones(states.shape[1])])
    #         attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
    #         states = torch.cat(
    #             [torch.zeros((states.shape[0], self.horizon-states.shape[1], self.observation_dim), device=states.device), states],
    #             dim=1).to(dtype=torch.float32)
    #         actions = torch.cat(
    #             [torch.zeros((actions.shape[0], self.horizon - actions.shape[1], self.action_dim),
    #                          device=actions.device), actions],
    #             dim=1).to(dtype=torch.float32)
    #         returns_to_go = torch.cat(
    #             [torch.zeros((returns_to_go.shape[0], self.horizon-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
    #             dim=1).to(dtype=torch.float32)
    #         timesteps = torch.cat(
    #             [torch.zeros((timesteps.shape[0], self.horizon-timesteps.shape[1]), device=timesteps.device), timesteps],
    #             dim=1
    #         ).to(dtype=torch.long)
    #     else:
    #         attention_mask = None

    #     _, action_preds, return_preds = self.forward(
    #         states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

    #     return action_preds[0,-1]
