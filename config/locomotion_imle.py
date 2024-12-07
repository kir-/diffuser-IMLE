import socket

from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('staleness', 'S'),
    ('sample_factor', 'SF'),
    ## value kwargs
    ('discount', 'd'),
]

logbase = 'logs'

base = {
    'imle': {
        ## model
        'model': 'models.TemporalUnetIMLE',
        'imle': 'models.IMLEModel',
        'horizon': 32,
        # 'horizon': 128,
        'action_weight': 10,
        'loss_weights': None,
        'loss_discount': 1,
        'dim_mults': (1, 2, 4, 8),
        'attention': False,
        'renderer': 'utils.MuJoCoRenderer',
        'staleness': 5,  # Number of epochs before updating nearest neighbors
        'sample_factor': 10,  # Number of samples per training point
        'noise_coef': 0.01,  # Noise coefficient for perturbing z-space,
        'z_dim': 32, # Latent Space Dimension

        ## dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'GaussianNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 1000,

        ## serialization
        'logbase': logbase,
        'prefix': 'imle/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 1e6,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 20000,
        'sample_freq': 20000,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda',
        'seed': None,
    },

    'values': {
        'model': 'models.ValueFunctionIMLE',
        # 'imle': 'ValueIMLE',
        'horizon': 32,
        'dim_mults': (1, 2, 4, 8),
        'renderer': 'utils.MuJoCoRenderer',

        ## value-specific kwargs
        'discount': 0.99,
        'termination_penalty': -100,
        'normed': False,

        ## dataset
        'loader': 'datasets.ValueDataset',
        'normalizer': 'GaussianNormalizer',
        'preprocess_fns': [],
        'use_padding': True,
        'max_path_length': 1000,

        ## serialization
        'logbase': logbase,
        'prefix': 'values/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'value_l2',
        'n_train_steps': 200e3,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 0,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda',
        'seed': None,
    },

    'plan': {
        # 'guide': 'sampling.IMLEValueGuide',
        # 'policy': 'sampling.IMLEPolicy',
        'guide': 'sampling.ValueGuide',
        'policy': 'sampling.GuidedPolicy',
        'max_episode_length': 1000,
        'batch_size': 64,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': None,

        ## sample_kwargs
        'n_guide_steps': 2,
        'scale': 0.1,
        't_stopgrad': 2,
        'scale_grad_by_std': True,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/',
        'exp_name': watch(args_to_watch),
        'vis_freq': 100,
        'max_render': 8,

        ## IMLE model
        'horizon': 32,
        'n_diffusion_steps': 20,
        'staleness': 5,  # Sync with model staleness
        'sample_factor': 10,

        ## value function
        'discount': 0.99,

        ## loading
        # 'imle_loadpath': 'f:imle/defaults_H{horizon}_S{staleness}_SF{sample_factor}',
        'diffusion_loadpath': 'f:imle/defaults_H{horizon}_S{staleness}_SF{sample_factor}',
        # 'value_loadpath': 'f:values/defaults_H{horizon}_S{staleness}_SF{sample_factor}_d{discount}',
        'value_loadpath': 'f:values/defaults_H{horizon}_T{n_diffusion_steps}_d{discount}',

        # 'imle_epoch': 'latest',
        'value_epoch': 'latest',
        'diffusion_epoch': 'latest',

        'verbose': True,
        'suffix': '0',
    },
}


#------------------------ overrides ------------------------#


hopper_medium_expert_v2 = {
    'plan': {
        'scale': 0.0001,
        't_stopgrad': 4,
    },
}


halfcheetah_medium_replay_v2 = halfcheetah_medium_v2 = halfcheetah_medium_expert_v2 = {
    'imle': {
        'horizon': 4,
        'dim_mults': (1, 4, 8),
        'attention': True,
    },
    'values': {
        'horizon': 4,
        'dim_mults': (1, 4, 8),
    },
    'plan': {
        'horizon': 4,
        'scale': 0.001,
        't_stopgrad': 4,
    },
}
