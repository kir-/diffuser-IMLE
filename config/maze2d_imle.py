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

plan_args_to_watch = [
    ('prefix', ''),
    ##
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('value_horizon', 'V'),
    ('discount', 'd'),
    ('normalizer', ''),
    ('batch_size', 'b'),
    ##
    ('conditional', 'cond'),
]

logbase = 'logs'

base = {
    'imle': {
        ## model
        'model': 'models.TemporalUnetIMLE',
        'imle': 'models.IMLEModel',
        'horizon': 256,
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        # 'dim_mults': (1, 4, 8),
        'dim_mults': (1, 2, 4, 8, 16),
        'attention': False,
        'renderer': 'utils.Maze2dRenderer',
        'staleness': 5,  # Number of epochs before updating nearest neighbors
        # 'sample_factor': 10,  # Number of samples per training point
        'sample_factor': 20,
        'noise_coef': 0.01,  # Noise coefficient for perturbing z-space,
        'z_dim': 32, # Latent Space Dimension

        ## dataset
        'loader': 'datasets.GoalDataset',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': ['maze2d_set_terminals'],
        'clip_denoised': True,
        'use_padding': False,
        'max_path_length': 40000,

        ## serialization
        'logbase': logbase,
        'prefix': 'imle/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        # 'n_steps_per_epoch': 10000,
        'n_steps_per_epoch': 1000,
        'loss_type': 'l2',
        'n_train_steps': 2e6,
        # 'n_train_steps': 2e4,
        'batch_size': 32,
        # 'learning_rate': 2e-4,
        'learning_rate': 0.0001,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        # 'save_freq': 1000,
        'save_freq': 20000,
        'sample_freq': 1000,
        'n_saves': 50,
        'save_parallel': False,
        'n_reference': 50,
        'n_samples': 10,
        'bucket': None,
        'device': 'cuda',
    },

    'plan': {
        'batch_size': 1,
        'device': 'cuda',

        ## IMLE
        'horizon': 384,
        'n_diffusion_steps': 256,
        'normalizer': 'LimitsNormalizer',
        'staleness': 5,  # Sync with model staleness
        # 'sample_factor': 10,
        'sample_factor': 20,

        ## serialization
        'vis_freq': 10,
        'logbase': 'logs',
        'prefix': 'plans/release',
        'exp_name': watch(plan_args_to_watch),
        'suffix': '0',

        'conditional': False,

        ## loading
        'diffusion_loadpath': 'f:imle/defaults_H{horizon}_S{staleness}_SF{sample_factor}',
        'diffusion_epoch': 'latest',

    },
}


#------------------------ overrides ------------------------#


'''
    maze2d maze episode steps:
        umaze: 150
        medium: 250
        large: 600
'''

maze2d_umaze_v1 = {
    'imle': {
        'horizon': 128,
    },
    'plan': {
        'horizon': 128,
    },
}

maze2d_large_v1 = {
    'imle': {
        'horizon': 384,
    },
    'plan': {
        'horizon': 384,
    },
}
