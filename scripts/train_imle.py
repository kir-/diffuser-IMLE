import diffuser.utils as utils
import torch

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'hopper-medium-expert-v2'
    config: str = 'config.locomotion_imle'

args = Parser().parse_args('imle')


#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
)

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    env=args.dataset,
)

dataset = dataset_config()
renderer = render_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

# Calculate input_dim and output_dim based on previous suggestions
input_dim = args.z_dim + observation_dim  # z_dim + state_dim (s_t)
output_dim = observation_dim + action_dim  # s_{t+1:t+f}, a_{t:t+f}
cond_dim = observation_dim  # Since cond is s_t

# Remove 'transition_dim' as it's not used in the model initialization
model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    input_dim=input_dim,
    output_dim=output_dim,
    cond_dim=cond_dim,
    dim_mults=args.dim_mults,
    device=args.device,
)

imle_config = utils.Config(
    args.imle,
    savepath=(args.savepath, 'imle_config.pkl'),
    horizon=args.horizon,
    sample_factor=args.sample_factor,
    noise_coef=args.noise_coef,
    z_dim=args.z_dim,
    device=args.device,
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config()

imle = imle_config(model)

trainer = trainer_config(imle, dataset, renderer)

#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

utils.report_parameters(model)

def batchify(batch):
    """
    Converts a Batch object into tensors for model input.

    Args:
        batch: A Batch object with 'trajectories' and 'conditions'.

    Returns:
        x: Tensor of trajectories [batch_size, horizon, feature_dim].
        s_t: Tensor of initial states [batch_size, cond_dim].
    """
    # Extract trajectories and conditions from the batch
    trajectory = batch.trajectories  # NumPy array: [horizon, feature_dim]
    s_t = batch.conditions[0]        # NumPy array: [cond_dim]

    # Convert to tensors and add batch dimension
    x = torch.tensor(trajectory, dtype=torch.float32).unsqueeze(0)  # [1, horizon, feature_dim]
    s_t = torch.tensor(s_t, dtype=torch.float32).unsqueeze(0)       # [1, cond_dim]

    return x, s_t

print('Testing forward...', end=' ', flush=True)
x_batch, s_t_batch = batchify(dataset[0])

# Move tensors to the correct device
x_batch = x_batch.to(args.device)
s_t_batch = s_t_batch.to(args.device)

# Compute loss and perform backward pass
loss = imle.loss(x_batch, s_t_batch)
loss.backward()
print('✓')

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i+1} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)
