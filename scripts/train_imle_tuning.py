import diffuser.utils as utils
import torch

def wandb_train(config=None):

    #-----------------------------------------------------------------------------#
    #----------------------------------- wandb -----------------------------------#
    #-----------------------------------------------------------------------------#


    wandb_dim_mults = (1, 2, 4, 8)
    wandb_staleness = 5
    wandb_sample_factor = 10
    wandb_noise_coef = 0.01

    wandb_learning_rate = 2e-4



    wandb_n_train_steps = 2e4
    wandb_save_freq = 1e6 # do not save when hyperparameter tuning
    wandb_sample_freq = 1e6 # do not save when hyperparameter tuning


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

    model_config = utils.Config(
        args.model,
        savepath=(args.savepath, 'model_config.pkl'),
        horizon=args.horizon,
        transition_dim=observation_dim + action_dim,
        cond_dim=observation_dim,
        dim_mults=wandb_dim_mults,
        attention=args.attention,
        device=args.device,
    )

    imle_config = utils.Config(
        args.imle,
        savepath=(args.savepath, 'imle_config.pkl'),
        horizon=args.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        sample_factor=wandb_sample_factor,
        noise_coef=wandb_noise_coef,
        staleness=wandb_staleness,
        z_dim=args.z_dim,
        device=args.device,
    )

    trainer_config = utils.Config(
        utils.Trainer,
        savepath=(args.savepath, 'trainer_config.pkl'),
        train_batch_size=args.batch_size,
        train_lr=wandb_learning_rate,
        gradient_accumulate_every=args.gradient_accumulate_every,
        ema_decay=args.ema_decay,
        sample_freq=wandb_sample_freq,
        save_freq=wandb_save_freq,
        label_freq=int(wandb_n_train_steps // args.n_saves),
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

    print('Testing forward...', end=' ', flush=True)
    batch = utils.batchify(dataset[0])
    loss, _ = imle.loss(*batch)
    loss.backward()
    print('✓')

    # #-----------------------------------------------------------------------------#
    # #--------------------------------- main loop ---------------------------------#
    # #-----------------------------------------------------------------------------#

    n_epochs = int(wandb_n_train_steps // args.n_steps_per_epoch)

    for i in range(n_epochs):
        print(f'Epoch {i+1} / {n_epochs} | {args.savepath}')
        trainer.train(n_train_steps=args.n_steps_per_epoch, epoch=i)



wandb_train()