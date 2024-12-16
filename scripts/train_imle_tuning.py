import diffuser.utils as utils
import torch
import wandb




def wandb_train(config=None):

    #-----------------------------------------------------------------------------#
    #----------------------------------- wandb -----------------------------------#
    #-----------------------------------------------------------------------------#

    with wandb.init(config=config):
        wandb_config = wandb.config

        #-----------------------------------------------------------------------------#
        #----------------------------------- setup -----------------------------------#
        #-----------------------------------------------------------------------------#

        class Parser(utils.Parser):
            dataset: str = 'maze2d-large-v1'
            config: str = 'config.maze2d_imle'

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
            cond_dim=2*observation_dim,
            dim_mults=wandb_config.wandb_dim_mults,
            # dim_mults=args.dim_mults,
            attention=args.attention,
            device=args.device,
        )

        imle_config = utils.Config(
            args.imle,
            savepath=(args.savepath, 'imle_config.pkl'),
            horizon=args.horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            sample_factor=wandb_config.wandb_sample_factor,
            noise_coef=wandb_config.wandb_noise_coef,
            # sample_factor=args.sample_factor,
            # noise_coef=args.noise_coef,
            z_dim=args.z_dim,
            device=args.device,
        )

        trainer_config = utils.Config(
            utils.Trainer,
            savepath=(args.savepath, 'trainer_config.pkl'),
            train_batch_size=args.batch_size,
            train_lr=wandb_config.wandb_learning_rate,
            gradient_accumulate_every=args.gradient_accumulate_every,
            ema_decay=args.ema_decay,
            sample_freq=args.sample_freq,
            save_freq=args.save_freq,
            label_freq=int(args.n_train_steps // args.n_saves),
            save_parallel=args.save_parallel,
            results_folder=args.savepath,
            bucket=args.bucket,
            n_reference=args.n_reference,
            n_samples=args.n_samples,
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

        n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

        for i in range(n_epochs):
            print(f'Epoch {i+1} / {n_epochs} | {args.savepath}')
            trainer.train(n_train_steps=args.n_steps_per_epoch)


# ===============


sweep_config = {
    'method': 'random'
    }


metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }

sweep_config['metric'] = metric




parameters_dict = {
    'wandb_dim_mults': {
        'values': [(1, 4), (1, 2, 4), (1, 4, 16), (1, 2, 4, 8), (1, 2, 4, 8, 16)]
        },
    'wandb_sample_factor': {
        'values': [1, 3, 5, 10, 20]
        },
    'wandb_noise_coef': {
          'values': [0.001, 0.01, 0.1]
        },
    'wandb_learning_rate': {
          'values': [1e-4, 2e-4, 1e-3]
        },
    }

sweep_config['parameters'] = parameters_dict


sweep_id = wandb.sweep(sweep_config, project="maze2d_imle")

# wandb_train()
wandb.agent(sweep_id, wandb_train, count=50)
