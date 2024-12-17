import diffuser.utils as utils
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
            dataset: str = 'hopper-medium-expert-v2'
            config: str = 'config.locomotion_transformer'

        args = Parser().parse_args('diffusion')


        #-----------------------------------------------------------------------------#
        #---------------------------------- dataset ----------------------------------#
        #-----------------------------------------------------------------------------#

        dataset_config = utils.Config(
            args.loader,
            savepath=(args.savepath, 'dataset_config.pkl'),
            env=args.dataset,
            horizon=args.horizon,
            past_horizon=args.past_horizon,
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
            past_horizon=args.past_horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_size=args.hidden_size,
            # hidden_size=wandb_config.wandb_hidden_size,
            device=args.device,
        )

        diffusion_config = utils.Config(
            args.diffusion,
            savepath=(args.savepath, 'diffusion_config.pkl'),
            horizon=args.horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            # n_timesteps=args.n_diffusion_steps,
            n_timesteps=wandb_config.wandb_n_diffusion_steps,
            loss_type=args.loss_type,
            clip_denoised=args.clip_denoised,
            predict_epsilon=args.predict_epsilon,
            ## loss weighting
            action_weight=args.action_weight,
            loss_weights=args.loss_weights,
            loss_discount=args.loss_discount,
            device=args.device,
        )

        trainer_config = utils.Config(
            utils.Trainer,
            savepath=(args.savepath, 'trainer_config.pkl'),
            train_batch_size=args.batch_size,
            # train_lr=args.learning_rate,
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
        )

        #-----------------------------------------------------------------------------#
        #-------------------------------- instantiate --------------------------------#
        #-----------------------------------------------------------------------------#

        model = model_config()

        diffusion = diffusion_config(model)

        trainer = trainer_config(diffusion, dataset, renderer)


        #-----------------------------------------------------------------------------#
        #------------------------ test forward & backward pass -----------------------#
        #-----------------------------------------------------------------------------#

        utils.report_parameters(model)

        print('Testing forward...', end=' ', flush=True)
        batch = utils.batchify(dataset[0])
        loss, _ = diffusion.loss(*batch)
        loss.backward()
        print('✓')


        #-----------------------------------------------------------------------------#
        #--------------------------------- main loop ---------------------------------#
        #-----------------------------------------------------------------------------#

        n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

        for i in range(n_epochs):
            print(f'Epoch {i} / {n_epochs} | {args.savepath}')
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
    # 'wandb_hidden_size': {
    #     'values': [256, 512, 768, 1024]
    #     },
    'wandb_n_diffusion_steps': {
        'values': [5, 10, 20, 50]
        },
    'wandb_learning_rate': {
          'values': [1e-4, 2e-4, 1e-3]
        },
    }

sweep_config['parameters'] = parameters_dict


sweep_id = wandb.sweep(sweep_config, project="walker2d_transformer")

# wandb_train()
wandb.agent(sweep_id, wandb_train, count=50)