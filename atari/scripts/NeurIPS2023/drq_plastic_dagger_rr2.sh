cd ../..

python run_parallel.py \
    --group_name plastic_dagger \
    --exp_name drq_plastic_dagger_rr2 \
    --config_name drq \
    --seeds 0 1 2 3 4 \
    --num_games 26 \
    --num_devices 8 \
    --num_exp_per_device 1 \
    --overrides agent.optimize_per_env_step=2 \
                agent.reset_type=llf \
                agent.reset_target=True \
                agent.reset_per_optimize_step=40000 \
                model.backbone.normalization=layernorm \
                ++agent.shrink_perturb=False 

