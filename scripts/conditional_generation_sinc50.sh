python -W ignore conditional_generation.py \
    --model_path /home/tsaidi/Research/cdvae_xrd/hydra/singlerun/2024-04-07/mp_20_sinc_filt_gauss_filt_nanomaterial_size_50 \
    --num_starting_points 100 \
    --num_candidates 5 \
    --lr 0.1 \
    --min_lr 1e-4 \
    --l2_penalty 2e-4 \
    --num_tested_materials 200 \
    --label _sinc50_ \
    --num_gradient_steps 5000 \
    --num_atom_lambda 0.1 \
    --n_step_each 100 \
    --composition_lambda 0.1 \
    --l1_loss \
    --r_min 0 \
    --r_max 30 \
    --output_dir paper_results_PRELIM