python -W ignore conditional_generation.py \
    --model_path /home/gabeguo/hydra/singlerun/2024-04-07/mp_20_sincSmooth10_EXPERIMENTAL_TEST \
    --num_starting_points 100 \
    --num_candidates 5 \
    --lr 0 \
    --min_lr 0 \
    --l2_penalty 0 \
    --num_tested_materials 200 \
    --label _random_baseline_EXPERIMENTAL_ \
    --num_gradient_steps 1 \
    --num_atom_lambda 0.1 \
    --n_step_each 0 \
    --composition_lambda 0.1 \
    --l1_loss \
    --r_min 0 \
    --r_max 30 \
    --output_dir paper_results_PRELIM