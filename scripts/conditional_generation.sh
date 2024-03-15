python -W ignore conditional_generation.py \
    --model_path /home/gabeguo/hydra/singlerun/2024-03-12/mp_20 \
    --num_starting_points 150 \
    --lr 0.1 \
    --min_lr 0.001 \
    --l2_penalty 5e-5 \
    --num_tested_materials 20 \
    --label mp_20_tryCompositionInfo \
    --num_gradient_steps 5000 \
    --num_atom_lambda 0.1 \
    --composition_lambda 0.1 \
    --l1_loss

python -W ignore conditional_generation.py \
    --model_path /home/gabeguo/hydra/singlerun/2024-03-12/mp_20 \
    --num_starting_points 150 \
    --lr 0.1 \
    --min_lr 0.001 \
    --l2_penalty 5e-5 \
    --num_tested_materials 20 \
    --label mp_20_comp1e2 \
    --num_gradient_steps 5000 \
    --num_atom_lambda 0.1 \
    --composition_lambda 1e2 \
    --l1_loss

python -W ignore conditional_generation.py \
    --model_path /home/gabeguo/hydra/singlerun/2024-03-12/mp_20 \
    --num_starting_points 150 \
    --lr 0.1 \
    --min_lr 0.001 \
    --l2_penalty 5e-5 \
    --num_tested_materials 20 \
    --label mp_20_comp1e-4 \
    --num_gradient_steps 5000 \
    --num_atom_lambda 0.1 \
    --composition_lambda 1e-4 \
    --l1_loss

python -W ignore conditional_generation.py \
    --model_path /home/gabeguo/hydra/singlerun/2024-03-12/mp_20 \
    --num_starting_points 150 \
    --lr 0.1 \
    --min_lr 0.001 \
    --l2_penalty 5e-5 \
    --num_tested_materials 20 \
    --label mp_20_noCompositionOpt \
    --num_gradient_steps 5000 \
    --num_atom_lambda 0.1 \
    --composition_lambda 0 \
    --l1_loss