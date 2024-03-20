python -W ignore conditional_generation.py \
    --model_path /home/gabeguo/hydra/singlerun/2024-03-17/mp_20_vaeBeta05_lr5e-4 \
    --num_starting_points 150 \
    --lr 0.1 \
    --min_lr 0.001 \
    --l2_penalty 5e-5 \
    --num_tested_materials 20 \
    --label kldBeta_0_05_mp_20_l2_5e-5 \
    --num_gradient_steps 5000 \
    --num_atom_lambda 0.1 \
    --composition_lambda 0.1 \
    --l1_loss

python -W ignore conditional_generation.py \
    --model_path /home/gabeguo/hydra/singlerun/2024-03-17/mp_20_vaeBeta05_lr5e-4 \
    --num_starting_points 150 \
    --lr 0.1 \
    --min_lr 0.001 \
    --l2_penalty 1e-3 \
    --num_tested_materials 20 \
    --label kldBeta_0_05_mp_20_l2_1e-3 \
    --num_gradient_steps 5000 \
    --num_atom_lambda 0.1 \
    --composition_lambda 0.1 \
    --l1_loss

python -W ignore conditional_generation.py \
    --model_path /home/gabeguo/hydra/singlerun/2024-03-17/mp_20_vaeBeta05_lr5e-4 \
    --num_starting_points 150 \
    --lr 0.1 \
    --min_lr 0.001 \
    --l2_penalty 5e-5 \
    --num_tested_materials 20 \
    --label kldBeta_0_05_mp_20_l2_noFormula \
    --num_gradient_steps 5000 \
    --num_atom_lambda 0 \
    --composition_lambda 0 \
    --l1_loss

