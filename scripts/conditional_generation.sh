python -W ignore conditional_generation.py \
    --model_path /home/gabeguo/hydra/singlerun/2024-03-12/mp_20 \
    --num_starting_points 100 \
    --lr 0.1 \
    --min_lr 0.001 \
    --l2_penalty 5e-5 \
    --num_tested_materials 20 \
    --label mp_20 \
    --num_gradient_steps 5000 \
    --l1_loss
