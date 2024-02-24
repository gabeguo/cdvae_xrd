# multi-sampling
python -W ignore evaluate.py \
    --model_path /home/gabeguo/hydra/singlerun/2024-02-16/mp_20 \
    --tasks recon \
    --label xrd_threeSamples \
    --num_evals 3
# forcing composition
python -W ignore evaluate.py \
    --model_path /home/gabeguo/hydra/singlerun/2024-02-16/mp_20 \
    --tasks recon \
    --label xrd_givenComposition \
    --force_atom_types