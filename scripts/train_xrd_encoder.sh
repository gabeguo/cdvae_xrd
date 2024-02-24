# python train_xrd_encoder.py \
#     --epochs 100 \
#     --batch_size 256 \
#     --model_path /home/gabeguo/hydra/singlerun/2024-02-17/perov \
#     --data_dir /home/gabeguo/cdvae/data/perov_5 \
#     --lr 1e-4

# python evaluate.py \
#     --model_path /home/gabeguo/hydra/singlerun/2024-02-17/perov \
#     --data_dir /home/gabeguo/cdvae/data/perov_5 \
#     --tasks recon \
#     --xrd

# python -W ignore train_xrd_encoder.py \
#     --epochs 100 \
#     --batch_size 256 \
#     --model_path /home/gabeguo/hydra/singlerun/2024-02-16/mp_20 \
#     --data_dir /home/gabeguo/cdvae/data/mp_20 \
#     --lr 1e-4

# python -W ignore evaluate.py \
#     --model_path /home/gabeguo/hydra/singlerun/2024-02-16/mp_20 \
#     --data_dir /home/gabeguo/cdvae/data/mp_20 \
#     --tasks recon \
#     --xrd

# python train_xrd_encoder.py \
#     --epochs 100 \
#     --batch_size 256 \
#     --model_path /home/gabeguo/hydra/singlerun/2024-02-16/carbon \
#     --data_dir /home/gabeguo/cdvae/data/carbon_24 \
#     --lr 1e-4

# python evaluate.py \
#     --model_path /home/gabeguo/hydra/singlerun/2024-02-16/carbon \
#     --data_dir /home/gabeguo/cdvae/data/carbon_24 \
#     --tasks recon \
#     --xrd
