python gen_xrd.py \
    --data_dir /home/gabeguo/cdvae_xrd/data/mp_20_oldSplit/no_xrd \
    --save_dir /home/gabeguo/cdvae_xrd/data/mp_20_oldSplit

python split_data.py \
    --init_data_dir /home/gabeguo/cdvae_xrd/data/mp_20_oldSplit \
    --new_data_dir /home/gabeguo/cdvae_xrd/data/mp_20 \
    --train_percent 0.9 \
    --test_percent 0.025