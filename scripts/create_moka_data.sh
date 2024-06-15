python gen_xrd.py \
    --data_dir /home/gabeguo/cdvae_xrd/data/mp_20_oldSplit/no_xrd \
    --save_dir /home/gabeguo/cdvae_xrd/data/mp_20_oldSplit_moka \
    --wave_source MoKa

python split_data.py \
    --init_data_dir /home/gabeguo/cdvae_xrd/data/mp_20_oldSplit_moka \
    --new_data_dir /home/gabeguo/cdvae_xrd/data/mp_20_moka \
    --train_percent 0.9 \
    --test_percent 0.025