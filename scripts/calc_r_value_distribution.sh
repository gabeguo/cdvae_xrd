python calculate_r_value_distribution.py \
    --r_values_pxrdnet_json /home/gabeguo/cdvae_xrd/paper_results_PRELIM/_sinc10_/r_values.json \
    --r_values_latentSearch_json /home/gabeguo/cdvae_xrd/paper_results_PRELIM/_sinc10_baseline_noOpt/r_values.json \
    --r_values_random_json /home/gabeguo/cdvae_xrd/paper_results_PRELIM/_random_baseline_sinc10_/r_values.json \
    --output_dir /home/gabeguo/cdvae_xrd/paper_results_PRELIM/r_value_distribution/sinc10_

python calculate_r_value_distribution.py \
    --r_values_pxrdnet_json /home/gabeguo/cdvae_xrd/paper_results_PRELIM/_sinc100_/r_values.json \
    --r_values_latentSearch_json /home/gabeguo/cdvae_xrd/paper_results_PRELIM/_sinc100_baseline_noOpt/r_values.json \
    --r_values_random_json /home/gabeguo/cdvae_xrd/paper_results_PRELIM/_random_baseline_sinc100_/r_values.json \
    --output_dir /home/gabeguo/cdvae_xrd/paper_results_PRELIM/r_value_distribution/sinc100_ \
    --disable_y_label
