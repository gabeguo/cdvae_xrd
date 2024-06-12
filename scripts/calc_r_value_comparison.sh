python calculate_r_factor_comparison.py \
    --unrefined_input_directory /home/gabeguo/cdvae_xrd/Data_to_Max_05_07_24 \
    --refined_input_directory /home/gabeguo/refined_candidates_05_16_24_rVal/fitted/pred_10/sinc10_and_gaussian_refineUnitCell \
    --output_dir /home/gabeguo/cdvae_xrd/paper_results_PRELIM/refined_candidates_05_16_24_rVal/xrd_comparison/sinc10 \
    --sinc_level 10

python calculate_r_factor_comparison.py \
    --unrefined_input_directory /home/gabeguo/cdvae_xrd/Data_to_Max_05_07_24 \
    --refined_input_directory /home/gabeguo/refined_candidates_05_16_24_rVal/fitted/pred_100/sinc100_and_gaussian_unitCellRefine \
    --output_dir /home/gabeguo/cdvae_xrd/paper_results_PRELIM/refined_candidates_05_16_24_rVal/xrd_comparison/sinc100 \
    --sinc_level 100

python calculate_r_factor_comparison.py \
    --unrefined_input_directory /home/gabeguo/cdvae_xrd/Data_to_Max_05_07_24 \
    --refined_input_directory /home/gabeguo/refined_candidates_05_16_24_rVal/fitted/pred_100/sinc100_and_gaussian_unitCellRefine \
    --output_dir /home/gabeguo/cdvae_xrd/paper_results_PRELIM/refined_candidates_05_16_24_rVal/xrd_comparison/sinc100_WITH_OUTLIER \
    --sinc_level 100 \
    --thresh 2.5