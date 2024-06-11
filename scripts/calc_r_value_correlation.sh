python calculate_r_value_correlation.py \
    --unrefined_json /home/gabeguo/cdvae_xrd/paper_results_PRELIM/_sinc10_replicate/r_values.json \
    --refined_txt /home/gabeguo/refined_candidates_05_16_24_rVal/fitted/pred_10/sinc10_and_gaussian/Rw_values.txt \
    --save_dir /home/gabeguo/cdvae_xrd/paper_results_PRELIM/05_16_24_rw_refinement/sinc10/atomic_refinement \
    --curated_candidates_folder /home/gabeguo/cdvae_xrd/Data_to_Max_05_07_24 \
    --sinc 10 \
    --thresh_x 1.4 \
    --thresh_y 1.4

python calculate_r_value_correlation.py \
    --unrefined_json /home/gabeguo/cdvae_xrd/paper_results_PRELIM/_sinc10_replicate/r_values.json \
    --refined_txt /home/gabeguo/refined_candidates_05_16_24_rVal/fitted/pred_10/sinc10_and_gaussian_refineUnitCell/Rw_values.txt \
    --save_dir /home/gabeguo/cdvae_xrd/paper_results_PRELIM/05_16_24_rw_refinement/sinc10/lattice_refinement \
    --curated_candidates_folder /home/gabeguo/cdvae_xrd/Data_to_Max_05_07_24 \
    --sinc 10 \
    --thresh_x 1.4 \
    --thresh_y 1.4

python calculate_r_value_correlation.py \
    --unrefined_json /home/gabeguo/cdvae_xrd/paper_results_PRELIM/_sinc100_/r_values.json \
    --refined_txt /home/gabeguo/refined_candidates_05_16_24_rVal/fitted/pred_100/sinc100_and_gaussian/Rw_values.txt \
    --save_dir /home/gabeguo/cdvae_xrd/paper_results_PRELIM/05_16_24_rw_refinement/sinc100/atomic_refinement \
    --curated_candidates_folder /home/gabeguo/cdvae_xrd/Data_to_Max_05_07_24 \
    --sinc 100 \
    --thresh_x 1.4 \
    --thresh_y 1.4

python calculate_r_value_correlation.py \
    --unrefined_json /home/gabeguo/cdvae_xrd/paper_results_PRELIM/_sinc100_/r_values.json \
    --refined_txt /home/gabeguo/refined_candidates_05_16_24_rVal/fitted/pred_100/sinc100_and_gaussian_unitCellRefine/Rw_values.txt \
    --save_dir /home/gabeguo/cdvae_xrd/paper_results_PRELIM/05_16_24_rw_refinement/sinc100/lattice_refinement \
    --curated_candidates_folder /home/gabeguo/cdvae_xrd/Data_to_Max_05_07_24 \
    --sinc 100 \
    --thresh_x 1.4 \
    --thresh_y 1.4

python calculate_r_value_correlation.py \
    --unrefined_json /home/gabeguo/cdvae_xrd/paper_results_PRELIM/_sinc100_/r_values.json \
    --refined_txt /home/gabeguo/refined_candidates_05_16_24_rVal/fitted/pred_100/sinc100_and_gaussian_unitCellRefine/Rw_values.txt \
    --save_dir /home/gabeguo/cdvae_xrd/paper_results_PRELIM/05_16_24_rw_refinement/sinc100/lattice_refinement_NO_OUTLIERS \
    --curated_candidates_folder /home/gabeguo/cdvae_xrd/Data_to_Max_05_07_24 \
    --sinc 100 \
    --thresh_x 2.5 \
    --thresh_y 2.5
