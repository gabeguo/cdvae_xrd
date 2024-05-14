# *XRDnet*: *Ab Initio* Nanostructure Solutions from PXRD via Score-Based Generative Modeling

This is the repository for *XRDnet*, the world's first end-to-end nanostructure solver from powder x-ray diffraction (PXRD) patterns.

All code blocks assume you start from this directory.

## Requirements

<span style="color:red">tbd, need to add requirements.txt</span>

## Data Generation

<span style="color:red">tbd, need to figure out how to upload</span>

## Training

This trains the $10 &#8491$ and $100 &#8491$ models. They will be saved under `../hydra/singlerun/[today's date]`. 

On a single GeForce RTX 3090 (24 GB), each model should take about one day to train.

```
cd scripts
CUDA_VISIBLE_DEVICES=x bash train_mp20_model_sinc10.sh
CUDA_VISIBLE_DEVICES=x bash train_mp20_model_sinc100.sh
```

## Evaluation

You will have to change `--model_path` inside each script to have the appropriate home directory (rather than `/home/gabeguo/`) and date (rather than `2024-04-07`). 

On a single GeForce RTX 3090 (24 GB), each evaluation (per model) should take about one day to conduct.

### MP-20

#### *XRDnet*

```
cd scripts
CUDA_VISIBLE_DEVICES=x bash conditional_generation_sinc10.sh
CUDA_VISIBLE_DEVICES=x bash conditional_generation_sinc100.sh
```

#### *Semi-Random Baseline*

```
cd scripts
CUDA_VISIBLE_DEVICES=x bash conditional_generation_random_baseline_sinc10.sh
CUDA_VISIBLE_DEVICES=x bash conditional_generation_random_baseline_sinc100.sh
```

#### *Latent Space Search Baseline*
```
cd scripts
CUDA_VISIBLE_DEVICES=x bash conditional_generation_sinc10_baseline_noOpt.sh
CUDA_VISIBLE_DEVICES=x bash conditional_generation_sinc100_baseline_noOpt.sh
```

### Experimentally Collected Data

#### Getting Correct Configs

Your file directory should look something like this:
```
cdvae_xrd/
  ... [some stuff here] ...
hydra/singlerun/
  [whatever date you trained model on]/
    mp_20_sinc10/
      .hydra/
        config.yaml
        hydra.yaml
        overrides.yaml
      hparams.yaml
      ... [other stuff here] ...
    mp_20_sinc100/
      ... [same stuff here] ...
```

Run the following code (assuming you are in `cdvae_xrd`) to create the proper evaluation setup for experimental data:
```
cd ../hydra/singlerun
cp mp_20_sinc10 mp_20_sinc10_EXPERIMENTAL_TEST
```

Now, go into `mp_20_sinc10_experimental/.hydra/config.yaml` and change line 7 to be:
```
root_path: ${oc.env:PROJECT_ROOT}/data/experimental_xrd
```
from
~~root_path: ${oc.env:PROJECT_ROOT}/data/mp_20~~

Do exactly the same change for `mp_20_sinc10_experimental/hparams.yaml`.

#### Running the Evals

Again, remember to change `--model_path` inside each script to have the appropriate home directory (rather than `/home/gabeguo/`) and date (rather than `2024-04-07`). 

This should only take a few hours at most, due to there being fewer experimental PXRD patterns.

```
cd scripts
CUDA_VISIBLE_DEVICES=x bash conditional_generation_experimental.sh
CUDA_VISIBLE_DEVICES=x bash conditional_generation_baseline_noOpt.sh
CUDA_VISIBLE_DEVICES=x bash conditional_generation_random_baseline_experimental.sh
```

### Getting R-Factors

As before, in the `__main__` part, change the home directory from `/home/gabeguo/` to whatever your home directory is. 

This should take less than an hour.

```
cd scripts
python calculate_xrd_patterns_post_hoc.py
python calculate_r_factor_post_hoc.py
```

### Calculating Results by Crystal System

Reiterating (as you've already guessed), in `extract_results_by_crystal_system.sh`, change the home directory from `/home/gabeguo/` to whatever your home directory is. 

This should take less than an hour.

```
cd scripts
bash extract_results_by_crystal_system.sh
```
