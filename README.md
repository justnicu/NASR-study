#  Neural Attention for Symbolic Reasoning - study
This repository looks at ``Learning Where and When to Reason in Neuro-Symbolic Inference`` (ICLR-2023) authored by Cristina Cornelio, Jan Stuehmer, Shell Xu Hu, Timothy Hospedales.

The paper can be found here: [ICLR publication](https://openreview.net/forum?id=en9V5F8PR-).

Their original repo can found on [Cornelio Cristina's github](https://github.com/corneliocristina/NASR) and on [SamsungLabs github](https://github.com/SamsungLabs/NASR).

In the pursuit to understand this exciting subject and to showcase it to our peers we have created the following [report](https://github.com/justnicu/NASR-study/blob/main/A%20Study%20on%20the%20Neural%20Attention%20for%20Symbolic%20Reasoning%20Architecture.pdf) and [presentation](https://github.com/justnicu/NASR-study/blob/main/NASR.pptx).

The Google Colab "version" on this repository can be found [here]().

[Please note that the code and instructions found in this repository were slightly modified from the original in order to make it compatible with a Windows machine without a graphics card with CUDA cores.]

## NASR code and data
To produce the best results (a bit better than the ones presented in the paper) it is recommended to use the pre-trained models in the outputs folder. [Please note that if you retrain the models you will overwrite the pre-trained models.]

## Installation
1 - Follow the instructions found in the ``setup.sh`` file.

2 - [Download SatNet dataset](https://powei.tw/sudoku.zip) and unzip the data (4 files) then place `features.pt`, `features_img.pt`, `labels.pt`, and `perm.pt` in `data/original_data`.
Check that inside the ``data/original_data`` folder there are 7 files: `puzzles0_kaggle`, `puzzles2_17_clue`, `puzzles7_serg_benchmark`, `features.pt`, `features_img.pt`, `labels.pt`, and `perm.pt`.

3 - Install [Prolog](https://www.swi-prolog.org/Download.html). To check if the Prolog installation is succesfull try to call Prolog from terminal by typing ``swipl`` (to exit prolog type ``halt.``).

NOTE: if unable to install prolog we provide another symbolic-solver for Sudoku: "backtrack" (which is slower and may even crash).

NOTE: if you have prolog related issues at runtime try installing pyswip using the following command ``pip install git+https://github.com/yuce/pyswip@master#egg=pyswip``.

4 - Install pytorch. The authors recommend running ``conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia``.

NOTE: if you encounter errors at runtime like I did try ``pip install torchvision torchaudio`` and installing the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Windows).

5 - Run ``python src/sudoku_data.py --solver default`` to generate the data. If there were problems installing prolog use ``--solver backtrack`` (it might take longer to run or even cause crashes). More options/statistics available in ``src/sudoku_data.py``.

NOTE: due to the hardware limitations of my machine this will result in only the ``multiple_sol`` dataset being generated. Modify the ``sudoku_data.py`` script to generate all the datasets. 

Additional Notes:

* When generating the data for `satnet_dataset`: use noise `[0-20]` for training the mask predictor in NASR using the default (transformer) Neuro-Solver. If you want to use Satnet as Neuro-solver, it would be better to use noise `[10-30]` to generate the mask data (since Satnet has worse performance).
* For optimal results, select an amount of noise to introduce in the data for the mask predictor that can resamble the amount of noise expected to be in the output of the Neuro-solver.
* Every time you generate the data (unless you fix a seed) you will obtain different datasets since both the mask predictor data and the image generation are randomly generated. This might slightly influence the performance of the single components (and the overall system). Also the performance of NASR and the single components may slightly differ with different seeds. In that case please adjust the parameters (pos weights/lerning rate etc.) when you train the models for optimal results.
* For the data and code we used in the paper we choose a random seed, and we used the hyperparameter reported below.

## How to run

### Main experiment on multiple_sol dataset

NOTE: 
* The evaluation scripts were changed so that the ``--gpu-id`` argument is largely ignored. The effect is that there is no more GPU acceleration as everything is done by the CPU.
* The ``--workers 0`` argument was added to solve parallelization errors caused by Windows.
* We cannot guarantee that the training scripts work as those have not been properly tested.

| Type | Module | Script |
| --- | --- | --- |
| Train  | Perception | `python src/train_perception.py --batch-size 128 --epochs 100 --lr 1.0 --data multiple_sol --gpu-id 0` |
| Eval  | Perception  | `python src/eval_perception.py --batch-size 128 --data multiple_sol  --gpu-id 0 --workers 0` |
| Train  | Neuro-Solver | `python src/train_transformer_models.py --module solvernn --data multiple_sol --epochs 200 --warmup 10 --batch-size 128 --lr 0.0001 --weight-decay 3e-1 --clip-grad-norm 1 --gpu-id 0` |
| Eval  | Neuro-Solver | `python src/eval_transformer_models.py --module solvernn --data multiple_sol --gpu-id 0 --workers 0` |
| Train  | Mask-Predictor | `python src/train_transformer_models.py --module mask --data multiple_sol --pos-weights 0.01 --epochs 200 --warmup 10 --batch-size 128 --lr 0.0001 --weight-decay 3e-1 --clip-grad-norm 1 --gpu-id 0` |
| Eval | Mask-Predictor | `python src/eval_transformer_models.py --module mask --pos-weights 0.01 --data multiple_sol --gpu-id 0 --workers 0`|
| Eval | NASR without RL| `python src/rl_eval_sudoku.py --nasr pretrained --solver prolog --data multiple_sol --gpu-id 0 --workers 0`|
| Train | NASR with RL | `python src/rl_train_sudoku.py --nasr rl --solver prolog --data multiple_sol --epochs 200 --warmup 10 --batch-size 256 --lr 0.00001  --weight-decay 3e-1 --clip-grad-norm 1 --gpu-id 0`|
| Train | Fine-tune only Mask-Predictor with RL | `python src/rl_train_sudoku.py --nasr rl --train-only-mask --solver prolog --data multiple_sol --epochs 200 --warmup 10 --batch-size 256 --lr 0.00001  --weight-decay 3e-1 --clip-grad-norm 1 --gpu-id 0`|
| Eval | NASR with RL  |`python src/rl_eval_sudoku.py --nasr rl --solver prolog --data multiple_sol --gpu-id 0 --workers 0` |

To use other dataset use the same as above but substituting `big_kaggle` with the choosen dataset name.
The datasets available are: [`big_kaggle`, `minimal_17`, `multiple_sol`, `satnet`].

### Other experiments

| Experiment | Script |
| --- | --- |
| Symbolic Baseline | `python src/baseline.py --data multiple_sol --solver prolog --workers 0` |
| Ablation | (see instruction in `baseline_ablation.py`) `python src/baseline_ablation.py` |
| Efficiency | `python src/efficiency.py  --solver prolog --data multiple_sol --nasr pretrained --gpu-id 0` |
| Blur noise (blur with sigma in `[0.8,0.9]`) | Add `--transform-data blur --transform-data-param 0.8` to Eval-NASR with or without RL |
| Rotation of digits noise (rotation of 45 degree) | Add `--transform-data rotation --transform-data-param 45`  to Eval-NASR with or without RL |
| Attention Maps | python Notebook `src/VisualizeAttention.ipynb`|


### Configurations

**RL configurations**

| Dataset | batch | RL-lr |
| --- |---| --- |
| `satnet` | 200 | 0.00005/0.00001 |
| `big_kaggle` | 200/300 | 0.00001 |
| `minimal_17` | 200 | 0.0001 |
| `multiple_sol` | 200 | 0.0001 |

**Transformers configurations**

* big model: `embed_dim`=192, `depth`=12, `num_heads`=3
* small model: `embed_dim`=192, `depth`=4, `num_heads`=3



