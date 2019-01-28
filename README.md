# Pruning neural networks: is it time to nip it in the bud?

Code used to reproduce experiments in https://arxiv.org/abs/1810.04622.

To prune, we fill our networks with custom`MaskBlocks`, which are manipulated using `Pruner` in funcs.py. There will certainly be a better way to do this, but we leave this an exercise to someone who can code much better than we can.
## Setup
This is best done in a clean conda environment:

```
conda create -n prunes python=3.6
conda activate prunes
conda install pytorch torchvision -c pytorch
```

## Repository layout
-`train.py`: contains all of the code for training large models from scratch and for training pruned models from scratch  
-`prune.py`: contains the code for pruning trained models   
-`funcs.py`: contains useful pruning functions and any functions we used commonly   

## CIFAR Experiments
First, you will need some initial models. 

To train a WRN-40-2:
```
python train.py --net='res' --depth=40 --width=2.0 --data_loc=<path-to-data> --save_file='res'
```

The default arguments of train.py are suitable for training WRNs. The following trains a DenseNet-BC-100 (k=12) with its default hyperparameters:

```
python train.py --net='dense' --depth=100 --data_loc=<path-to-data> --save_file='dense' --no_epochs 300 -b 64 --epoch_step '[150,225]' --weight_decay 0.0001 --lr_decay_ratio 0.1
```

These will automatically save checkpoints to the `checkpoints` folder.
 
 
 
### Pruning 
Once training is finished, we can prune our networks using prune.py (defaults are set to WRN pruning, so extra arguments are needed for DenseNets)  
```
python prune.py --net='res'   --data_loc=<path-to-data> --base_model='res' --save_file='res_fisher'
python prune.py --net='res'   --data_loc=<path-to-data> --l1_prune=True --base_model='res' --save_file='res_l1'

python prune.py --net='dense' --depth 100 --data_loc=<path-to-data> --base_model='dense' --save_file='dense_fisher' --learning_rate 1e-3 --weight_decay 1e-4 --batch_size 64 --no_epochs 2600
python prune.py --net='dense' --depth 100 --data_loc=<path-to-data> --l1_prune=True --base_model='dense' --save_file='dense_l1'  --learning_rate 1e-3 --weight_decay 1e-4 --batch_size 64  --no_epochs 2600

```
Note that the default is to perform Fisher pruning, so you don't need to pass a flag to use it.  
Once finished, we can train the pruned models from scratch, e.g.:  
```
python train.py --data_loc=<path-to-data> --net='RES' --base_file='res_fisher_<N>_prunes' --deploy --mask=1 --save_file='res_fisher_<N>_prunes_scratch'
```

Each model can then be evaluated using:
```
python train.py --deploy --eval --data_loc=<path-to-data> --net='res' --mask=1 --base_file='res_fisher_<N>_prunes'
```


### Training Reduced models

This can be done by varying the input arguments to train.py. To reduce depth or width of a WRN, change the corresponding option:
```
python train.py --net='res' --depth=<REDUCED DEPTH> --width=<REDUCE WIDTH> --data_loc=<path-to-data> --save_file='res_reduced'
```

To add bottlenecks, use the following:

```
python train.py --net='res' --depth=40 --width=2.0 --data_loc=<path-to-data> --save_file='res_bottle' --bottle --bottle_mult <Z>
```

With DenseNets you can modify the `depth` or `growth`, or use `--bottle --bottle_mult <Z>` as above.


### Acknowledgements

Code has been liberally borrowed from many a repo, including but not limited to:

```
https://github.com/xternalz/WideResNet-pytorch
https://github.com/bamos/densenet.pytorch
https://github.com/kuangliu/pytorch-cifar
https://github.com/ShichenLiu/CondenseNet
```
### Citing this work

If you would like to cite this work, please use the following bibtex entry:

```
@article{prunes,
  title={Pruning neural networks: is it time to nip it in the bud?},
  author={Crowley, Elliot J and Turner, Jack and Storkey, Amos and O'Boyle, Michael},
  journal={arXiv preprint arXiv:1810.04622},
  year={2018}
```