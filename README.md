# COG

This repository accompanies the following paper:

**COG: Connecting New Skills to Past Experience with Offline Reinforcement Learning** <br/>
Avi Singh, Albert Yu, Jonathan Yang, Jesse Zhang, Aviral Kumar, Sergey Levine <br/>
[Conference on Robot Learning](https://www.robot-learning.org/), 2020 <br/>
[Website](https://sites.google.com/view/cog-rl) | Arxiv (coming soon) | Video (coming soon)

| Open drawer, take object out      |   Close top drawer, take object out | Remove obstacle, take object out | 
:-------------------------:|:-------------------------:|:-------------------------:
![](https://lh6.googleusercontent.com/DpqLrbyEZ4AWwTHlLQNJJUiwd9JzMDQVChd4tB4MTpSGmV35ygfXd7YVQHAlLel2GDsJ-O0=w1280) | ![](https://lh5.googleusercontent.com/GNF7PKJP7DaA__VBxQxkqi0TV5poFVA48ijvMfT4n2sXEQlKnJtVj90DxjLQAQM9xAU795hR=w1280) | ![](https://lh3.googleusercontent.com/x268zzRNzPBijYdy7E9DyYTiGr0olg9MZr8B0YzKHCDrHPCll8ISU-hUx_NOCDG9V0isGiwD=w1280) |

In this paper, we propose an approach to incorporate a large amount of prior 
data, either from previously solved tasks or from unsupervised or undirected
environment interaction, to extend and generalize learned behavior. This prior 
data is not specific to any one task, and can be used to extend a variety of
downstream skills. We train our policies in an end-to-end fashion, mapping 
high-dimensional image observations to low-level robot control commands. 

This code is based on the original [CQL](https://github.com/aviralkumar2907/CQL)
implementation. 

## Usage

By default, all logs will be stored in `cog/data/`. If you would like to save to
a different directory, update `CUSTOM_LOG_DIR` in the relevant launch script. 
An example command for using our method in offline mode: 

`python examples/cog.py --env=Widow250PickTray-v0 --max-path-length=40 
--prior-buffer=pickplace_prior.npy --task-buffer=pickplace_task.npy`

An example command for online finetuning from a saved checkpoint:

`python examples/cog_finetune.py --checkpoint-dir=LOG_DIR 
--online-data-only --checkpoint-epoch=1000`

An example command for running the behavior cloning baseline:

`python examples/chaining_bc.py --env=Widow250PickTray-v0 --max-path-length=40 
--prior-buffer=pickplace_prior.npy --task-buffer=pickplace_task.npy`

The datasets mentioned above can be downloaded from this
[Google drive link](https://drive.google.com/drive/folders/1jxBQE1adsFT1sWsfatbhiZG6Zkf3EW0Q?usp=sharing). 

Here are the exact commands to reproduce all results for our method in the paper:

```shell script
python cog.py --env=Widow250DoubleDrawerOpenGraspNeutral-v0 --max-path-length=50 --prior-buffer=closed_drawer_prior.npy --task-buffer=drawer_task.npy
python cog.py --env=Widow250DoubleDrawerCloseOpenGraspNeutral-v0 --max-path-length=80 --prior-buffer=blocked_drawer_1_prior.npy --task-buffer=drawer_task.npy
python cog.py --env=Widow250DoubleDrawerPickPlaceOpenGraspNeutral-v0 --max-path-length=80 --prior-buffer=blocked_drawer_2_prior.npy --task-buffer=drawer_task.npy
```

Replacing `cog.py` with `chaining_bc.py` will allow reproducing experiments
for the BC baseline. 

## Setup
Our code is based on CQL, which is in turn based on [rlkit](https://github.com/vitchyr/rlkit). 
The setup instructions are similar to rlkit, but we repeat them here for convenience:

```shell script
conda env create -f environment/linux-gpu-env.yml
source activate cql-env
pip install -e .
```

After the above, please install [roboverse](https://github.com/avisingh599/roboverse)
and its dependencies in the same conda env. 

## Datasets
The datasets used in this project can be downloaded using this 
[Google drive link](https://drive.google.com/drive/folders/1jxBQE1adsFT1sWsfatbhiZG6Zkf3EW0Q?usp=sharing). 

If you would like to download the dataset on a remote machine via the command
line, consider using [gdown](https://pypi.org/project/gdown/). 

## Known issues
- There is an OpenCV/ffmpeg warning when saving videos. It does not effect any 
of RL functionality, and the saved videos can still be viewed. However, we will fix this soon. 
- There is a torch checkpoint loading warning for online fine-tuning experiments.
This also does not effect functionality as long as the correct checkpoints are 
being loaded. We plan to look into this as well. 


## TODO
### High priority
- [ ] Remove hard-coded image key in line 114 in `rlkit/samplers/rollout_functions.py`
- [ ] Fix OpenCV/ffmpeg video saving warning - just need to switch to a difference codec. 
- [ ] Fix torch checkpoint loading warning -- maybe explicitly saving all the networks and optimizer instead of the trainer object can help here. 

### Soon
- [ ] Clean up `rlkit/data_management/load_buffer.py`
- [ ] `ObsDictReplayBuffer` and `ObsDictRelabelingBuffer` share a lot of code, but are currently implemented independently.
- [ ] Hard-coded `.cuda()` in line 234 in `rlkit/torch/sac/cql.py`
- [ ] Perhaps the `if` statement (line 437) for image format conversion is a bit too hacky in `ObsDictReplayBuffer`. 
