import argparse
import json
import os
import pickle

import torch
import roboverse

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.load_buffer import load_data_from_npy_chaining
from rlkit.launchers.launcher_util import run_experiment
from rlkit.samplers.data_collector import MdpPathCollector, \
    CustomMDPPathCollector
from rlkit.torch.conv_networks import CNN, ConcatCNN
from rlkit.torch.sac.cql import CQLTrainer
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.util.video import VideoSaveFunction

DEFAULT_PRIOR_BUFFER = ('/media/avi/data/Work/github/avisingh599/minibullet'
                        '/data/oct6_Widow250DrawerGraspNeutral-v0_20K_save_all'
                        '_noise_0.1_2020-10-06T19-37-26_100.npy')
DEFAULT_TASK_BUFFER = ('/media/avi/data/Work/github/avisingh599/minibullet'
                        '/data/oct6_Widow250DrawerGraspNeutral-v0_20K_save_all'
                        '_noise_0.1_2020-10-06T19-37-26_100.npy')
DEFAULT_CHECKPOINT_DIR = ('/media/avi/data/Work/data/cql-private-checkpoints/'
                          '10-13-cql-private-chaining-Widow250DoubleDrawerPick'
                          'PlaceOpenGraspNeutral-v0_2020_10_13_00_25_21_0000--'
                          's-22995')
NFS_PATH = '/nfs/kun1/users/avi/doodad-output/'


def experiment(variant):

    # variant_json = os.path.join(variant['checkpoint_dir'], 'variant.json')
    # with open(variant_json) as f:
    #     load_variant = json.load(f)

    checkpoint_filepath = os.path.join(variant['checkpoint_dir'],
                                       'itr_{}.pkl'.format(
                                           variant['checkpoint_epoch']))
    checkpoint = torch.load(checkpoint_filepath)

    # the following does not work for Bullet envs yet
    # eval_env = checkpoint['evaluation/env']
    # expl_env = checkpoint['exploration/env']

    eval_env = roboverse.make(variant['env'], transpose_image=True)
    expl_env = eval_env

    qf1 = checkpoint['trainer/qf1']
    qf2 = checkpoint['trainer/qf2']
    target_qf1 = checkpoint['trainer/target_qf1']
    target_qf2 = checkpoint['trainer/target_qf2']

    policy = checkpoint['trainer/policy']
    eval_policy = checkpoint['evaluation/policy']
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )

    observation_key = 'image'
    replay_buffer = load_data_from_npy_chaining(
        variant, expl_env, observation_key,
        extra_buffer_size=500*10*variant['algorithm_kwargs']['max_path_length'])

    trainer = CQLTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        eval_both=False,
        batch_rl=False,
        **variant['algorithm_kwargs']
    )
    video_func = VideoSaveFunction(variant)
    algorithm.post_epoch_funcs.append(video_func)

    algorithm.to(ptu.device)
    algorithm.train()


def enable_gpus(gpu_str):
    if gpu_str is not "":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="CQL",
        version="normal",
        algorithm_kwargs=dict(
            # num_epochs=100,
            # num_eval_steps_per_epoch=50,
            # num_trains_per_train_loop=100,
            # num_expl_steps_per_train_loop=100,
            # min_num_steps_before_training=100,
            # max_path_length=10,
            num_epochs=3000,
            num_eval_steps_per_epoch=300,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=300,
            min_num_steps_before_training=300,
            max_path_length=30,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            policy_lr=1E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,

            # Target nets/ policy vs Q-function update
            policy_eval_start=10000,
            num_qs=2,

            # min Q
            temp=1.0,
            min_q_version=3,
            min_q_weight=5.0,

            # lagrange
            with_lagrange=False,  # Defaults to False
            lagrange_thresh=10.0,

            # extra params
            num_random=1,
            max_q_backup=False,
            deterministic_backup=False,
        ),
        cnn_params=dict(
            kernel_sizes=[3, 3, 3],
            n_channels=[16, 16, 16],
            strides=[1, 1, 1],
            hidden_sizes=[1024, 512, 256],
            paddings=[1, 1, 1],
            pool_type='max2d',
            pool_sizes=[2, 2, 1],  # the one at the end means no pool
            pool_strides=[2, 2, 1],
            pool_paddings=[0, 0, 0],
            image_augmentation=True,
            image_augmentation_padding=4,
        ),
        dump_video_kwargs=dict(
            imsize=48,
            save_video_period=1,
        ),
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--max-path-length", type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--checkpoint-epoch", type=int, default=520)
    parser.add_argument("--prior-buffer", type=str, default=DEFAULT_PRIOR_BUFFER)
    parser.add_argument("--task-buffer", type=str, default=DEFAULT_TASK_BUFFER)

    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--min-q-weight", default=1.0, type=float,
                        help="Value of alpha in CQL")
    parser.add_argument("--use-lagrange", action="store_true", default=False)
    parser.add_argument("--lagrange-thresh", default=5.0, type=float,
                        help="Value of tau, used with --use-lagrange")
    parser.add_argument("--use-positive-rew", action="store_true", default=False)

    parser.add_argument("--max-q-backup", action="store_true", default=False,
                        help="For max_{a'} backups, set this to true")
    parser.add_argument("--no-deterministic-backup", action="store_true",
                        default=False,
                        help="By default, deterministic backup is used")
    parser.add_argument("--policy-eval-start", default=10000,
                        type=int)
    parser.add_argument("--policy-lr", default=1e-4, type=float)
    parser.add_argument("--min-q-version", default=3, type=int,
                        help=("min_q_version = 3 (CQL(H)), "
                              "version = 2 (CQL(rho))"))
    parser.add_argument("--num-eval-per-epoch", type=int, default=5)
    parser.add_argument("--seed", default=10, type=int)

    args = parser.parse_args()
    enable_gpus(args.gpu)
    variant['env'] = args.env
    variant['checkpoint_dir'] = args.checkpoint_dir
    variant['checkpoint_epoch'] = args.checkpoint_epoch

    variant['algorithm_kwargs']['max_path_length'] = args.max_path_length
    variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = \
        args.num_eval_per_epoch*args.max_path_length
    variant['algorithm_kwargs']['num_expl_steps_per_train_loop'] = \
        10*args.max_path_length
    variant['algorithm_kwargs']['min_num_steps_before_training'] = \
        10*args.max_path_length

    # For testing
    # variant['algorithm_kwargs']['num_expl_steps_per_train_loop'] = 100
    # variant['algorithm_kwargs']['min_num_steps_before_training'] = 100

    variant['prior_buffer'] = args.prior_buffer
    variant['task_buffer'] = args.task_buffer

    variant['trainer_kwargs']['max_q_backup'] = args.max_q_backup
    variant['trainer_kwargs']['deterministic_backup'] = \
        not args.no_deterministic_backup
    variant['trainer_kwargs']['min_q_weight'] = args.min_q_weight
    variant['trainer_kwargs']['policy_lr'] = args.policy_lr
    variant['trainer_kwargs']['min_q_version'] = args.min_q_version
    variant['trainer_kwargs']['policy_eval_start'] = args.policy_eval_start
    variant['trainer_kwargs']['lagrange_thresh'] = args.lagrange_thresh
    variant['trainer_kwargs']['with_lagrange'] = args.use_lagrange

    # Translate 0/1 rewards to +4/+10 rewards.
    variant['use_positive_rew'] = args.use_positive_rew
    variant['seed'] = args.seed

    ptu.set_gpu_mode(True)
    exp_prefix = 'cql-private-chaining-finetune-{}'.format(args.env)

    if os.path.isdir(NFS_PATH):
        base_log_dir = NFS_PATH
    else:
        base_log_dir = None

    run_experiment(
        experiment,
        base_log_dir=base_log_dir,
        exp_prefix=exp_prefix,
        mode='local',
        variant=variant,
        use_gpu=True,
        snapshot_mode='gap_and_last',
        snapshot_gap=10,
    )
