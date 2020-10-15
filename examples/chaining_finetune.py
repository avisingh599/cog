import argparse
import json
import os
import pickle

import torch
import roboverse

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.load_buffer import load_data_from_npy_chaining
from rlkit.launchers.launcher_util import run_experiment
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.cql import CQLTrainer
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.util.video import VideoSaveFunction
from rlkit.data_management.obs_dict_replay_buffer import \
    ObsDictReplayBuffer

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
    online_buffer_size = 500 * 10 * variant['algorithm_kwargs'][
        'max_path_length']

    if variant['online_data_only']:
        replay_buffer = ObsDictReplayBuffer(online_buffer_size, expl_env,
                                            observation_key=observation_key)
    else:
        replay_buffer = load_data_from_npy_chaining(
            variant, expl_env, observation_key,
            extra_buffer_size=online_buffer_size)

    trainer_kwargs = variant['trainer_kwargs']
    if trainer_kwargs['min_q_weight'] > 0.:
        trainer = CQLTrainer(
            env=eval_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            **trainer_kwargs
        )
    else:
        cql_trainner_kwargs = ['policy_eval_start', 'num_qs', 'min_q_weight',
                               'lagrange_thresh', 'num_random', 'temp',
                               'min_q_version', 'with_lagrange',
                               'max_q_backup', 'deterministic_backup']
        [trainer_kwargs.pop(key) for key in cql_trainner_kwargs]
        trainer = SACTrainer(
            env=eval_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            **trainer_kwargs
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str,
                        default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--checkpoint-epoch", type=int, default=520)
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--min-q-weight", default=None, type=float,
                        help="Value of alpha in CQL")
    parser.add_argument("--online-data-only", action="store_true",
                        default=False)
    parser.add_argument("--seed", default=10, type=int)
    args = parser.parse_args()

    variant_json = os.path.join(args.checkpoint_dir, 'variant.json')
    with open(variant_json) as f:
        variant = json.load(f)

    enable_gpus(args.gpu)
    variant['checkpoint_dir'] = args.checkpoint_dir
    variant['checkpoint_epoch'] = args.checkpoint_epoch
    variant['online_data_only'] = args.online_data_only

    variant['algorithm_kwargs']['num_expl_steps_per_train_loop'] = \
        10 * variant['algorithm_kwargs']['max_path_length']
    variant['algorithm_kwargs']['min_num_steps_before_training'] = \
        10 * variant['algorithm_kwargs']['max_path_length']

    # For testing (should normally be commented)
    # variant['prior_buffer'] = DEFAULT_PRIOR_BUFFER
    # variant['task_buffer'] = DEFAULT_TASK_BUFFER
    # variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = 100
    # variant['algorithm_kwargs']['num_expl_steps_per_train_loop'] = 100
    # variant['algorithm_kwargs']['min_num_steps_before_training'] = 100
    # variant['algorithm_kwargs']['num_trains_per_train_loop'] = 100

    variant['offline_min_q_weight'] = variant['trainer_kwargs']['min_q_weight']
    if args.min_q_weight is not None:
        variant['trainer_kwargs']['min_q_weight'] = args.min_q_weight

    variant['trainer_kwargs']['policy_eval_start'] = 0

    variant['seed'] = args.seed

    ptu.set_gpu_mode(True)
    exp_prefix = 'cql-private-chaining-finetune-{}'.format(variant['env'])

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
