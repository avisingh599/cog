import argparse
import json
import os

import torch
import roboverse

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.load_buffer import load_data_from_npy
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.conv_networks import ConcatCNN
from rlkit.torch.sac.cql import CQLTrainer
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.util.video import VideoSaveFunction

DEFAULT_CHECKPOINT_DIR = ('/media/avi/data/Work/data/cql-private-checkpoints/'
                          '20-10-15-cql-private-BC-Widow250DoubleDrawerPick'
                          'PlaceOpenGraspNeutral-v0_2020_10_15_02_13_01_0000'
                          '--s-121/')
DEFAULT_BUFFER = ('/media/avi/data/Work/github/avisingh599/minibullet'
                   '/data/oct6_Widow250DrawerGraspNeutral-v0_20K_save_all'
                   '_noise_0.1_2020-10-06T19-37-26_100.npy')
CUSTOM_LOG_DIR = '/nfs/kun1/users/avi/doodad-output/'


def experiment(variant):
    checkpoint_filepath = os.path.join(variant['checkpoint_dir'],
                                       'itr_{}.pkl'.format(
                                           variant['checkpoint_epoch']))
    checkpoint = torch.load(checkpoint_filepath)

    eval_env = roboverse.make(variant['env'], transpose_image=True)
    expl_env = eval_env

    action_dim = eval_env.action_space.low.size
    cnn_params = variant['cnn_params']
    cnn_params.update(
        input_width=48,
        input_height=48,
        input_channels=3,
        output_size=1,
        added_fc_input_size=action_dim,
    )
    qf1 = ConcatCNN(**cnn_params)
    qf2 = ConcatCNN(**cnn_params)
    target_qf1 = ConcatCNN(**cnn_params)
    target_qf2 = ConcatCNN(**cnn_params)

    policy = checkpoint['evaluation/policy']
    eval_policy = MakeDeterministic(policy)

    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )

    observation_key = 'image'
    replay_buffer = load_data_from_npy(variant, expl_env, observation_key)

    trainer_kwargs = variant['trainer_kwargs']
    trainer = CQLTrainer(
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
        batch_rl=True,
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
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str,
                        default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--checkpoint-epoch", type=int, default=200)
    parser.add_argument("--max-path-length", type=int, default=50)
    parser.add_argument("--buffer", type=str, default=DEFAULT_BUFFER)
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--seed", default=10, type=int)
    args = parser.parse_args()

    variant_json = os.path.join(args.checkpoint_dir, 'variant.json')
    with open(variant_json) as f:
        variant = json.load(f)

    variant['trainer_kwargs'] = dict(
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
        min_q_weight=1.0,

        # lagrange
        with_lagrange=False,  # Defaults to False
        lagrange_thresh=5.0,

        # extra params
        num_random=1,
        max_q_backup=False,
        deterministic_backup=True,
    )

    enable_gpus(args.gpu)
    variant['env'] = args.env
    variant['checkpoint_dir'] = args.checkpoint_dir
    variant['checkpoint_epoch'] = args.checkpoint_epoch
    variant['buffer'] = args.buffer

    # For testing (should normally be commented)
    # variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = 100
    # variant['algorithm_kwargs']['num_expl_steps_per_train_loop'] = 100
    # variant['algorithm_kwargs']['min_num_steps_before_training'] = 100
    # variant['algorithm_kwargs']['num_trains_per_train_loop'] = 100

    variant['seed'] = args.seed
    ptu.set_gpu_mode(True)
    exp_prefix = 'cql-from-bc-init-{}'.format(variant['env'])

    if os.path.isdir(CUSTOM_LOG_DIR):
        base_log_dir = CUSTOM_LOG_DIR
    else:
        base_log_dir = None

    setup_logger(exp_prefix, variant=variant, base_log_dir=base_log_dir,
                 snapshot_mode='gap_and_last', snapshot_gap=10,)
    experiment(variant)
