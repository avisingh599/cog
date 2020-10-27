import argparse
import json
import os

import torch
import roboverse

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.load_buffer import load_data_from_npy_chaining
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.conv_networks import ConcatCNN
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.util.video import VideoSaveFunction
from rlkit.data_management.obs_dict_replay_buffer import \
    ObsDictReplayBuffer


DEFAULT_CHECKPOINT_DIR = ('/media/avi/data/Work/data/cql-private-checkpoints/'
                          '20-10-16-cql-private-BC-Widow250DoubleDrawerCloseOpen'
                          'Neutral-v0_2020_10_16_14_43_42_0000--s-220')
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
    parser.add_argument("--checkpoint-epoch", type=int, default=800)
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--use-biased-sampling", action="store_true",
                        default=False)
    parser.add_argument("--online-data-only", action="store_true",
                        default=False)
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
    )

    enable_gpus(args.gpu)
    variant['checkpoint_dir'] = args.checkpoint_dir
    variant['checkpoint_epoch'] = args.checkpoint_epoch
    variant['online_data_only'] = args.online_data_only
    variant['biased_sampling'] = args.use_biased_sampling
    assert not (variant['biased_sampling'] and variant['online_data_only'])

    variant['algorithm_kwargs']['num_expl_steps_per_train_loop'] = \
        10 * variant['algorithm_kwargs']['max_path_length']
    variant['algorithm_kwargs']['min_num_steps_before_training'] = \
        100 * variant['algorithm_kwargs']['max_path_length']

    # For testing (should normally be commented)
    # variant['prior_buffer'] = DEFAULT_PRIOR_BUFFER
    # variant['task_buffer'] = DEFAULT_TASK_BUFFER
    # variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = 100
    # variant['algorithm_kwargs']['num_expl_steps_per_train_loop'] = 100
    # variant['algorithm_kwargs']['min_num_steps_before_training'] = 100
    # variant['algorithm_kwargs']['num_trains_per_train_loop'] = 100

    variant['seed'] = args.seed
    ptu.set_gpu_mode(True)
    exp_prefix = 'sac-from-bc-finetune-{}'.format(variant['env'])

    if os.path.isdir(CUSTOM_LOG_DIR):
        base_log_dir = CUSTOM_LOG_DIR
    else:
        base_log_dir = None

    setup_logger(exp_prefix, variant=variant, base_log_dir=base_log_dir,
                 snapshot_mode='gap_and_last', snapshot_gap=25,)
    experiment(variant)
