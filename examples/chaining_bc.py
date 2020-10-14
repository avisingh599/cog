import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.load_buffer import load_data_from_npy_chaining
from rlkit.samplers.data_collector import MdpPathCollector, \
    CustomMDPPathCollector

from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.bc import BCTrainer
from rlkit.torch.conv_networks import CNN, ConcatCNN
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.util.video import VideoSaveFunction

import argparse, os
import roboverse

DEFAULT_PRIOR_BUFFER = ('/media/avi/data/Work/github/avisingh599/minibullet'
                        '/data/oct6_Widow250DrawerGraspNeutral-v0_20K_save_all'
                        '_noise_0.1_2020-10-06T19-37-26_100.npy')
DEFAULT_TASK_BUFFER = ('/media/avi/data/Work/github/avisingh599/minibullet'
                        '/data/oct6_Widow250DrawerGraspNeutral-v0_20K_save_all'
                        '_noise_0.1_2020-10-06T19-37-26_100.npy')


def experiment(variant):
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

    cnn_params.update(
        output_size=256,
        added_fc_input_size=0,
        hidden_sizes=[1024, 512],
    )

    policy_obs_processor = CNN(**cnn_params)
    policy = TanhGaussianPolicy(
        obs_dim=cnn_params['output_size'],
        action_dim=action_dim,
        hidden_sizes=[256, 256, 256],
        obs_processor=policy_obs_processor,
    )

    if variant['stoch_eval_policy']:
        eval_policy = policy
    else:
        eval_policy = MakeDeterministic(policy)

    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = CustomMDPPathCollector(
        eval_env,
    )

    observation_key = 'image'
    replay_buffer = load_data_from_npy_chaining(
        variant, expl_env, observation_key)

    trainer = BCTrainer(
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
        batch_rl=True,
        **variant['algorithm_kwargs']
    )
    video_func = VideoSaveFunction(variant)
    algorithm.post_epoch_funcs.append(video_func)

    algorithm.to(ptu.device)
    algorithm.train()


def enable_gpus(gpu_str):
    if (gpu_str is not ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="weightedBC-corrected",
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
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
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
            policy_eval_start=4 * 10 ** 7,
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
        dump_video_kwargs=dict(
            imsize=48,
            save_video_period=1,
        ),
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--max-path-length", type=int, required=True)
    parser.add_argument("--prior-buffer", type=str, default=DEFAULT_PRIOR_BUFFER)
    parser.add_argument("--task-buffer", type=str, default=DEFAULT_TASK_BUFFER)
    parser.add_argument("--gpu", default='0', type=str)
    # parser.add_argument("--max-q-backup", type=str,
    #                     default="False")  # if we want to try max_{a'} backups, set this to true
    parser.add_argument("--deterministic-backup", type=str,
                        default="True")  # defaults to true, it does not backup entropy in the Q-function, as per Equation 3
    # parser.add_argument("--policy-eval-start", default=4 * 10 ** 7,
    #                     type=int)
    # parser.add_argument('--min-q-weight', default=1.0,
    #                     type=float)  # the value of alpha, set to 5.0 or 10.0 if not using lagrange
    parser.add_argument('--policy-lr', default=1e-4,
                        type=float)  # Policy learning rate
    # parser.add_argument('--min-q-version', default=3,
    #                     type=int)  # min_q_version = 3 (CQL(H)), version = 2 (CQL(rho))
    # parser.add_argument('--lagrange-thresh', default=5.0,
    #                     type=float)  # the value of tau, corresponds to the CQL(lagrange) version
    parser.add_argument('--num-eval-per-epoch', type=int, default=5)
    parser.add_argument("--stoch-eval-policy", action="store_true", default=False)
    parser.add_argument('--seed', default=10, type=int)

    args = parser.parse_args()
    enable_gpus(args.gpu)
    variant['env'] = args.env
    variant['algorithm_kwargs']['max_path_length'] = args.max_path_length
    variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = \
        args.num_eval_per_epoch*args.max_path_length

    variant['prior_buffer'] = args.prior_buffer
    variant['task_buffer'] = args.task_buffer

    # variant['trainer_kwargs']['max_q_backup'] = (
    #     True if args.max_q_backup == 'True' else False)
    variant['trainer_kwargs']['deterministic_backup'] = (
        True if args.deterministic_backup == 'True' else False)
    # variant['trainer_kwargs']['min_q_weight'] = args.min_q_weight
    variant['trainer_kwargs']['policy_lr'] = args.policy_lr
    # variant['trainer_kwargs']['min_q_version'] = args.min_q_version
    # variant['trainer_kwargs']['policy_eval_start'] = args.policy_eval_start
    # variant['trainer_kwargs']['lagrange_thresh'] = args.lagrange_thresh

    variant['cnn_params'] = dict(
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
    )

    variant['stoch_eval_policy'] = args.stoch_eval_policy
    variant['seed'] = args.seed
    ptu.set_gpu_mode(True)
    exp_prefix = 'cql-private-BC-{}'.format(args.env)

    run_experiment(
        experiment,
        exp_prefix=exp_prefix,
        mode='local',
        variant=variant,
        use_gpu=True,
        snapshot_mode='gap_and_last',
        snapshot_gap=10,
        seed=args.seed,
    )
