from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class BCTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            # qf1,
            # qf2,
            # target_qf1,
            # target_qf2,
            # behavior_policy=None,
            # dim_mult=1,

            # discount=0.99,
            # reward_scale=1.0,

            policy_lr=1e-3,
            optimizer_class=optim.Adam,

            # soft_target_tau=1e-2,
            # target_update_period=1,
            # plotter=None,
            # render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            # use_target_nets=True,
            # num_qs=2,

            # use_robot_state=False,
            # observation_keys=("observations",),
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        # self.qf1 = qf1
        # self.qf2 = qf2
        # self.target_qf1 = target_qf1
        # self.target_qf2 = target_qf2
        # self.soft_target_tau = soft_target_tau
        # self.target_update_period = target_update_period

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(
                    self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            # self.alpha_optimizer = optimizer_class(
            #     [self.log_alpha],
            #     lr=3e-4,
            # )

        # self.plotter = plotter
        # self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )

        self._optimizer_class = optimizer_class #for loading

        # self.discount = discount
        # self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        # self._use_target_nets = use_target_nets

        self._current_epoch = 0
        # self._policy_update_ctr = 0
        # self._num_q_update_steps = 0
        self._num_policy_update_steps = 0
        # self._num_policy_steps = 1

        # if not self._use_target_nets:
        #     self.target_qf1 = qf1
        #     self.target_qf2 = qf2

        # self.num_qs = num_qs
        self.discrete = False
        # self.needs_online_flag = True
        # self.use_robot_state = use_robot_state
        # self.observation_keys = observation_keys
        # if self.use_robot_state:
            # assert len(self.observation_keys) > 1

    def preprocess_obs(self, obs, num_repeat, actions):
        if isinstance(obs, torch.Tensor):
            obs_shape = obs.shape[0]
            if num_repeat is None:
                action_shape = actions.shape[0]
                num_repeat = int(action_shape / obs_shape)
            return (obs.unsqueeze(1).repeat(1, num_repeat, 1).view(
                obs.shape[0] * num_repeat, obs.shape[1]), obs.shape[0], num_repeat)
        elif isinstance(obs, dict):
            obs_dict = {}
            for key in obs.keys():
                obs_dict[key], obs_shape_dim0, num_repeat = self.preprocess_obs(
                    obs[key], num_repeat, actions)
            return obs_dict, obs_shape_dim0, num_repeat
        else:
            raise NotImplementedError

    def _get_tensor_values(self, obs, actions, network=None):
        obs_temp, obs_shape, num_repeat = self.preprocess_obs(
            obs, None, actions)
        preds = network(obs_temp, actions)
        preds = preds.view(obs_shape, num_repeat, 1)
        return preds

    def _get_policy_actions(self, obs, num_actions, network=None):
        obs_temp, obs_shape_dim0, _ = self.preprocess_obs(
            obs, num_actions, None)
        new_obs_actions, _, _, new_obs_log_pi, *_ = network(
            obs_temp, reparameterize=True, return_log_prob=True,
        )
        return new_obs_actions, new_obs_log_pi.view(
            obs_shape_dim0, num_actions, 1)

    def filter_dict(self, key_prefix, batch_dict):
        assert isinstance(key_prefix, str)
        return dict([(key, batch_dict[key]) for key in batch_dict.keys() 
            if key.find(key_prefix) == 0])

    def train_from_torch(self, batch, online=False):
        self._current_epoch += 1

        # if self.use_robot_state:
        #     obs = self.filter_dict("observations", batch)
        # else:
        obs = batch['observations']
        actions = batch['actions']

        """
        Policy and Alpha Loss
        """
        """Start with BC"""
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )
        alpha = 0.0
        policy_log_prob = self.policy.log_prob(obs, actions)
        policy_loss = (alpha * log_pi - policy_log_prob).mean()
        """
        Update networks
        """
        self._num_policy_update_steps += 1
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=False)
        self.policy_optimizer.step()

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            # self.eval_statistics['Num Q Updates'] = self._num_q_update_steps
            self.eval_statistics[
                'Num Policy Updates'] = self._num_policy_update_steps
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        base_list = [self.policy]
        #     self.qf1,
        #     self.qf2,
        #     self.target_qf1,
        #     self.target_qf2,
        # ]
        return base_list
