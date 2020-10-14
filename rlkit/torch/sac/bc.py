from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from torch import autograd

class BCTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            behavior_policy=None,
            dim_mult=1,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            use_target_nets=True,
            policy_eval_start=0,
            num_qs=2,

            ## For min_Q runs
            with_min_q=False,
            new_min_q=False,
            min_q_version=0,
            temp=1.0,
            hinge_bellman=False,
            use_projected_grad=False,
            normalize_magnitudes=False,
            regress_constant=False,
            min_q_weight=1.0,
            data_subtract=False,

            ## sort of backup
            max_q_backup=False,
            deterministic_backup=False,
            num_random=4,
            max_q_num_actions=4,

            ## Handling discrete actions
            discrete=False,

            # Automatic minq tuning
            with_lagrange=False,
            lagrange_thresh=5.0,
            use_robot_state=True,
            observation_keys=("observations",),
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(
                    self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=3e-4,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self._optimizer_class = optimizer_class #for loading

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self._use_target_nets = use_target_nets
        self.policy_eval_start = policy_eval_start

        self._current_epoch = 0
        self._policy_update_ctr = 0
        self._num_q_update_steps = 0
        self._num_policy_update_steps = 0
        self.policy_eval_start = policy_eval_start
        self._num_policy_steps = 1

        if not self._use_target_nets:
            self.target_qf1 = qf1
            self.target_qf2 = qf2

        self.softmax = torch.nn.Softmax(dim=-1)
        self.num_qs = num_qs

        ## min Q
        self.with_min_q = with_min_q
        self.new_min_q = new_min_q
        self.temp = temp
        self.min_q_version = min_q_version
        self.use_projected_grad = use_projected_grad
        self.normalize_magnitudes = normalize_magnitudes
        self.regress_constant = regress_constant
        self.min_q_weight = min_q_weight
        self.original_min_q_weight = min_q_weight # Temp variable right now, just used for annealing from batch_rl_alg
        self.softmax = torch.nn.Softmax(dim=1)
        self.hinge_bellman = hinge_bellman
        self.softplus = torch.nn.Softplus(beta=self.temp, threshold=20)
        self.data_subtract = data_subtract

        self.max_q_backup = max_q_backup
        self.deterministic_backup = deterministic_backup
        self.num_random = num_random
        self.max_q_num_actions = max_q_num_actions
        self.discrete = discrete

        # lagrange
        self.update_lagrange_params(with_lagrange, lagrange_thresh, qf_lr)

        self.needs_online_flag = True

        self.use_robot_state = use_robot_state
        self.observation_keys = observation_keys
        if self.use_robot_state:
            assert len(self.observation_keys) > 1

    def update_lagrange_params(self, with_lagrange, lagrange_thresh, alpha_prime_lr, log_alpha_prime=0.):
        # Used when switching from offline lagrange False Trainer
        # to online lagrange True Trainer
        self.with_lagrange = with_lagrange
        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh
            self.log_alpha_prime = ptu.tensor([log_alpha_prime], requires_grad=True)
            self.alpha_prime_optimizer = self._optimizer_class(
                [self.log_alpha_prime],
                lr=alpha_prime_lr,
            )

    def compute_new_grad(self, grad1, grad2):
        new_grad = []
        for (grad1i, grad2i) in zip(grad1, grad2):
            proj_i = ((grad1i * grad2i).sum() * grad2i) / (
                        grad2i * grad2i + 1e-7).sum()
            # conditional =
            if self.normalize_magnitudes:
                proj1 = (grad1i - proj_i).clamp_(max=0.01, min=-0.01)
                new_grad.append(proj1 + grad2i)
            else:
                new_grad.append(grad1i - proj_i + grad2i)
        return new_grad

    def compute_mt_grad(self, grad1, grad2):
        """Solution from Koltun paper."""
        new_grad = []
        for (grad1i, grad2i) in zip(grad1, grad2):
            l2_norm_grad = torch.norm(grad1i - grad2i).pow(2)
            alpha_i = ((grad2i - grad1i) * grad2i).sum() / (l2_norm_grad + 1e-7)
            alpha_i = alpha_i.clamp_(min=0.0, max=1.0)
            new_grad.append(grad1i * alpha_i + (1.0 - alpha_i) * grad2i)
        return new_grad

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
        if not self.discrete:
            return new_obs_actions, new_obs_log_pi.view(
                obs_shape_dim0, num_actions, 1)
        else:
            return new_obs_actions

    def filter_dict(self, key_prefix, batch_dict):
        assert isinstance(key_prefix, str)
        return dict([(key, batch_dict[key]) for key in batch_dict.keys() 
            if key.find(key_prefix) == 0])

    def train_from_torch(self, batch, online=False):
        self._current_epoch += 1
        rewards = batch['rewards']
        terminals = batch['terminals']

        if self.use_robot_state:
            obs = self.filter_dict("observations", batch)
            next_obs = self.filter_dict("next_observations", batch)
        else:
            obs = batch['observations']
            next_obs = batch['next_observations']

        if not self.discrete:
            actions = batch['actions']
        else:
            actions = batch['actions'].argmax(dim=-1)

        """
        Policy and Alpha Loss
        """
        if self._current_epoch < self.policy_eval_start:
            """Start with BC"""
            new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
                obs, reparameterize=True, return_log_prob=True,
            )
            alpha = 0.0
            policy_log_prob = self.policy.log_prob_aviral(obs, actions)
            policy_loss = (alpha * log_pi - policy_log_prob).mean()
            # print ('Policy Loss: ', policy_loss.item())
        """
        QF Loss
        """
        """
        Update networks
        """
        # Update the Q-functions iff

        self._num_policy_update_steps += 1
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=False)
        self.policy_optimizer.step()

        """
        Soft Updates
        """
        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.OFFLINE_ONLY-minQ-SAC-{}-state-{}-min-q-ver-{}'.format(args.env, buffer_name, args.min_q_versioni
            """
            self.eval_statistics['Num Q Updates'] = self._num_q_update_steps
            self.eval_statistics[
                'Num Policy Updates'] = self._num_policy_update_steps
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            if not self.discrete:
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Policy mu',
                    ptu.get_numpy(policy_mean),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Policy log std',
                    ptu.get_numpy(policy_log_std),
                ))
            else:
                self.eval_statistics['Policy entropy'] = ptu.get_numpy(
                    entropies).mean()
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        base_list = [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]
        return base_list
    def get_snapshot(self):
        return dict(trainer=self)

    def load_from_dict(self, dict, device):
        loadable = self.get_snapshot()
        # First, load the models
        for key, value in loadable.items():
            key = 'trainer/%s' % key
            if key in dict and 'log_alpha' in key:
                #with torch.no_grad():
                #    self.log_alpha = torch.log(ptu.ones(dict[key].shape, requires_grad=False) * dict[key].cpu().to(device))
                self.log_alpha = torch.log(ptu.ones(1, requires_grad=False) * dict[key].data[0])
                self.log_alpha.requires_grad = True
            elif key in dict and 'optim' not in key:
                value.load_state_dict(dict[key].state_dict())
        # Reinitialize the optimizers:
        optimizers = [
            self.policy_optimizer,
            self.qf1_optimizer, 
            self.qf2_optimizer,
            ]
        models = [
            self.policy,
            self.qf1,
            self.qf2,
            ]
        for (optimizer, model) in zip(optimizers, models):
            optimizer.__init__(
                model.parameters(),
                **optimizer.defaults,
            )
        if hasattr(self, 'alpha_optimizer'):
            self.alpha_optimizer.__init__(
                [self.log_alpha],
                **self.alpha_optimizer.defaults,
            )
        # Now load state dicts of optimizers
        for key, value in loadable.items():
            key = 'trainer/%s' % key
            if key in dict and 'optim' in key:
                value.load_state_dict(dict[key].state_dict())
                # Needed to avoid a device mismtach problem
                for state in value.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device=device)
            elif key not in dict:
                print("Could not load %s!" % key)
        if 'trainer/log_alpha' not in loadable:
            #self.log_alpha = ptu.ones(1, requires_grad=True)
            # Default to 1
            self.log_alpha = torch.log(ptu.ones(1, requires_grad=False))
            self.log_alpha.requires_grad = True
            self.alpha_optimizer = self._optimizer_class(
                [self.log_alpha],
                lr=3e-4,
            )
