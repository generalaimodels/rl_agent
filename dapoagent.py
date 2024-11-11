import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Any, Dict, Optional, Type, Union, Callable, Tuple
import logging

# Suppress DeprecationWarning for numpy
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_schedule_fn(
    schedule: Union[float, Callable[[float], float]]
) -> Callable[[float], float]:
    """
    Returns a schedule function based on the input.

    Args:
        schedule (Union[float, Callable[[float], float]]): A constant or a function that takes remaining progress and returns a value.

    Returns:
        Callable[[float], float]: A function mapping remaining progress to a value.
    """
    if isinstance(schedule, (float, int)):
        return lambda _: schedule
    elif callable(schedule):
        return schedule
    else:
        raise ValueError("Schedule must be a float or a callable.")


def explained_variance(y_pred: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the explained variance.

    Args:
        y_pred (np.ndarray): Predicted values.
        y (np.ndarray): True values.

    Returns:
        float: Explained variance.
    """
    var_y = np.var(y)
    return np.nan if var_y == 0 else 1 - np.var(y - y_pred) / var_y


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO and DAPO-KL.
    Supports both discrete and continuous action spaces.
    """
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_sizes: Tuple[int, ...] = (64, 64),
    ):
        super(ActorCritic, self).__init__()

        # Common feature extractor
        self.shared_layers = self._build_mlp(observation_space.shape[0], hidden_sizes)

        # Actor head
        if isinstance(action_space, gym.spaces.Discrete):
            self.actor = nn.Sequential(
                self._build_mlp_layer(hidden_sizes[-1], hidden_sizes[-1]),
                nn.Linear(hidden_sizes[-1], action_space.n),
            )
        elif isinstance(action_space, gym.spaces.Box):
            self.actor = nn.Sequential(
                self._build_mlp_layer(hidden_sizes[-1], hidden_sizes[-1]),
                nn.Linear(hidden_sizes[-1], action_space.shape[0]),
            )
            # Learnable log_std for continuous actions
            self.log_std = nn.Parameter(torch.zeros(action_space.shape[0]))
        else:
            raise NotImplementedError("Unsupported action space type.")

        # Critic head for V(s)
        self.critic = nn.Sequential(
            self._build_mlp_layer(hidden_sizes[-1], hidden_sizes[-1]),
            nn.Linear(hidden_sizes[-1], 1),
        )

        # Q-function head for DAPO-KL (Estimating Q(s, a))
        if isinstance(action_space, gym.spaces.Discrete):
            action_dim = action_space.n
        elif isinstance(action_space, gym.spaces.Box):
            action_dim = action_space.shape[0]
        else:
            raise NotImplementedError("Unsupported action space type.")

        # Q-function takes concatenated state and action_one_hot as input
        self.q_function = nn.Sequential(
            nn.Linear(hidden_sizes[-1] + action_dim, hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], 1),
        )

        # Placeholder for environment, to be set externally
        self.env: Optional[gym.Env] = None

    @staticmethod
    def _build_mlp(input_dim: int, hidden_sizes: Tuple[int, ...]) -> nn.Sequential:
        layers = []
        last_size = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last_size, size))
            layers.append(nn.ReLU())
            last_size = size
        return nn.Sequential(*layers)

    @staticmethod
    def _build_mlp_layer(input_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input observations.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Action logits or means, and value estimates.
        """
        shared = self.shared_layers(x)
        action_logits_or_means = self.actor(shared)
        value = self.critic(shared)
        return action_logits_or_means, value

    def get_q_value(self, state: torch.Tensor, action_one_hot: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-value for a given state-action pair.

        Args:
            state (torch.Tensor): State tensor **after shared_layers**.
            action_one_hot (torch.Tensor): One-hot encoded action tensor.

        Returns:
            torch.Tensor: Q-value estimate.
        """
        x = torch.cat([state, action_one_hot], dim=-1)
        q_value = self.q_function(x)
        return q_value.squeeze(-1)

    def get_action(
        self, obs: np.ndarray
    ) -> Tuple[Union[int, np.ndarray], torch.Tensor]:
        """
        Sample an action from the policy.

        Args:
            obs (np.ndarray): Observation from the environment.

        Returns:
            Tuple[Union[int, np.ndarray], torch.Tensor]: The sampled action and log probability.
        """
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(next(self.parameters()).device)
        action_output, value = self.forward(obs_tensor)
        action_output = action_output.squeeze(0)

        if isinstance(self.env.action_space, gym.spaces.Box):
            # Continuous action space
            action_mean = action_output
            action_std = torch.exp(self.log_std)
            distribution = Normal(action_mean, action_std)
            action = distribution.sample()
            log_prob = distribution.log_prob(action).sum(-1)
            # Clamp the action to ensure it's within the action space bounds
            action_clipped = torch.clamp(
                action,
                torch.from_numpy(self.env.action_space.low).to(action.device),
                torch.from_numpy(self.env.action_space.high).to(action.device)
            )
            return action_clipped.cpu().numpy(), log_prob
        else:
            # Discrete action space
            distribution = Categorical(logits=action_output)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
            return action.item(), log_prob

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions to obtain log probabilities, values, and entropy.

        Args:
            obs (torch.Tensor): Observations.
            actions (torch.Tensor): Actions taken.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Values, log probabilities, and entropy.
        """
        action_output, value = self.forward(obs)
        if isinstance(self.env.action_space, gym.spaces.Box):
            # Continuous action space
            action_mean = action_output
            action_std = torch.exp(self.log_std)
            distribution = Normal(action_mean, action_std)
            log_prob = distribution.log_prob(actions).sum(-1)
            entropy = distribution.entropy().sum(-1)
        else:
            # Discrete action space
            distribution = Categorical(logits=action_output)
            log_prob = distribution.log_prob(actions)
            entropy = distribution.entropy()
        return value.squeeze(-1), log_prob, entropy


class RolloutBuffer:
    """
    Buffer for storing rollout data.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.buffer_size = buffer_size
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.observations = np.zeros(
            (buffer_size, *observation_space.shape), dtype=np.float32
        )
        if isinstance(action_space, gym.spaces.Discrete):
            self.actions = np.zeros(buffer_size, dtype=np.int64)
        else:
            self.actions = np.zeros((buffer_size, action_space.shape[0]), dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)

        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        """
        Add a transition to the buffer.

        Args:
            obs (np.ndarray): Observation.
            action (Union[int, np.ndarray]): Action taken.
            reward (float): Reward received.
            done (bool): Whether the episode ended.
            log_prob (float): Log probability of the action.
            value (float): Value estimate.
        """
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def compute_returns_and_advantages(
        self, last_value: float, last_done: bool
    ) -> None:
        """
        Compute returns and advantages using GAE.

        Args:
            last_value (float): Value estimate for the last observation.
            last_done (bool): Whether the last observation was terminal.
        """
        path_slice = slice(0, self.size)
        rewards = self.rewards[path_slice]
        dones = self.dones[path_slice]
        values = self.values[path_slice]
        if last_done:
            last_value = 0
        deltas = rewards + self.gamma * (1 - dones) * np.append(
            values[1:], last_value
        ) - values
        advantages = np.zeros_like(rewards)
        adv = 0.0
        for t in reversed(range(len(rewards))):
            adv = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * adv
            advantages[t] = adv
        returns = advantages + values

        self.advantages[path_slice] = advantages
        self.returns[path_slice] = returns

    def get_batches(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Yield batches of data.

        Args:
            batch_size (int): Size of each batch.

        Yields:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Batch of observations, actions, log_probs, returns, advantages.
        """
        indices = np.arange(self.size)
        np.random.shuffle(indices)
        for start in range(0, self.size, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            yield (
                self.observations[batch_indices],
                self.actions[batch_indices],
                self.log_probs[batch_indices],
                self.returns[batch_indices],
                self.advantages[batch_indices],
            )

    def clear(self) -> None:
        """
        Clear the buffer.
        """
        self.ptr = 0
        self.size = 0


class DAPO:
    """
    Dual Approximation Policy Optimization (DAPO) algorithm - KL Variant.
    """
    def __init__(
        self,
        env: gym.Env,
        policy: Type[ActorCritic],
        learning_rate: Union[float, Callable[[float], float]] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        tau: float = 0.1,  # Temperature parameter for DAPO-KL
        tau_schedule: Union[float, Callable[[float], float]] = 0.1,
        clip_range: Union[float, Callable[[float], float]] = 0.2,  # Not used in DAPO-KL
        clip_range_vf: Optional[Union[float, Callable[[float], float]]] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = 0.01,
        tensorboard_log: Optional[str] = None,
        device: Union[torch.device, str] = "cpu",
        seed: Optional[int] = None,
    ):
        """
        Initialize the DAPO algorithm.

        Args:
            env (gym.Env): The environment to train on.
            policy (Type[ActorCritic]): The policy network class.
            learning_rate (Union[float, Callable[[float], float]], optional): Learning rate or schedule.
            n_steps (int, optional): Number of steps per rollout.
            batch_size (int, optional): Batch size for updates.
            n_epochs (int, optional): Number of epochs per update.
            gamma (float, optional): Discount factor.
            gae_lambda (float, optional): GAE lambda.
            tau (float, optional): Temperature parameter for DAPO-KL.
            tau_schedule (Union[float, Callable[[float], float]], optional): Schedule for tau.
            clip_range (Union[float, Callable[[float], float]], optional): Clipping range (not used in DAPO-KL).
            clip_range_vf (Optional[Union[float, Callable[[float], float]]], optional): Clipping range for value function.
            ent_coef (float, optional): Entropy coefficient.
            vf_coef (float, optional): Value function coefficient.
            max_grad_norm (float, optional): Max gradient norm for clipping.
            target_kl (Optional[float], optional): Target KL divergence.
            tensorboard_log (Optional[str], optional): TensorBoard log directory.
            device (Union[torch.device, str], optional): Device to use.
            seed (Optional[int], optional): Random seed.
        """
        self.env = env
        self.device = torch.device(device) if isinstance(device, str) else device
        self.seed = seed
        if self.seed is not None:
            self.env.reset(seed=self.seed)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        self.policy = policy(env.observation_space, env.action_space).to(self.device)
        self.policy.env = env  # To access env properties within ActorCritic
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.learning_rate = get_schedule_fn(learning_rate)
        self.tau = tau
        self.tau_schedule = get_schedule_fn(tau_schedule)
        self.clip_range = get_schedule_fn(clip_range)  # Not used in DAPO-KL
        self.clip_range_vf = (
            get_schedule_fn(clip_range_vf) if clip_range_vf is not None else None
        )
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.n_steps,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        self.logger = {}  # Placeholder for logging
        self.tensorboard_log = tensorboard_log
        if tensorboard_log is not None:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=tensorboard_log)
        else:
            self.writer = None

        self.total_steps = 0

        # Initialize target critic network for stability
        self.target_critic_net = ActorCritic(env.observation_space, env.action_space).to(self.device)
        self.target_critic_net.load_state_dict(self.policy.state_dict())
        self.tau_update = 0.005  # Soft update parameter

    def soft_update_target_network(self) -> None:
        """
        Soft update target critic network parameters.
        """
        for target_param, param in zip(self.target_critic_net.parameters(), self.policy.parameters()):
            target_param.data.copy_(self.tau_update * param.data + (1.0 - self.tau_update) * target_param.data)

    def _collect_rollout(self) -> None:
        """
        Collect a rollout by interacting with the environment.
        """
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result

        for _ in range(self.n_steps):
            with torch.no_grad():
                action, log_prob = self.policy.get_action(obs)
                # For value, need to convert obs to tensor
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                _, value = self.policy.forward(obs_tensor)
                value = value.item()
            step_result = self.env.step(action)
            # Handle different Gym versions
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_result

            self.rollout_buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                done=done,
                log_prob=log_prob.item(),
                value=value,
            )
            self.total_steps += 1
            obs = next_obs
            if done:
                reset_result = self.env.reset()
                if isinstance(reset_result, tuple):
                    obs, _ = reset_result
                else:
                    obs = reset_result

    def _update(self) -> None:
        """
        Update the policy using the collected rollout.
        """
        # Compute advantages and returns
        with torch.no_grad():
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                last_obs, _ = reset_result
            else:
                last_obs = reset_result
            last_obs_tensor = torch.from_numpy(last_obs).float().unsqueeze(0).to(self.device)
            _, last_value = self.policy.forward(last_obs_tensor)
            last_value = last_value.item()
            # Track if the last state was done
            last_done = False  # Modify based on your specific use-case or track it during rollout
        self.rollout_buffer.compute_returns_and_advantages(last_value, last_done)

        # Normalize advantages
        advantages = self.rollout_buffer.advantages[: self.rollout_buffer.size]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        returns = torch.from_numpy(self.rollout_buffer.returns[: self.rollout_buffer.size]).float().to(self.device)
        observations = torch.from_numpy(self.rollout_buffer.observations[: self.rollout_buffer.size]).float().to(self.device)
        actions = self.rollout_buffer.actions[: self.rollout_buffer.size]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            actions = torch.from_numpy(actions).long().to(self.device)
        else:
            actions = torch.from_numpy(actions).float().to(self.device)
        old_log_probs = torch.from_numpy(self.rollout_buffer.log_probs[: self.rollout_buffer.size]).float().to(self.device)
        advantages = torch.from_numpy(advantages).float().to(self.device)

        # Compute Q-values for all actions in the batch using the critic
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            num_actions = self.env.action_space.n
            # **Corrected Part: Process observations through shared_layers**
            with torch.no_grad():
                shared_obs = self.policy.shared_layers(observations)
            # Repeat shared observations for each action
            expanded_obs = shared_obs.unsqueeze(1).repeat(1, num_actions, 1).view(-1, shared_obs.shape[-1])  # Shape: (batch_size * num_actions, hidden_size)
            # Create action indices
            all_actions = torch.arange(num_actions).unsqueeze(0).expand(len(observations), -1).reshape(-1).to(self.device)
            # Convert to one-hot
            all_actions_one_hot = F.one_hot(all_actions, num_classes=num_actions).float()

            # Compute Q-values using the policy's Q-function
            q_values = self.policy.get_q_value(expanded_obs, all_actions_one_hot)  # Shape: (batch_size * num_actions,)
            q_values = q_values.view(len(observations), num_actions)  # Shape: (batch_size, num_actions)
        else:
            # For continuous action spaces, more complex handling is required
            raise NotImplementedError("DAPO-KL is currently implemented for Discrete action spaces only.")

        # Define temperature parameter tau
        current_tau = self.tau_schedule(1.0)

        # Compute the target policy distribution
        # target_pi(s,a) proportional to pi_old(s,a) * exp(Q(s,a)/tau)
        # Here, pi_old(s,a) is approximated using the log_probs from the rollout
        # To handle this correctly, ensure that log_probs correspond to the actions taken

        # However, for simplicity, let's approximate pi_old(s,a) using the log_probs
        # This is a simplification and may not be accurate
        pi_old = torch.exp(old_log_probs).unsqueeze(1)  # Shape: (batch_size, 1)
        pi_old = pi_old.repeat(1, num_actions)       # Shape: (batch_size, num_actions)

        # Now compute target_pi
        target_pi_unnormalized = pi_old * torch.exp(q_values / current_tau)
        target_pi = target_pi_unnormalized / (target_pi_unnormalized.sum(dim=1, keepdim=True) + 1e-8)

        # Evaluate current policy
        values, log_probs, entropies = self.policy.evaluate_actions(observations, actions)
        values = values.squeeze(-1)

        # Compute KL divergence loss between the current policy and the target policy
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            # Gather log probabilities for all actions
            logits = self.policy(observations)[0]  # Shape: (batch_size, n_actions)
            log_probs_new = F.log_softmax(logits, dim=-1)  # Shape: (batch_size, n_actions)
            kl_div = F.kl_div(log_probs_new, target_pi, reduction='batchmean')
            policy_loss = kl_div
        else:
            raise NotImplementedError("DAPO-KL is currently implemented for Discrete action spaces only.")

        # Value function loss (same as PPO)
        if self.clip_range_vf is not None:
            # **Corrected Part: Process observations through shared_layers before critic**
            with torch.no_grad():
                shared_obs_for_clip = self.policy.shared_layers(observations)
                value_pred_clipped = self.policy.critic(shared_obs_for_clip).squeeze(-1).detach()
            value_pred_clipped += torch.clamp(
                values - value_pred_clipped,
                min=-self.clip_range_vf(1.0),
                max=self.clip_range_vf(1.0)
            )
            value_loss_unclipped = F.mse_loss(values, returns)
            value_loss_clipped = F.mse_loss(value_pred_clipped, returns)
            value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
        else:
            value_loss = F.mse_loss(values, returns).mean()

        # Entropy loss (same as PPO)
        entropy_loss = -entropies.mean()

        # Total loss
        loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

        # Optimize the policy
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Logging
        if self.writer is not None:
            self.writer.add_scalar("loss/policy_kl", policy_loss.item(), self.total_steps)
            self.writer.add_scalar("loss/value", value_loss.item(), self.total_steps)
            self.writer.add_scalar("loss/entropy", entropy_loss.item(), self.total_steps)
            self.writer.add_scalar("metrics/kl_divergence", kl_div.item(), self.total_steps)

        # Update target critic network
        self.soft_update_target_network()

        # Logging explained variance
        ev = explained_variance(values.detach().numpy(), returns.cpu().numpy())
        logger.info(f"Explained Variance: {ev:.4f}")

        self.rollout_buffer.clear()

    def learn(
        self, 
        total_timesteps: int, 
        log_interval: int = 10000, 
        eval_env: Optional[gym.Env] = None, 
        eval_interval: int = 20000, 
        eval_episodes: int = 10
    ) -> None:
        """
        Train the DAPO agent.

        Args:
            total_timesteps (int): Total number of timesteps to train for.
            log_interval (int, optional): Interval for logging training progress.
            eval_env (Optional[gym.Env], optional): Environment for evaluation.
            eval_interval (int, optional): Interval for evaluation.
            eval_episodes (int, optional): Number of episodes to run during evaluation.
        """
        try:
            while self.total_steps < total_timesteps:
                self._collect_rollout()
                self._update()

                # Logging total steps
                if self.total_steps >= log_interval and (self.total_steps // log_interval) > 0:
                    logger.info(f"Total Steps: {self.total_steps}")

                # Evaluation
                if eval_env is not None and self.total_steps >= eval_interval and (self.total_steps // eval_interval) > 0:
                    avg_reward = self.evaluate(eval_env, eval_episodes)
                    logger.info(f"Evaluation at step {self.total_steps}: Average Reward: {avg_reward}")
                    if self.writer is not None:
                        self.writer.add_scalar("evaluation/average_reward", avg_reward, self.total_steps)

        except KeyboardInterrupt:
            print("Training interrupted by user.")

    def evaluate(self, env: gym.Env, episodes: int = 10) -> float:
        """
        Evaluate the current policy.

        Args:
            env (gym.Env): Environment to evaluate on.
            episodes (int, optional): Number of episodes to run.

        Returns:
            float: Average reward over the episodes.
        """
        total_reward = 0.0
        for episode in range(episodes):
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, _ = reset_result
            else:
                obs = reset_result
            done = False
            episode_reward = 0.0
            while not done:
                action, _ = self.policy.get_action(obs)
                step_result = env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_result
                episode_reward += reward
            total_reward += episode_reward
        avg_reward = total_reward / episodes
        return avg_reward

    def save(self, path: str) -> None:
        """
        Save the model parameters.

        Args:
            path (str): Path to save the model.
        """
        torch.save(self.policy.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """
        Load the model parameters.

        Args:
            path (str): Path to load the model from.
        """
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.policy.to(self.device)
        print(f"Model loaded from {path}")


def visualize_agent(env: gym.Env, agent: DAPO, episodes: int = 5) -> None:
    """
    Visualize the trained agent.

    Args:
        env (gym.Env): The environment to visualize on.
        agent (DAPO): The trained DAPO agent.
        episodes (int, optional): Number of episodes to visualize.
    """
    for episode in range(episodes):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result
        done = False
        episode_reward = 0.0
        while not done:
            env.render()
            action, _ = agent.policy.get_action(obs)
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            episode_reward += reward
        print(f"Episode {episode + 1}: Reward = {episode_reward}")
    env.close()


def main():
    # Initialize the environment
    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")  # For evaluation purposes

    # Set seeds for reproducibility
    seed = 42
    reset_result = env.reset(seed=seed)
    if isinstance(reset_result, tuple):
        obs, _ = reset_result
    else:
        obs = reset_result
    reset_result = eval_env.reset(seed=seed + 1)
    if isinstance(reset_result, tuple):
        eval_obs, _ = reset_result
    else:
        eval_obs = reset_result
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate the DAPO agent
    dapo_agent = DAPO(
        env=env,
        policy=ActorCritic,
        learning_rate=1e-4,          # Reduced learning rate for stability
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        tau=0.1,                      # Temperature parameter
        tau_schedule=0.1,             # Constant tau for simplicity
        clip_range=0.2,               # Not used in DAPO-KL
        clip_range_vf=0.2,
        ent_coef=0.01,                # Small entropy coefficient for exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.01,
        tensorboard_log="./dapo_cartpole_tensorboard/",
        device="cpu",                 # Change to "cuda" if GPU is available
        seed=seed,
    )

    # Define total timesteps for training
    total_timesteps = 50000  # Adjust as needed

    # Start training
    print("Starting DAPO-KL training...")
    dapo_agent.learn(
        total_timesteps=total_timesteps,
        log_interval=10000,
        eval_env=eval_env,
        eval_interval=20000,
        eval_episodes=10,
    )
    print("Training completed.")

    # Save the trained model
    dapo_agent.save("dapo_cartpole.pth")

    # Evaluate the trained agent
    avg_reward = dapo_agent.evaluate(eval_env, episodes=100)
    print(f"Average Reward over 100 episodes: {avg_reward}")

    # Optionally, visualize the agent
    visualize = False  # Set to True to visualize
    if visualize:
        visualize_agent(env, dapo_agent, episodes=5)

    # Close environments
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()