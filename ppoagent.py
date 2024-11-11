import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Any, Dict, Optional, Type, Union, Callable, Tuple


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
    Actor-Critic network for PPO.
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

        # Critic head
        self.critic = nn.Sequential(
            self._build_mlp_layer(hidden_sizes[-1], hidden_sizes[-1]),
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


class PPO:
    """
    Proximal Policy Optimization (PPO) algorithm.
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
        clip_range: Union[float, Callable[[float], float]] = 0.2,
        clip_range_vf: Optional[Union[float, Callable[[float], float]]] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = 0.05,
        tensorboard_log: Optional[str] = None,
        device: Union[torch.device, str] = "cpu",
        seed: Optional[int] = None,
    ):
        """
        Initialize the PPO algorithm.

        Args:
            env (gym.Env): The environment to train on.
            policy (Type[ActorCritic]): The policy network class.
            learning_rate (Union[float, Callable[[float], float]], optional): Learning rate or schedule.
            n_steps (int, optional): Number of steps per rollout.
            batch_size (int, optional): Batch size for updates.
            n_epochs (int, optional): Number of epochs per update.
            gamma (float, optional): Discount factor.
            gae_lambda (float, optional): GAE lambda.
            clip_range (Union[float, Callable[[float], float]], optional): Clipping range.
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
        self.clip_range = get_schedule_fn(clip_range)
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
            # Here, you might need to track if the last state was done
            last_done = False  # Modify based on your specific use-case or track it during rollout
        self.rollout_buffer.compute_returns_and_advantages(last_value, last_done)

        # Normalize advantages
        advantages = (
            self.rollout_buffer.advantages[: self.rollout_buffer.size]
        )
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

        kl_total = 0.0
        kl_count = 0

        for epoch in range(self.n_epochs):
            # Create random minibatch indices
            indices = np.arange(self.rollout_buffer.size)
            np.random.shuffle(indices)
            for start in range(0, self.rollout_buffer.size, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Evaluate current policy
                values, log_probs, entropies = self.policy.evaluate_actions(batch_obs, batch_actions)
                values = values.squeeze(-1)

                # Ratio for clipping
                ratio = torch.exp(log_probs - batch_old_log_probs)
                clip_range = self.clip_range(1.0)
                clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                policy_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()

                # Value function loss
                if self.clip_range_vf is not None:
                    # Clipped value function
                    value_pred_clipped = self.rollout_buffer.values[: self.rollout_buffer.size][batch_indices]
                    value_pred_clipped = torch.from_numpy(value_pred_clipped).float().to(self.device) + \
                        torch.clamp(
                            values - torch.from_numpy(self.rollout_buffer.values[: self.rollout_buffer.size][batch_indices]).float().to(self.device),
                            -self.clip_range_vf(1.0),
                            self.clip_range_vf(1.0)
                        )
                    value_loss1 = F.mse_loss(values, batch_returns)
                    value_loss2 = F.mse_loss(value_pred_clipped, batch_returns)
                    value_loss = torch.max(value_loss1, value_loss2).mean()
                else:
                    value_loss = F.mse_loss(values, batch_returns).mean()

                # Entropy loss
                entropy_loss = -entropies.mean()

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                # Optimize the policy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Calculate KL divergence for the batch
                kl_div = (batch_old_log_probs - log_probs).mean().item()
                kl_total += kl_div
                kl_count += 1

                # Early stopping based on average KL divergence
                avg_kl = kl_total / kl_count
                if self.target_kl is not None and avg_kl > 1.5 * self.target_kl:
                    print(f"Early stopping at epoch {epoch} due to exceeding target KL.")
                    return

                # Logging
                if self.writer is not None:
                    self.writer.add_scalar("loss/policy", policy_loss.item(), self.total_steps)
                    self.writer.add_scalar("loss/value", value_loss.item(), self.total_steps)
                    self.writer.add_scalar("loss/entropy", entropy_loss.item(), self.total_steps)
                    self.writer.add_scalar("metrics/kl_divergence", kl_div, self.total_steps)

        # Logging total loss
        if self.writer is not None:
            self.writer.add_scalar("loss/total", loss.item(), self.total_steps)

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
        Train the PPO agent.

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

                if self.total_steps >= log_interval and self.total_steps % log_interval < self.n_steps:
                    print(f"Total Steps: {self.total_steps}")

                if eval_env is not None and self.total_steps >= eval_interval and self.total_steps % eval_interval < self.n_steps:
                    avg_reward = self.evaluate(eval_env, eval_episodes)
                    print(f"Evaluation at step {self.total_steps}: Average Reward: {avg_reward}")
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

# test_ppo.py

import gym
import torch
# from ppo_module import ActorCritic, PPO

def main():
    # Initialize the environment
    env = gym.make("CartPole-v1")
    eval_env =gym.make("CartPole-v1")
    # Set seeds for reproducibility
    seed = 42
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 1)
    torch.manual_seed(seed)
    import numpy as np
    np.random.seed(seed)

    # Instantiate the PPO agent
    ppo_agent = PPO(
        env=env,
        policy=ActorCritic,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.01,
        tensorboard_log="./ppo_cartpole_tensorboard/",
        device="cpu",  # Change to "cuda" if GPU is available
        seed=seed,
    )

    # Define total timesteps for training
    total_timesteps = 100000  # Adjust as needed

    # Start training
    print("Starting training...")
    ppo_agent.learn(
        total_timesteps=total_timesteps,
        log_interval=10000,
        eval_env=eval_env,
        eval_interval=20000,
        eval_episodes=10,
    )
    print("Training completed.")

    # Save the trained model
    ppo_agent.save(r"C:\Users\heman\Desktop\Notes\ppo_cartpole.pth")

    # Evaluate the trained agent
    # avg_reward = ppo_agent.evaluate(eval_env, episodes=1000)
    # print(f"Average Reward over 20 episodes: {avg_reward}")

    # Optionally, visualize the agent
    visualize = True
    if visualize:
        visualize_agent(env, ppo_agent)

    # Close environments
    env.close()
    eval_env.close()

def visualize_agent(env: gym.Env, agent: PPO, episodes: int = 40) -> None:
    """
    Visualize the trained agent.

    Args:
        env (gym.Env): The environment to visualize on.
        agent (PPO): The trained PPO agent.
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

if __name__ == "__main__":
    main()