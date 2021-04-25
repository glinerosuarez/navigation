import random
import numpy as np
from model import QNetwork
from config import settings
from collections import deque
from typing import List, Tuple
from dataclasses import dataclass

import torch
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F

# Select gpu if available.
DEVICE: torch._C.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class Experience:
    """Represents the experience gathered by the model"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class Agent:
    """Interacts with and learns from the environment."""

    @staticmethod
    def soft_update(local_model: QNetwork, target_model: QNetwork, tau: float):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def __init__(self, state_size: int, action_size: int, seed: int):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        random.seed(seed)

        self.state_size: int = state_size
        self.action_size: int = action_size

        # Q-Network
        self.qnetwork_local: QNetwork = QNetwork(state_size, action_size, seed).to(DEVICE)
        self.qnetwork_target: QNetwork = QNetwork(state_size, action_size, seed).to(DEVICE)
        self.optimizer: optim.Adam = optim.Adam(self.qnetwork_local.parameters(), lr=settings.agent.lr)

        # Replay memory
        self.memory: ReplayBuffer = ReplayBuffer(
            action_size,
            settings.agent.replay_buffer_size,
            settings.agent.batch_size,
            seed,
        )
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step: int = 0

    def step(self, exp: Experience) -> None:
        # Save experience in replay memory
        self.memory.add(exp)

        # Learn every update_every time steps.
        self.t_step = (self.t_step + 1) % settings.agent.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > settings.agent.batch_size:
                experiences: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor] = self.memory.sample()
                self.learn(experiences, settings.agent.gamma)

    def act(self, state: np.ndarray, eps: float = 0.) -> int:
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state: Tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values: Tensor = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return int(np.argmax(action_values.cpu().data.numpy()))
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], gamma: float):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        q_targets_next: Tensor = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        q_targets: Tensor = rewards + (gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected: Tensor = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss: Tensor = F.mse_loss(q_expected, q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, settings.agent.tau)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size: int, buffer_size: int, batch_size: int, seed: int):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """

        random.seed(seed)

        self.action_size: int = action_size
        self.memory: deque[Experience] = deque(maxlen=buffer_size)
        self.batch_size: int = batch_size
        self.seed: int = seed

    def add(self, exp: Experience) -> None:
        """Add a new experience to memory."""

        self.memory.append(exp)

    def sample(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Randomly sample a batch of experiences from memory."""

        experiences: List[Experience] = random.sample(self.memory, k=self.batch_size)

        states: Tensor = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None]))\
            .float()\
            .to(DEVICE)
        actions: Tensor = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None]))\
            .long()\
            .to(DEVICE)
        rewards: Tensor = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None]))\
            .float()\
            .to(DEVICE)
        next_states: Tensor = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None]))\
            .float()\
            .to(DEVICE)
        dones: Tensor = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8))\
            .float()\
            .to(DEVICE)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return the current size of internal memory."""

        return len(self.memory)
