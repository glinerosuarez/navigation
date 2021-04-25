import click
import time
import torch
import numpy as np
from pathlib import Path
from model import QNetwork
from config import settings
from collections import deque
from typing import List, Deque
import matplotlib.pyplot as plt
from dqn_agent import Agent, Experience
from unityagents import UnityEnvironment, BrainParameters, BrainInfo


def navigate_randomly():
    # Start environment.
    env = UnityEnvironment(file_name=settings.env_file)

    # Environments contain brains which are responsible for deciding the actions of their associated agents.
    # Here we check for the first brain available, and set it as the default brain we will be controlling from Python.
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # Take random actions in the environment.
    action_size = brain.vector_action_space_size        # number of actions
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]             # get the current state
    score = 0                                           # initialize the score

    while True:
        action = np.random.randint(action_size)         # select an action
        env_info = env.step(action)[brain_name]         # send the action to the environment
        next_state = env_info.vector_observations[0]    # get the next state
        reward = env_info.rewards[0]                    # get the reward
        done = env_info.local_done[0]                   # see if episode has finished
        score += reward                                 # update the score
        state = next_state                              # roll over the state to next time step
        if done:                                        # exit loop if episode finished
            env.close()
            break

    print("Score: {}".format(score))


def learn_to_navigate():
    """train an agent to solve the Banana environment using a Deep Q-learning algorithm"""

    def create_output_files(
            model: QNetwork, scores: List[float],
            last_episode: int,
            output_dir: Path,
            checkpoint_dir: str
    ) -> None:
        """create the output files after training"""

        def save_model(model: QNetwork, output_file: Path) -> None:
            """Save the weights of the trained agent"""

            torch.save(model.state_dict(), output_file)

        def save_scores_plot(scores: List[float], output_file: Path) -> None:
            """Save the plot of the scores in the output dir"""

            # create plot
            fig: plt.Figure() = plt.figure()
            ax: plt.axes.SubplotBase = fig.add_subplot(111)
            plt.plot(np.arange(len(scores)), scores)
            plt.ylabel('Score')
            plt.xlabel('Episode #')

            # save to file
            plt.savefig(output_file)

        # file names.
        # training suffix.
        timestamp: str = time.strftime('%b-%d-%Y_%H%M', time.localtime())
        suffix: str = f"{last_episode}_{timestamp}"
        # model checkpoint.
        weights_file: Path = output_dir/checkpoint_dir/f"checkpoint_{suffix}.pth"
        # average score plot.
        avg_score_file: Path = output_dir/checkpoint_dir/f"avg_score_{suffix}.png"

        # create files.
        save_model(model, weights_file)
        save_scores_plot(scores, avg_score_file)

    def dqn(n_episodes: int, max_t: int, eps_start: float, eps_end: float, eps_decay: float) -> None:
        """Deep Q-Learning.
        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """

        scores: List[float] = []                                                # list containing scores from each episode
        scores_window: Deque[float] = deque(maxlen=settings.score_window_size)  # last settings.score_window_size scores
        eps: float = eps_start                                                  # initialize epsilon

        for i_episode in range(1, n_episodes + 1):
            env_info: BrainInfo = env.reset(train_mode=True)[brain_name]        # reset the environment
            state: np.ndarray = env_info.vector_observations[0]                 # get the current state
            score: float = 0

            for t in range(max_t):
                action: int = agent.act(state, eps)
                env_info: BrainInfo = env.step(action)[brain_name]
                next_state: np.ndarray = env_info.vector_observations[0]        # get the next state
                reward: float = env_info.rewards[0]                             # get the reward
                done: bool = env_info.local_done[0]                             # see if episode has finished
                agent.step(Experience(state, action, reward, next_state, done))
                state = next_state
                score += reward
                if done:
                    break

            scores_window.append(score)                                         # save most recent score
            scores.append(score)                                                # save most recent score

            eps = max(eps_end, eps_decay * eps)                                 # decrease epsilon

            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % settings.score_window_size == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window) >= settings.solved_at:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                    i_episode - settings.score_window_size, np.mean(scores_window)
                ))
                # The env is solved, save the outputs.
                create_output_files(
                    agent.qnetwork_local,
                    scores,
                    i_episode,
                    Path()/settings.output_dir,
                    settings.checkpoints_dir
                )
                break

        # episodes reached, save the outputs.
        create_output_files(
            agent.qnetwork_local,
            scores,
            n_episodes,
            Path() / settings.output_dir,
            settings.checkpoints_dir
        )

    # Start environment.
    env: UnityEnvironment = UnityEnvironment(file_name=settings.env_file, seed=settings.seed)

    # Environments contain brains which are responsible for deciding the actions of their associated agents.
    # Here we check for the first brain available, and set it as the default brain we will be controlling from Python.
    brain_name: str = env.brain_names[0]
    brain: BrainParameters = env.brains[brain_name]

    # Accumulate experience and train the agent.
    action_size: int = brain.vector_action_space_size                                       # number of actions
    state_size: int = len(env.reset(train_mode=False)[brain_name].vector_observations[0])   # get the current stat

    agent: Agent = Agent(state_size, action_size, settings.seed)

    dqn(settings.episodes, settings.max_t, settings.eps_start, settings.eps_end, settings.eps_decay)


if __name__ == "__main__":
    learn_to_navigate()


