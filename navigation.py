import sys
import time
import torch
import random
import numpy as np
from pathlib import Path
from model import QNetwork
from config import settings
from collections import deque
import matplotlib.pyplot as plt
from typing import List, Deque, Tuple
from dqn_agent import Agent, Experience
from argparse import ArgumentParser, Namespace
from unityagents import UnityEnvironment, BrainParameters, BrainInfo


def navigate_randomly():
    # Init environment.
    env, brain_name, state_size, action_size, _ = init_env(settings.env_file, train_mode=False)

    # Take random actions in the environment.
    score: float = 0                                            # initialize the score

    while True:
        action: int = np.random.randint(action_size)            # select an action
        env_info: BrainInfo = env.step(action)[brain_name]      # send the action to the environment
        reward: float = env_info.rewards[0]                     # get the reward
        done: bool = env_info.local_done[0]                     # see if episode has finished
        score += reward                                         # update the score
        if done:                                                # exit loop if episode finished
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

    def dqn(
            agent: Agent,
            env: UnityEnvironment,
            n_episodes: int,
            max_t: int,
            eps_start: float,
            eps_end: float,
            eps_decay: float
    ) -> None:
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
                env.close()
                break

        # episodes reached, save the outputs.
        create_output_files(
            agent.qnetwork_local,
            scores,
            n_episodes,
            Path() / settings.output_dir,
            settings.checkpoints_dir
        )
        env.close()

    # Init environment.
    env, brain_name, state_size, action_size, _ = init_env(settings.env_file, train_mode=True, seed=settings.seed)

    # Init agent.
    agent: Agent = Agent(state_size, action_size, settings.seed)

    # Train agent.
    dqn(agent, env, settings.episodes, settings.max_t, settings.eps_start, settings.eps_end, settings.eps_decay)


def navigate_smart(model_file: Path) -> None:
    """Take the model trained the longest to navigate through the Banana environment"""

    # Init environment.
    env, brain_name, state_size, action_size, state = init_env(settings.env_file, train_mode=False)
    # Load last trained model.
    agent: Agent = Agent(state_size, action_size, random.randint(0, 100))
    agent.qnetwork_local.load_state_dict(torch.load(model_file))
    agent.qnetwork_local.eval()

    score: float = 0.0

    while True:
        action: int = agent.act(state)
        env_info: BrainInfo = env.step(action)[brain_name]
        next_state: np.ndarray = env_info.vector_observations[0]
        reward: float = env_info.rewards[0]
        done: bool = env_info.local_done[0]
        state = next_state
        score += reward
        if done:
            break

    print("Score: {}".format(score))


if __name__ == "__main__":

    def init_env(
            env_file: str,
            train_mode: bool = True,
            seed: int = random.randint(0, 100)
    ) -> Tuple[UnityEnvironment, str, int, int, Tuple[float]]:
        """initialize Banana UnityEnvironment"""

        env: UnityEnvironment = UnityEnvironment(file_name=env_file, seed=seed)

        # Environments contain brains which are responsible for deciding the actions of their associated agents.
        # Here we check for the first brain available, and set it as the default brain we will be controlling from Python.
        brain_name: str = env.brain_names[0]
        brain: BrainParameters = env.brains[brain_name]

        # Accumulate experience and train the agent.
        action_size: int = brain.vector_action_space_size                                       # number of actions
        state: Tuple[float] = env.reset(train_mode)[brain_name].vector_observations[0]          # initial state
        state_size: int = len(state)                                                            # get the current stat

        return env, brain_name, state_size, action_size, state

    def get_last_model_file(checkpoints_dir: Path) -> Path:
        """Return the saved model which was trained the longest"""
        try:
            last_model_file: Path = Path(max([str(p) for p in checkpoints_dir.rglob("*.pth")]))
            print(f"loading model {last_model_file}")
        except ValueError:
            print("It seems there is no trained agent yet, run navigation.py --train to train an agent")
            sys.exit()
        return last_model_file

    navigate_smart(get_last_model_file(Path() / settings.output_dir / settings.checkpoints_dir))

    parser: ArgumentParser = ArgumentParser()
    parser.add_argument(
        "-t",
        "--train",
        help="train an agent to solve the Banana environment while navigating through it",
        action="store_true"
    )
    parser.add_argument(
        "-n",
        "--navigate",
        help="use the agent that was trained the longest to navigate through the Banana environment, "
             "this is the default option",
        action="store_true"
    )
    parser.add_argument(
        "-r",
        "--random",
        help="use an agent that chooses actions at random to navigate through the Banana environment",
        action="store_true"
    )

    args: Namespace = parser.parse_args()

    if args.train:
        learn_to_navigate()
    elif args.random:
        navigate_randomly()
    else:
        navigate_smart(get_last_model_file(Path()/settings.output_dir/settings.checkpoints_dir))
