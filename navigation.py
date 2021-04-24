import click
import numpy as np
from unityagents import UnityEnvironment


def navigate():
    # Start environment.
    env = UnityEnvironment(file_name="environment\Banana.exe")

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
            break

    print("Score: {}".format(score))


if __name__ == "__main__":
    navigate()


