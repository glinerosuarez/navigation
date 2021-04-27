[//]: # (Image References)

[image1]: ezgif.com-gif-maker.gif "Trained Agent"

# Navigation

### Introduction

In this project, we will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  
Thus, the goal of our agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around 
agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete 
actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, our agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below. In this repo I am using the Windows (64-bit) environment, 
   you only need to select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) 
   if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Set `env_file` in the `settings.toml` file to the path where your environment file is stored.

3. Install the dependencies:
   - cd to the navigation directory.
   - activate your virtualenv.
   - run: `pip install -r requirements.txt` in your shell.

### Instructions

There are three options currently available to run `navigation.py`:

1. Run `python navigation.py -r` to navigate through the Banana environment with an agent that selects actions randomly.
   
2. Run `python navigation.py -t` to train an agent to solve the Banana environment while navigating through it. 
   Hyperparameters are specified in `settings.toml`, feel free to tune them to see if you can get better results! Also, 
   you can change `checkpoints_every` which controls how often the agent weights, and the plot of the performance are
   stored in the output dir.

3. Run `python navigation.py -n` to use the agent that was trained the longest(the highest number of episodes) to navigate 
   through the Banana environment.

