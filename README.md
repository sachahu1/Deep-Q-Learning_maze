This repository contains the 2nd Reinforcement Learning coursework from the Department of Computing, Imperial College London, Academic Year 2019-2020, delivered by Dr A. Aldo Faisal and Dr Edward Johns. The coursework was developed with their PhD students. <br>

# DQN_RL_maze

This project consisted of building an DQN implementation to solve a maze problem where an agent had to reach a given goal position. The agent's stepsize was limited to 2 pixels. <br>
The random_environment.py and train_and_test.py files were given. <br>
The random_environment.py file creates random maze environments. <br>
The train_and_test.py creates an environment, trains the agent 10 minutes, and then tests it against a new maze.

Important features of the DQN implementation in the agent.py file comprise of: <br>
- Epsilon-greedy policy (line 34)
- Epsilon decay rate (line 35)
- Epsilon decay clipping (line 36)
- Experience replay buffer (lines 301-340)
- Prioritised experience replay (lines 322-340)
- Episode length (line 22)
- Reward function which penalises hitting the wall proportionally to the distance from the goal; gives 0 reward for moving vertically; rewards positively when entering a circle close to the reward state (lines 95-108)


An example of a **simple** maze, where the red dot is the agent and the green dot is the goal:

<img src="images/maze_simple_example.png" width=500>





## Requirements

You need to use Python 3.6 or greater.

## Installing the environment on a Unix system 

We created this repository to ensure that everybody uses exactly the same versions of the libraries.

To install the libraries, start by cloning this repository and enter the created folder:

```shell script
git clone https://github.com/sachahu1/Deep-Q-Learning_maze.git
```

Setting up a virtual environment (called ```venv``` here):

```shell script
python3 -m venv ./venv 
```

Then enter the environment:
```shell script
source venv/bin/activate
```

And install the libraries in the environment by launching the following command:
```shell script
pip3 install -r requirements.txt
```

This will install the following libraries (and their dependencies) in the virtual environment ```venv```:

- ```torch``` 
- ```opencv-python```
- ```numpy```

## How to run a script ?

Before launching your experiment, be sure to use the right virtual environment in your shell:
```shell script
source venv/bin/activate  # To launch in the project directory
```

Once you are in the right virtual environment, you can directly launch the scripts 
by using one of the following command:
```shell script
python3 ./train_and_test.py  # To launch the coursework script
```

## Leaving the virtual environment

If you want to leave the virtual environment, you just need to enter the following command:
```shell script
deactivate
```
