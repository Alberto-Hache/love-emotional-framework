# Code and data for "A Generic Self-Learning Emotional Framework for Machines"

This repository contains the code and data to reproduce the results presented in the paper:

- **Title:** A Generic Self-Learning Emotional Framework for Machines  
- **Authors:** Alberto Hernández-Marcos, Eduardo Ros  
- **Publisher:** Springer Nature  
- **Journal:** *Scientific Reports*  
- **DOI:** 10.1038/s41598-024-72817-x
- **Article Link:** [nature.com/articles/s41598-024-72817-x](https://www.nature.com/articles/s41598-024-72817-x)

## Citation
If you use this code or data, please cite the paper:

Hernández-Marcos, A., Ros, E. A generic self-learning emotional framework for machines. Sci Rep 14, 25858 (2024). https://doi.org/10.1038/s41598-024-72817-x


## Repository contents

* [`data/`](./data/) Containing all the data used and generated in the original paper.
* [`code/`](./code/) Containing the notebooks, as well as some auxiliary modules they use.
* [`extra/`](./extra/EXTRA_CONTENT.md) Extra information about the framework.

### Step-by-step details

The whole process described in the article is reproduced in consecutive, independent notebooks, each taking all inputs from the [`data/original/`](./data/original/) folder.
In order to preserve the original inputs of each notebook, the outputs produced are saved into the [`data/new_runs/`](./data/new_runs/) folder.
For maximum simplicity, all the notebooks install and import the libraries they need from scratch (not using any virtual environment).

For detailed explanations of the framework, methods and experiments, please refer to the paper.


### [Step 0. Pretraining of a conventional RL agent]
The methodology introduced is independent from the RL algorithm and library used, as long as stepwise trajectories of the agent are registered for the required values (e.g. reward and state-value at each time-step).
Examples of the trajectories with the format used by the framework can be seen in folder [`/data/original/trajectories/`](./data/original/trajectories/).

We recommend the open-source library [OpenAI’s Spinning Up](https://spinningup.openai.com/en/latest/) [Achiam, 2018], compatible with [OpenAI’s Gym](https://github.com/openai/gym) [Brockman, 2016], because of its modular, well-documented implementation of RL algorithms like PPO (Proximal Policy Optimization).

### Step 1. Trajectory dataset preparation
Exploration and creation of a dataset of RL trajectories from the previously trained RL agent. The trajectories are split in a train set and a test set.

Notebook: [01_trajectory_dataset_preparation.ipynb](./code/01_trajectory_dataset_preparation.ipynb)

### Step 2. Emotion learning
Training of an emotional encoder based on the dataset with RL trajectories from a previously trained RL agent.

Notebook: [02_emotion_learning.ipynb](./code/02_emotion_learning.ipynb)

### Step 3. Interpretation of the latent emotional space
Identification of the patterns learned by the emotional encoder trained.

Notebook: [03_interpretation_of_latent_space.ipynb](./code/03_interpretation_of_latent_space.ipynb)

### Step 4. Experimental validation of learned emotions with humans

**I. Emotional attribution test with humans**

Distinguishability and correspondence of the learned emotions with independent human observations.

**II. Mapping PAD values to referenced PAD values**

Mapping of the distribition of PAD values obtained for each learned emotion vs academy experimental references from human subjects.

Notebook: [04_experimental_validation_of_learned_emotions.ipynb](./code/04_experimental_validation_of_learned_emotions.ipynb)