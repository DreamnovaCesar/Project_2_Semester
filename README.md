![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
[![Build Status](https://travis-ci.org/anfederico/clairvoyant.svg?branch=master)](https://travis-ci.org/anfederico/clairvoyant)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

<a name="readme-top"></a>

<span style="font-size:2em;"> $${\color{red}{\text{This is version 0.5. Version 1.0 is planned to incorporate this algorithm into a library.}}}$$</span>


# Euler-number-3D-using-Regression-Models

Hey there! I see you're interested in learning about the Euler number. This is a fundamental concept in mathematics that helps us understand the topology of objects in 2D and 3D space.

Let's start with 2D objects. In this case, the Euler number is calculated as the number of connected components (also known as "holes") in a given object minus the number of its boundaries (also known as "genus"). Essentially, it tells us how many holes an object has, and if it's open or closed. The formula for the Euler number in 2D is:

Euler number = Number of connected components - Number of boundaries

It's important to note that the Euler number is always an integer and can be negative, zero, or positive. For example, if an object has one hole and one boundary, the Euler number would be zero. Now, let's move on to 3D objects. The Euler number in this case is calculated in a similar way, but it takes into account not only the number of holes and boundaries, but also the number of handles (or "tunnels"). The formula for the Euler number in 3D is:

Euler number = Number of connected components - Number of handles + Number of boundaries

We present a new library designed to simplify the analysis of Euler characteristics. This program addresses the difficulties involved in generating 3D test objects and the complexities of extracting Octo-Voxel patterns. The library uses a novel method to rapidly generate data, AND extract descriptors by using effective multiprocessing. Furthermore, a method for extracting discrete CHUNKS from an image has been developed, allowing for separate multiprocessing assessment. This method accelerates the process of combination extraction and offers researchers a quick and effective way to look into Euler characteristics in a variety of applications. Our system provides a comprehensive solution for researchers looking for effective ways to create and analyze data, which will advance the discovery of Euler characteristics across a wide range of areas.

## Regression Model

`ARD Regression` : ARD regression is a Bayesian regression technique that automatically determines the relevance of input features by estimating their importance. It assigns a coefficient to each input feature, and the model automatically selects the most relevant features while shrinking the coefficients of less relevant ones towards zero. This technique helps prevent overfitting by effectively performing feature selection and regularization simultaneously.

`LassoLars` : Lasso Lars, or Least Angle Regression Lasso, is a variant of Lasso regression that combines the least angle regression (Lars) algorithm with Lasso regularization. It iteratively updates the coefficients of the model, gradually adding features that are most correlated with the target variable while simultaneously shrinking the coefficients of less relevant features towards zero.

## Metaheuristic Algorithms

`Artificial Bee Colony (ABC)` : The Artificial Bee Colony algorithm is inspired by the foraging behavior of honey bees. It mimics the process of bees searching for food, sharing information, and selecting optimal solutions. ABC is widely used for optimization problems due to its simplicity and effectiveness in exploring large search spaces.

`Differential Evolution (DE)` : Differential Evolution is a population-based optimization algorithm that works through the differential variation of solutions. It involves mutation, crossover, and selection processes to evolve candidate solutions. DE is known for its robustness and efficiency in handling complex, multi-dimensional optimization problems.

`Harris Hawks Optimization (HHO)` : The Harris Hawks Optimization algorithm simulates the cooperative hunting strategy of Harris's hawks. It employs a dynamic strategy to switch between exploration and exploitation phases, improving its capability to escape local optima and find global solutions. HHO is particularly effective in solving non-linear and multimodal optimization problems.

`Particle Swarm Optimization (PSO)`: Particle Swarm Optimization is inspired by the social behavior of bird flocking. It optimizes problems by having a population of candidate solutions, called particles, that move through the search space influenced by their own and their neighbors' best-known positions. PSO is simple to implement and effective in converging quickly to optimal solutions.

## Setup

To set up a virtual environment with Anaconda (we utilized Python 3.11.7), adhere to the steps outlined below:

Launch the Anaconda Prompt by navigating to the Start menu and searching for "Anaconda Prompt".
Create a new virtual environment named "tfenv" (You can name as you pleased) by executing the following command:

```python
conda create --name tfenv
```

Activate the virtual environment by entering the command:

```python
conda activate tfenv
```

Finally, install the dependencies listed in requirements.txt by running:

```python
conda install requirements.txt
```

By following these steps, you'll successfully create a virtual environment with using Anaconda.

## Code information

- `src` : Contains the main code packages and scripts for execution. This package is divided into:
    - `Data` : Folder for internal storage and CSV files containing patterns extracted previously from various images.
      - `Metaheuristic` : Directory housing outcomes from different metaheuristic algorithms such as ABC, DE, HHO, and PSO.
    - `Utils` : Module with various functions for configuration and variable handling, reusable across the project.
      - `Utils.py` : File containing essential functions for project-wide use.
        - `ABC_ARD.py`: Python script for training the ARD regression model using the ABC metaheuristic.
        - `ABC_LL.py`: Python script for training the LassoLars regression model using the ABC metaheuristic.

        - `DE_ARD.py`: Python script for training the ARD regression model using the DE metaheuristic.
        - `DE_LL.py`: Python script for training the LassoLars regression model using the DE metaheuristic.

        - `HHO_ARD.py`: Python script for training the ARD regression model using the HHO metaheuristic.
        - `HHO_LL.py`: Python script for training the LassoLars regression model using the HHO metaheuristic.

        - `PSO_ARD.py`: Python script for training the ARD regression model using the PSO metaheuristic.
        - `PSO_LL.py`: Python script for training the LassoLars regression model using the PSO metaheuristic.

- `main.py` : This is the main function of testing each of the scripts above.

## Info

All the information for each data set is stored in each of the folders within `Metaheuristic`. The files ABC_ARD_log.txt and ABC_LL_log.txt contain the training information for each.


### Built With

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)&nbsp;

### Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

### 🤝🏻 &nbsp;Connect with Me

<p align="center">
<a href="https://www.linkedin.com/in/cesar-eduardo-mu%C3%B1oz-chavez-a00674186/"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/></a>
<a href="https://twitter.com/CesarEd43166481"><img src="https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white"/></a>