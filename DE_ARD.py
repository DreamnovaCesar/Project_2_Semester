import os
import numpy as np
import pandas as pd
from sklearn.linear_model import ARDRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from src.Utils.DataProcessor import DataProcessor
from typing import Tuple

from src.Utils.Config import Config
from src.Utils.Utils import save_to_csv

import logging
import time

# * ? Main usage:
def DE_main_ARD() -> None:

    # * Bees
    List_individuals = [30, 60, 90];

    for _, Num_individuals in enumerate(List_individuals):
        for _, (Name, CSV) in enumerate(Config.Euler_csv_files.items()):
            
            Meta_name = "DE_ARD";

            # * DE parameters
            #Num_individuals = 30;

            # * Maximum number of iterations
            Max_iter = 10;

            # * Number of hyperparameters to optimize (alpha and max_iter)
            DIM = 4;

            # * Lower bounds for the hyperparameters
            LB = np.array([1e-6, 1e-6, 1e-6, 1e-6]);

            # * Upper bounds for the hyperparameters
            UB = np.array([1.0, 1.0, 1.0, 1.0]);

            # * Mutation factor
            F = 0.8;

            # * Crossover probability
            CR = 0.7;

            # * Instantiate DataProcessor
            DP = DataProcessor(CSV);

            # * Read features and labels from CSV
            X, y = DP.read_csv_and_extract_features_labels();

            # * Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42);

            # * Define the objective function
            def objective_function(Hyperparams : np.ndarray) -> float:
                """
                Evaluate the performance of the LassoLars model with given hyperparameters.
                
                Parameters
                ----------
                hyperparams : np.ndarray
                    Array of hyperparameters [alpha, max_iter].
                
                Returns
                -------
                float
                    Mean squared error on the test set.
                """
                
                Alpha_1, Alpha_2, Lambda_1, Lambda_2 = Hyperparams;

                print(F"Objective function (Alpha 1: {Alpha_1}, Alpha 2: {Alpha_2}, Lambda 1: {Lambda_1}, Lambda 1: {Lambda_2})\n");

                Model = ARDRegression(alpha_1 = Alpha_1, alpha_2 = Alpha_2, lambda_1 = Lambda_1, lambda_2 = Lambda_2);

                Model.fit(X_train, y_train);

                y_pred = Model.predict(X_test);

                MSE = mean_squared_error(y_test, y_pred);

                return MSE

            # * Initialize population
            def initialize_population(Num_individuals: int, DIM: int, LB: np.ndarray, UB: np.ndarray) -> np.ndarray:
                """
                Initialize the population with random values within the given bounds.
                
                Parameters
                ----------
                Num_individuals : int
                    Number of individuals in the population.
                DIM : int
                    Number of dimensions (hyperparameters) for each individual.
                LB : np.ndarray
                    Lower bounds for the hyperparameters.
                UB : np.ndarray
                    Upper bounds for the hyperparameters.
                
                Returns
                -------
                np.ndarray
                    Initialized population.
                """

                # * Initialize population with random values
                IP = np.random.uniform(low = LB, high = UB, size = (Num_individuals, DIM));
                print(F"Initialize population : {IP}\n");
                return IP

            # * Evaluate population
            def evaluate_population(Population: np.ndarray) -> np.ndarray:
                """
                Evaluate the entire population.
                
                Parameters
                ----------
                Population : np.ndarray
                    Array of individuals (population) to evaluate.
                
                Returns
                -------
                np.ndarray
                    Array of fitness scores (MSE) for each individual.
                """

                # * Evaluate each individual in the population
                return np.array([objective_function(IND) for IND in Population])

            # * Perform mutation
            def mutation(Population: np.ndarray, Index: int, F: float, LB: np.ndarray, UB: np.ndarray) -> np.ndarray:
                """
                Perform mutation to create a mutant vector.
                
                Parameters
                ----------
                Population : np.ndarray
                    Array of individuals (population).
                Index : int
                    Index of the current individual.
                F : float
                    Mutation factor.
                LB : np.ndarray
                    Lower bounds for the hyperparameters.
                UB : np.ndarray
                    Upper bounds for the hyperparameters.
                
                Returns
                -------
                np.ndarray
                    Mutant vector.
                """

                # * Get indices of individuals excluding current index
                Indexes = [i for i in range(len(Population)) if i != Index];

                # * Select three random individuals for mutation
                A, B, C = Population[np.random.choice(Indexes, 3, replace=False)];

                print(F"Mutation A: {A}");
                print(F"Mutation B: {B}");
                print(F"Mutation C: {C}\n");

                # * Create mutant vector and clip to bounds
                Mutant = np.clip(A + F * (B - C), LB, UB);

                return Mutant

            # * Perform crossover
            def crossover(Target: np.ndarray, Mutant: np.ndarray, CR: float) -> np.ndarray:
                """
                Perform crossover to generate a trial vector.
                
                Parameters
                ----------
                Target : np.ndarray
                    Target vector (current individual).
                Mutant : np.ndarray
                    Mutant vector.
                CR : float
                    Crossover probability.
                
                Returns
                -------
                np.ndarray
                    Trial vector.
                """

                # * Get dimension of the target vector
                DIM = len(Target);
                
                # * Generate crossover points based on crossover probability
                Cross_points = (np.random.rand(DIM) < CR);
                
                print(F"Cross_points: {Cross_points}");

                # * Ensure at least one crossover point is selected
                if not np.any(Cross_points):
                    Cross_points[np.random.randint(0, DIM)] = True;
                
                # * Create trial vector by combining target and mutant
                Trial = np.where(Cross_points, Mutant, Target);

                print(F"Trial: {Trial}\n");

                return Trial

            # * Perform selection
            def selection(Population: np.ndarray, Fitness: np.ndarray, Trial: np.ndarray, Score_trial: float, Index: int) -> Tuple[np.ndarray, np.ndarray]:
                """
                Perform selection to choose the better vector between the target and the trial.
                
                Parameters
                ----------
                Population : np.ndarray
                    Array of individuals (population).
                Fitness : np.ndarray
                    Array of fitness scores for the population.
                Trial : np.ndarray
                    Trial vector.
                Score_trial : float
                    Fitness score of the trial vector.
                Index : int
                    Index of the current individual.
                
                Returns
                -------
                Tuple[np.ndarray, np.ndarray]
                    Updated population and fitness arrays.
                """

                # * If trial vector is better, update population and fitness
                if(Score_trial < Fitness[Index]):
                    
                    print(F"Population: {Trial}");
                    print(F"Fitness: {Score_trial}\n");

                    Population[Index] = Trial;
                    Fitness[Index] = Score_trial;
                
                return Population, Fitness

            # * Differential Evolution main loop
            def differential_evolution(Num_individuals: int, Max_iter: int, DIM: int, LB : np.ndarray, UB : np.ndarray, F : float, CR: float) -> Tuple[np.ndarray, float]:
                """
                Differential Evolution main loop.
                
                Parameters
                ----------
                Num_individuals : int
                    Number of individuals in the population.
                Max_iter : int
                    Maximum number of iterations.
                DIM : int
                    Number of dimensions (hyperparameters) for each individual.
                LB : np.ndarray
                    Lower bounds for the hyperparameters.
                UB : np.ndarray
                    Upper bounds for the hyperparameters.
                F : float
                    Mutation factor.
                CR : float
                    Crossover probability.
                
                Returns
                -------
                Tuple[np.ndarray, float]
                    Best individual and best score (MSE) found by the algorithm.
                """

                # * Initialize population
                Population = initialize_population(Num_individuals, DIM, LB, UB);

                # * Evaluate initial population
                Fitness = evaluate_population(Population);

                # * Find the index of the best individual
                Best_idx = np.argmin(Fitness);
                Best_individual = Population[Best_idx];
                Best_score = Fitness[Best_idx];
                
                # * Lists to store the best individuals and their scores for each iteration
                Best_individuals = [];
                Best_scores = [];

                # * Initialize best_iteration counter
                Best_iteration = 0;

                for j in range(Max_iter):
                    for i in range(Num_individuals):

                        Mutant = mutation(Population, i, F, LB, UB);

                        Trial = crossover(Population[i], Mutant, CR);

                        # * Evaluate the trial vector
                        Score_trial = objective_function(Trial);

                        Population, Fitness = selection(Population, Fitness, Trial, Score_trial, i);
                        
                        # * Append current individual and score to lists
                        Best_individuals.append(list(Trial));
                        Best_scores.append(Score_trial);

                        if(Score_trial < Best_score):

                            Best_individual = Trial;
                            Best_score = Score_trial;
                            Best_iteration = j + 1;
                    
                    print(f"Iteration {j + 1}/{Max_iter}, Best Score: {Best_score}\n");

                # * Create a DataFrame with the evolution of individuals and their scores
                Dataframe = pd.DataFrame({
                    'Alpha': [ind[0] for ind in Best_individuals],
                    'Max_iter': [ind[1] for ind in Best_individuals],
                    'MSE': Best_scores
                })

                return Best_individual, Best_score, Best_iteration, Dataframe

            # * Run DE algorithm
            Start_time = time.time();
            Best_individual, Best_score, Best_iteration, Dataframe_data = differential_evolution(Num_individuals, Max_iter, DIM, LB, UB, F, CR)
            End_time = time.time();
            
            Execution_time = End_time - Start_time;

            print(f"Optimal Hyperparameters: {Best_individual}");
            print(f"Optimal MSE: {Best_score}");

            # * Create the LassoLars model with the optimized hyperparameters
            Alpha_1, Alpha_2, Lambda_1, Lambda_2 = Best_individual;
            Model = ARDRegression(alpha_1 = Alpha_1, alpha_2 = Alpha_2, lambda_1 = Lambda_1, lambda_2 = Lambda_2);

            # * Train the model with optimal hyperparameters
            Model.fit(X_train, y_train);

            # * Predict on the test set with optimal hyperparameters
            y_pred = Model.predict(X_test);

            # * Calculate the final mean squared error
            Final_mse = mean_squared_error(y_test, y_pred);
            print(f"Final MSE with optimal hyperparameters: {Final_mse}");

            # * Extract and print the coefficients
            Coefficients = Model.coef_;

            Coefficients_rounded = [round(coef, 3) if coef != 0.0 else 0.0 for coef in Coefficients];
            Non_zero_indices = np.nonzero(Coefficients_rounded)[0];
            Non_zero_count = len(Non_zero_indices);
            
            print(f"Theta found by stomp : {Coefficients_rounded}");
            print(f"Non-zero indices : {Non_zero_indices}");
            print(f"Non-zero : {Non_zero_count}");
            
            # * Save coefficients and related information to a DataFrame
            Dataframe_idx = pd.DataFrame({
                'Coefficient': [Coefficients_rounded],
                'Non-zero indices': [Non_zero_indices],
                'Non-zero count': Non_zero_count
            })

            File_path_meta = os.path.join(Config.Meta_path, Meta_name);

            # * Check if directory exists
            if not os.path.exists(File_path_meta):
                # * If directory doesn't exist, create it
                os.makedirs(File_path_meta);
            
            File_path_unique = f"{Meta_name}_{Num_individuals}_Coefficients_Info_{Name}.csv";
            CSV_file_path = os.path.join(File_path_meta, File_path_unique);
            Dataframe_idx.to_csv(CSV_file_path, index=False);

            # * Save evolution to CSV
            CSV_file_data = os.path.join(File_path_meta, f"{Meta_name}_{Num_individuals}_{Name}.csv");
            save_to_csv(CSV_file_data, Dataframe_data);

            Log_file = os.path.join(File_path_meta, f"{Meta_name}_{Num_individuals}_log.txt");

            # * Configure logging to write to a file
            logging.basicConfig(filename = Log_file, level=logging.INFO);

            # * Log best individual and score
            logging.info(f"Dataset: {Name}");
            logging.info(f"Name: {Meta_name}_{Num_individuals}_log.txt");
            logging.info(f"Optimal Hyperparameters: {Best_individual}");
            logging.info(f"Optimal MSE: {Best_score}");
            logging.info(f"Found on iteration: {Best_iteration}");
            logging.info(f"Time: {Execution_time}\n");

'''# * ? Main usage:
if __name__ == "__main__":
    DE_main_ARD();'''
