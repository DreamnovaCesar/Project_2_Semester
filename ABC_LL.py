import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLars
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from src.Utils.DataProcessor import DataProcessor
from typing import Tuple

from src.Utils.Config import Config
from src.Utils.Utils import save_to_csv

import logging
import time

# * Main usage
def ABC_main_LL() -> None:
        
    # * Bees
    List_bees = [30, 60, 90];

    for _, Num_bees in enumerate(List_bees):
        for _, (Name, CSV) in enumerate(Config.Euler_csv_files.items()):
            
            Meta_name = "ABC_LL";

            # * Bees
            #Num_bees = 90;

            # * Maximum number of iterations
            Max_iter = 10;

            # * Number of hyperparameters to optimize (alpha and Max_iter)
            DIM = 2;

            # * Lower bounds for the hyperparameters
            LB = np.array([1e-8, 100]);

            # * Upper bounds for the hyperparameters
            UB = np.array([1.0, 1000]);

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
                
                Alpha, Max_iter = Hyperparams;

                print(F"Objective function (Alpha : {Alpha}, Max_iter : {Max_iter})\n");

                Model = LassoLars(alpha=Alpha, max_iter=int(Max_iter));

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

            def ABC(Num_bees: int, Max_iter: int, DIM: int, LB: np.ndarray, UB: np.ndarray) -> Tuple[np.ndarray, float, pd.DataFrame]:
                """
                Artificial Bee Colony (ABC) main loop.

                Parameters
                ----------
                Num_bees : int
                    Number of bees (individuals) in the population.
                Max_iter : int
                    Maximum number of iterations.
                DIM : int
                    Number of dimensions (hyperparameters) for each individual.
                LB : np.ndarray
                    Lower bounds for the hyperparameters.
                UB : np.ndarray
                    Upper bounds for the hyperparameters.

                Returns
                -------
                Tuple[np.ndarray, float, pd.DataFrame]
                    Best individual (hyperparameters), best score (MSE), and DataFrame of best individuals and their scores over iterations.
                """
                
                # * Initialize the population with random values within the bounds
                Population = initialize_population(Num_bees, DIM, LB, UB);
                
                # * Evaluate the initial population
                Fitness = evaluate_population(Population);
                
                # * Find the index of the best individual in the initial population
                Best_idx = np.argmin(Fitness);
                
                # * Set the best individual and its fitness score
                Best_individual = Population[Best_idx];
                Best_score = Fitness[Best_idx];
                
                # * Lists to keep track of the best individuals and their scores over iterations
                Best_individuals = [];
                Best_scores = [];

                # * Initialize best_iteration counter
                Best_iteration = 0;

                for j in range(Max_iter):
                    for i in range(Num_bees):
                        # * Generate a random phi value for the update equation
                        phi = np.random.uniform(-1, 1, size = DIM);
                        
                        # * Randomly select another individual (bee) from the population
                        k = np.random.randint(0, Num_bees);
                        while(k == i):
                            k = np.random.randint(0, Num_bees);
                        
                        # * Create a new solution by modifying the current solution with the selected bee's solution
                        New_solution = Population[i] + phi * (Population[i] - Population[k])
                        
                        # * Clip the new solution to be within the bounds
                        New_solution = np.clip(New_solution, LB, UB)
                        
                        # * Evaluate the new solution
                        Score_trial = objective_function(New_solution)
                        
                        # * If the new solution is better, update the population and fitness
                        if(Score_trial < Fitness[i]):
                            Population[i] = New_solution;
                            Fitness[i] = Score_trial;
                            
                            # * If the new solution is the best found so far, update the best individual and score
                            if(Score_trial < Best_score):
                                Best_individual = New_solution;
                                Best_score = Score_trial;
                                Best_iteration = j + 1;
                        
                        # * Append the new solution and its score to the tracking lists
                        Best_individuals.append(list(New_solution))
                        Best_scores.append(Score_trial)
                    
                    # * Print the current iteration and the best score found so far
                    print(f"Iteration {j + 1}/{Max_iter}, Best Score: {Best_score}\n")
                
                # * Create a DataFrame to store the best individuals and their scores over iterations
                Dataframe = pd.DataFrame({
                    'Alpha': [ind[0] for ind in Best_individuals],
                    'Max_iter': [ind[1] for ind in Best_individuals],
                    'MSE': Best_scores
                })
                
                # * Return the best individual, best score, and the DataFrame
                return Best_individual, Best_score, Best_iteration, Dataframe
            
            Start_time = time.time();
            Best_individual, Best_score, Best_iteration, Dataframe_data = ABC(Num_bees, Max_iter, DIM, LB, UB);
            End_time = time.time();

            Execution_time = End_time - Start_time;

            print(f"Optimal Hyperparameters: {Best_individual}");
            print(f"Optimal MSE: {Best_score}");

            # * Create the LassoLars model with the optimized hyperparameters
            Alpha, Max_iter = Best_individual;
            Model = LassoLars(alpha=Alpha, max_iter = int(Max_iter));
            Model.fit(X_train, y_train);
            y_pred = Model.predict(X_test);
            Final_mse = mean_squared_error(y_test, y_pred);
            print(f"Final MSE with optimal hyperparameters: {Final_mse}");

            Coefficients = Model.coef_;
            Coefficients_rounded = [round(coef, 3) if coef != 0.0 else 0.0 for coef in Coefficients];
            Non_zero_indices = np.nonzero(Coefficients_rounded)[0];
            Non_zero_count = len(Non_zero_indices);

            print(f"Theta found by stomp: {Coefficients_rounded}");
            print(f"Non-zero indices: {Non_zero_indices}");
            print(f"Non-zero count: {Non_zero_count}");

            Dataframe_idx = pd.DataFrame({
                'Coefficient': [Coefficients_rounded],
                'Non-zero indices': [Non_zero_indices],
                'Non-zero count': Non_zero_count
            });

            File_path_meta = os.path.join(Config.Meta_path, Meta_name);

            # * Check if directory exists
            if not os.path.exists(File_path_meta):
                # * If directory doesn't exist, create it
                os.makedirs(File_path_meta);
            
            File_path_unique = f"{Meta_name}_{Num_bees}_Coefficients_Info_{Name}.csv";
            CSV_file_path = os.path.join(File_path_meta, File_path_unique);
            Dataframe_idx.to_csv(CSV_file_path, index=False);

            # * Save evolution to CSV
            CSV_file_data = os.path.join(File_path_meta, f"{Meta_name}_{Num_bees}_{Name}.csv");
            save_to_csv(CSV_file_data, Dataframe_data);

            Log_file = os.path.join(File_path_meta, f"{Meta_name}_log.txt");

            # * Configure logging to write to a file
            logging.basicConfig(filename = Log_file, level=logging.INFO);

            # * Log best individual and score
            logging.info(f"Dataset: {Name}");
            logging.info(f"Name: {Meta_name}_{Num_bees}_log.txt");
            logging.info(f"Optimal Hyperparameters: {Best_individual}");
            logging.info(f"Optimal MSE: {Best_score}");
            logging.info(f"Found on iteration: {Best_iteration}");
            logging.info(f"Time: {Execution_time}\n");

'''# * Main usageif __name__ == "__main__":
# * ? Main usage:
if __name__ == "__main__":
    ABC_main_LL();'''
