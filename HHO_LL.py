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
def HHO_main_LL() -> None:

    # * Hawks
    List_hawks = [30, 60, 90];

    for _, Num_hawks in enumerate(List_hawks):    
        for _, (Name, CSV) in enumerate(Config.Euler_csv_files.items()):
            
            Meta_name = "HHO_LL";

            # * Hawks
            #Num_hawks = 90;

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

            # * HHO main loop
            def HHO(Num_hawks: int, Max_iter: int, DIM: int, LB: np.ndarray, UB: np.ndarray) -> Tuple[np.ndarray, float, pd.DataFrame]:
                """
                Perform Harris Hawks Optimization (HHO) to optimize hyperparameters.

                Parameters
                ----------
                Num_hawks : int
                    The number of hawks in the population.
                Max_iter : int
                    The maximum number of iterations.
                DIM : int
                    The dimensionality of the search space.
                LB : np.ndarray
                    The lower bounds of the search space.
                UB : np.ndarray
                    The upper bounds of the search space.

                Returns
                -------
                Tuple[np.ndarray, float, pd.DataFrame]
                    The best found hyperparameters, the corresponding fitness value, and a dataframe recording the optimization process.
                """

                # * Initialize the population and evaluate their fitness
                Population = initialize_population(Num_hawks, DIM, LB, UB);
                Fitness = evaluate_population(Population);
                Best_idx = np.argmin(Fitness);

                # * Initialize the best individual and its fitness
                Best_individual = Population[Best_idx];
                Best_score = Fitness[Best_idx];

                # * Lists to store the best individuals and their scores for each iteration
                Best_individuals = [];
                Best_scores = [];

                # * Initialize best_iteration counter
                Best_iteration = 0;

                for j in range(Max_iter):
                    for i in range(Num_hawks):
                        # * Calculate energy factors
                        E1 = (2 * (1 - j / Max_iter));
                        E0 = (2 * np.random.rand() - 1);
                        E = (E1 * E0);

                        if abs(E) >= 1:
                            q = np.random.rand();
                            rand_idx = np.random.randint(0, Num_hawks);
                            X_rand = Population[rand_idx];

                            if(q >= 0.5):
                                # * Exploration phase
                                Population[i] = (X_rand - np.random.rand() * abs(X_rand - 2 * np.random.rand() * Population[i]));
                            else:
                                # * Exploration phase
                                Population[i] = (Best_individual - np.mean(Population, axis=0)) - np.random.rand() * (LB + np.random.rand() * (UB - LB));

                        else:
                            r = np.random.rand();
                            if(r >= 0.5 and abs(E) < 0.5):
                                # * Exploitation phase
                                Population[i] = (Best_individual - E * abs(Best_individual - Population[i]));
                            elif(r >= 0.5 and abs(E) >= 0.5):
                                # * Exploitation phase
                                Population[i] = (Best_individual - E * abs(Best_individual - Population[i]) / (E + np.finfo(float).eps));
                            elif(r < 0.5 and abs(E) < 0.5):
                                # * Exploitation phase
                                X_m = (Best_individual - E * abs(Best_individual - np.mean(Population, axis=0)));
                                Population[i] = (X_m - np.random.rand() * abs(X_m - Population[i]));
                            else:
                                # * Exploitation phase
                                X_m = (Best_individual - E * abs(Best_individual - np.mean(Population, axis=0)));
                                Population[i] = (X_m - np.random.rand() * abs(X_m - Population[i]) / (E + np.finfo(float).eps));

                        Population[i] = np.clip(Population[i], LB, UB)
                        Score_trial = objective_function(Population[i]);

                        if(Score_trial < Fitness[i]):
                            Fitness[i] = Score_trial;

                        if(Score_trial < Best_score):
                            Best_individual = Population[i];
                            Best_score = Score_trial;
                            Best_iteration = j + 1;
                        
                        Best_individuals.append(list(Population[i]))
                        Best_scores.append(Score_trial)

                    print(f"Iteration {j + 1}/{Max_iter}, Best Score: {Best_score}");

                Dataframe = pd.DataFrame({
                    'Alpha': [ind[0] for ind in Best_individuals],
                    'Max_iter': [ind[1] for ind in Best_individuals],
                    'MSE': Best_scores
                });

                return Best_individual, Best_score, Best_iteration, Dataframe
            
            Start_time = time.time();
            Best_individual, Best_score, Best_iteration, Dataframe_data = HHO(Num_hawks, Max_iter, DIM, LB, UB);
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
            
            File_path_unique = f"{Meta_name}_{Num_hawks}_Coefficients_Info_{Name}.csv";
            CSV_file_path = os.path.join(File_path_meta, File_path_unique);
            Dataframe_idx.to_csv(CSV_file_path, index=False);

            # * Save evolution to CSV
            CSV_file_data = os.path.join(File_path_meta, f"{Meta_name}_{Num_hawks}_{Name}.csv");
            save_to_csv(CSV_file_data, Dataframe_data);

            Log_file = os.path.join(File_path_meta, f"{Meta_name}_{Num_hawks}_log.txt");

            # * Configure logging to write to a file
            logging.basicConfig(filename = Log_file, level=logging.INFO);

            # * Log best individual and score
            logging.info(f"Dataset: {Name}");
            logging.info(f"Name: {Meta_name}_{Num_hawks}_log.txt");
            logging.info(f"Optimal Hyperparameters: {Best_individual}");
            logging.info(f"Optimal MSE: {Best_score}");
            logging.info(f"Found on iteration: {Best_iteration}");
            logging.info(f"Time: {Execution_time}\n");

'''# * Main usage
if __name__ == "__main__":
   HHO_main_LL();'''