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
def PSO_main_LL() -> None:

    # * Particles
    List_particles = [30, 60, 90];

    for _, Num_particles in enumerate(List_particles):
        for _, (Name, CSV) in enumerate(Config.Euler_csv_files.items()):
            
            Meta_name = "PSO_LL";

            # * Particles
            #Num_particles = 30;

            # * Maximum number of iterations
            Max_iter = 10;

            # * Number of hyperparameters to optimize (alpha and max_iter)
            DIM = 2;

            # * Lower bounds for the hyperparameters
            LB = np.array([1e-8, 100]);

            # * Upper bounds for the hyperparameters
            UB = np.array([1.0, 1000]);
            
            # * Inertia weight
            W = 0.729;

            # * Cognitive (particle) weight
            C1 = 1.494;

            # * Social (swarm) weight
            C2 = 1.494;

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

            # * PSO main loop
            def PSO(Num_particles : int, Max_iter : int, DIM : int, LB : np.ndarray, UB : np.ndarray, W : float, C1 : float, C2 : float) -> Tuple[np.ndarray, float, pd.DataFrame]:
                """
                Perform Particle Swarm Optimization (PSO) to optimize hyperparameters.

                Parameters
                ----------
                num_particles : int
                    The number of particles in the swarm.
                max_iter : int
                    The maximum number of iterations.
                dim : int
                    The dimensionality of the search space.
                lb : np.ndarray
                    The lower bounds of the search space.
                ub : np.ndarray
                    The upper bounds of the search space.
                w : float
                    The inertia weight.
                c1 : float
                    The cognitive coefficient.
                c2 : float
                    The social coefficient.

                Returns
                -------
                Tuple[np.ndarray, float, pd.DataFrame]
                    The best found hyperparameters, the corresponding fitness value, and a dataframe recording the optimization process.
                """

                # * Initialize the population and velocity
                Population = initialize_population(Num_particles, DIM, LB, UB);
                Velocity = np.zeros((Num_particles, DIM));
                Fitness = np.array([objective_function(ind) for ind in Population])

                # * Initialize personal best (pbest) and global best (gbest)
                PBest = Population.copy();
                PBest_fitness = Fitness.copy();

                GBest_idx = np.argmin(Fitness);
                GBest = Population[GBest_idx];
                GBest_fitness = Fitness[GBest_idx];

                # * Lists to store the best individuals and their scores for each iteration
                Best_individuals = [];
                Best_scores = [];

                # * Initialize best_iteration counter
                Best_iteration = 0;

                for j in range(Max_iter):
                    for i in range(Num_particles):

                        # * Generate random coefficients
                        R1 = np.random.rand(DIM);
                        R2 = np.random.rand(DIM);

                        # * Update velocity
                        Velocity[i] = (W * Velocity[i] + 
                                    C1 * R1 * (PBest[i] - Population[i]) + 
                                    C2 * R2 * (GBest - Population[i]));

                        # * Update position
                        Population[i] = Population[i] + Velocity[i];
                        Population[i] = np.clip(Population[i], LB, UB);

                        # * Evaluate fitness
                        Fitness[i] = objective_function(Population[i]);

                        # * Update personal best
                        if(Fitness[i] < PBest_fitness[i]):
                            PBest[i] = Population[i];
                            PBest_fitness[i] = Fitness[i];
                        
                        # * Update global best
                        if Fitness[i] < GBest_fitness:
                            GBest = Population[i]
                            GBest_fitness = Fitness[i]
                            Best_iteration = j + 1
                    
                    print(f"Iteration {j + 1}/{Max_iter}, Best Score: {GBest_fitness}");

                    # Record the global best individual and its score at the end of each iteration
                    Best_individuals.append(GBest.copy())
                    Best_scores.append(GBest_fitness)

                    print(f"Iteration {j + 1}/{Max_iter}, Best Score: {GBest_fitness}")

                Dataframe = pd.DataFrame({
                    'Alpha': [ind[0] for ind in Best_individuals],
                    'Max_iter': [ind[1] for ind in Best_individuals],
                    'MSE': Best_scores
                });

                return GBest, GBest_fitness, Best_iteration, Dataframe
            
            Start_time = time.time();
            GBest, GBest_fitness, Best_iteration, Dataframe_data = PSO(Num_particles, Max_iter, DIM, LB, UB, W, C1, C2);
            End_time = time.time();
            
            Execution_time = End_time - Start_time;

            print(f"Optimal Hyperparameters: {GBest}");
            print(f"Optimal MSE: {GBest_fitness}");

            # * Create the LassoLars model with the optimized hyperparameters
            Alpha, Max_iter = GBest;
            Model = LassoLars(alpha = Alpha, max_iter = int(Max_iter));
            Model.fit(X_train, y_train)
            y_pred = Model.predict(X_test)
            Final_mse = mean_squared_error(y_test, y_pred)
            print(f"Final MSE with optimal hyperparameters: {Final_mse}")

            Coefficients = Model.coef_
            Coefficients_rounded = [round(coef, 3) if coef != 0.0 else 0.0 for coef in Coefficients]
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
            
            File_path_unique = f"{Meta_name}_{Num_particles}_Coefficients_Info_{Name}.csv";
            CSV_file_path = os.path.join(File_path_meta, File_path_unique);
            Dataframe_idx.to_csv(CSV_file_path, index=False);

            # * Save evolution to CSV
            CSV_file_data = os.path.join(File_path_meta, f"{Meta_name}_{Num_particles}_{Name}.csv");
            save_to_csv(CSV_file_data, Dataframe_data);

            Log_file = os.path.join(File_path_meta, f"{Meta_name}_{Num_particles}_log.txt");

            # * Configure logging to write to a file
            logging.basicConfig(filename = Log_file, level=logging.INFO);

            # * Log best individual and score
            logging.info(f"Dataset: {Name}");
            logging.info(f"Name: {Meta_name}_{Num_particles}_log.txt");
            logging.info(f"Optimal Hyperparameters: {GBest}");
            logging.info(f"Optimal MSE: {GBest_fitness}");
            logging.info(f"Found on iteration: {Best_iteration}");
            logging.info(f"Time: {Execution_time}\n");

'''# * Main usage
if __name__ == "__main__":
    PSO_main_LL();'''