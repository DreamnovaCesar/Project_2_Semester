import os
import numpy as np
import pandas as pd
import multiprocessing

from src.Utils.Config import Config
from src.Utils.DataLoader import DataLoader
from src.Utils.Utils import count_combinations
from src.Utils.Utils import calculate_enclosing_surface
from src.Utils.Utils import calculate_contact_surface
from src.Utils.Utils import calculate_volume

from src.Decorator.TimerCMD import Timer
from src.Decorator.multiprocessing_info import multiprocessing_info

from typing import Optional, Union, List, Tuple

from src.Test_DM import DM
from src.Utils.ByteStorage import ByteStorage

class Test_OEMM_misc(DM):
    """
    Q_values denote the spatial extent of each identified pattern within the 3D image.
    Octo-Voxel Tetra Multi Manager (OCMM) - Asynchronous approaches and multiprocessing are employed in a class created
    specifically to handle Octo-Voxels, resulting in significant improvements in speed in the Tetra Descriptor Extractor.

    Attributes:
    -----------
    Path : str
        The file path for data and convert it into an array.
    Basepath : str
        The base name of the file path.
    Folder : bool
        Indicates whether the provided path is a directory.
    Num_processes : Optional[int]
        Number of parallel processes (default: half of available CPU cores).
    SLC : ByteStorage
        Instance of ByteStorage for handling binary storage lists.
    STORAGELIST : np.ndarray
        Numpy array representation of the binary storage list.

    Methods:
    --------
    save_to_csv(Combinations_data: Union[np.ndarray, list], Tetra_data: Union[np.ndarray, list]) -> None:
        Save the combinations and Tetra values to a CSV file.

    process_octovoxel(CHUNK: np.ndarray, STORAGELIST: np.ndarray) -> Tuple[np.ndarray, int]:
        Process a sub-volume of octovoxel and return combinations and Tetra value.

    get_array(Depth: int, Height: int, Width: int) -> np.ndarray:
        Calculate Combinations_int based on the given Storage_list using multiprocessing.
    """
    def __init__(self,
                 path: str,
                 num_processes: Optional[int] = (multiprocessing.cpu_count() // 2)) -> None:
        """
        Initialize OCMM.

        Parameters:
        ----------
        Path : str
            The file path for data and convert it into an array.
        Num_processes : Optional[int]
            Number of parallel processes (default: half of available CPU cores).
        """
        super().__init__(path, num_processes)
        self.Descriptor = "MISC_Async";
        
        self.SLC1 = ByteStorage;
        self.STORAGELIST_OCTO = self.SLC1.to_numpy_array();
    
    # * Class description
    def __str__(self) -> str:
        """
        Return a string description of the object.

        Returns:
        ----------
        None
        """
        return f'''{self.__class__.__name__}: A class for handling Octo-Voxels and obtaining q_values based on the given Storage_list.''';

    # * Deleting (Calling destructor)
    def __del__(self) -> None:
        """
        Destructor called when the object is deleted.

        Returns:
        ----------
        None
        """
        print(f'Destructor called, {self.__class__.__name__} class destroyed.');

    #@Timer.timer()
    def process_octovoxel(self, 
                          File : str, 
                          STORAGELIST : List[np.ndarray], 
                          Depth : int, 
                          Height : int, 
                          Width : int) -> Tuple[np.ndarray, int]:
        """
        Calculate Combinations_int based on the given Storage_list and return them as a numpy array.

        Parameters:
        -----------
        File : str
            The name of the file being processed.
        STORAGELIST : List[np.ndarray]
            List of numpy array representations of the binary storage list.
        Depth : int
            The depth dimension of the 3D array.
        Height : int
            The height dimension of the 3D array.
        Width : int
            The width dimension of the 3D array.

        Returns:
        -------
        Tuple[np.ndarray, int]
            A tuple containing an array of Combinations_int and Tetra value.

        Notes:
        ------
        - Octovoxel size is set to 2.
        - Tetra value is the sum of Combinations_int.

        """
        Misc = [];

        File_path = os.path.join(self.Path, File);
        Arrays = DataLoader.load_data(File_path);

        print(f"File path... {File_path}");

        # * Reshape the array to a 3D array based on the calculated height.
        Arrays = Arrays.reshape(Depth, Height, Width);
        
        # * Create sliding windows for CHUNK for faster access to sub-volumes
        view_CHUNK_2x2x1 = np.lib.stride_tricks.sliding_window_view(Arrays, (Config.Octovoxel_size, Config.Octovoxel_size, 1));
        view_CHUNK_2x1x2 = np.lib.stride_tricks.sliding_window_view(Arrays, (Config.Octovoxel_size, 1, Config.Octovoxel_size));
        view_CHUNK_1x2x2 = np.lib.stride_tricks.sliding_window_view(Arrays, (1, Config.Octovoxel_size, Config.Octovoxel_size));

        # * Create sliding windows for CHUNK for faster access to sub-volumes
        VIEW_CHUNK = np.lib.stride_tricks.sliding_window_view(Arrays, (Config.Octovoxel_size, Config.Octovoxel_size, Config.Octovoxel_size));

        # * Resize sliding windows to 1x2x2
        view_CHUNK_2x2x1_resized = view_CHUNK_2x2x1.reshape(-1, 1, Config.Octovoxel_size, Config.Octovoxel_size);
        view_CHUNK_2x1x2_resized = view_CHUNK_2x1x2.reshape(-1, 1, Config.Octovoxel_size, Config.Octovoxel_size);

        # * Convert STORAGELIST to a single numpy array for vectorized comparisons
        STORAGELIST_array = np.array(STORAGELIST);
        self.STORAGELIST_OCTO_array = np.array(self.STORAGELIST_OCTO);
        
        # * Count combinations for each sliding window
        Combinations_int_1 = count_combinations([[view_CHUNK_2x2x1_resized]], STORAGELIST_array);
        Combinations_int_2 = count_combinations([[view_CHUNK_2x1x2_resized]], STORAGELIST_array);
        Combinations_int_3 = count_combinations(view_CHUNK_1x2x2, STORAGELIST_array);
        Combinations_octovoxel = count_combinations(VIEW_CHUNK, self.STORAGELIST_OCTO_array);

        Concatenated_combinations = Combinations_int_1 + Combinations_int_2 + Combinations_int_3;

        TETRAVOXEL = Concatenated_combinations[-1];
        OCTOVOXEL = Combinations_octovoxel[-1];

        Enclosing_surface = calculate_enclosing_surface(Arrays);
        Contact_surface = calculate_contact_surface(Arrays);
        Volume = calculate_volume(Arrays);

        N2 = (Enclosing_surface + Contact_surface);
        N1 = (TETRAVOXEL + (2 * Enclosing_surface));
        N0 = ((2 * N1) - (4 * Volume) - (2 * Enclosing_surface) - OCTOVOXEL);

        False_Euler = (N1 - (((3 * Enclosing_surface) / 2)) - (2 * Volume) - OCTOVOXEL);

        # * Return the calculated Combinations_int as a numpy array.
        Combinations = (Combinations_octovoxel * Config._OUTPUT_3D_);
        Euler = np.sum(Combinations);

        print(f"Processing file... {File_path} -------- {False_Euler} ------ {Euler}");

        Misc.append(N0);
        Misc.append(N1);
        Misc.append(N2);
        Misc.append(Volume);
        Misc.append(Enclosing_surface);
        Misc.append(Contact_surface);
        Misc.append(TETRAVOXEL);
        Misc.append(OCTOVOXEL);

        #print(Concatenated_combinations);
        # * Concatenate the three Combinations_int arrays along axis 0, excluding the first value from each
        #Concatenated_combinations = np.concatenate([Combinations_int_1[1:], Combinations_int_2[1:], Combinations_int_3[1:]], axis=0)
        
        Misc = np.array(Misc);

        return Misc, False_Euler
    
    @Timer.timer("Execution_OCMM.log")
    @multiprocessing_info    
    def get_array(self, Depth: int, Height: int, Width: int) -> None:
        """
        Perform multiprocessing to calculate Combinations_int based on the given self.STORAGELIST.

        Parameters:
        -----------
        Depth : int
            The depth dimension of the 3D array.
        Height : int
            The height dimension of the 3D array.
        Width : int
            The width dimension of the 3D array.

        Returns:
        -------
        Tuple[List[np.ndarray], List[int], str]
            A tuple containing arrays of Combinations_int, Tetra values, and the descriptor obtained by processing multiple files in parallel.

        Notes:
        ------
        - If the `Path` attribute indicates a directory, the method processes all .txt files in that directory.
        - Utilizes multiprocessing to parallelize the calculation for each file.
        - Combinations_int and Tetra values are accumulated across all processed files.
        - The results are saved to a CSV file using the `save_to_csv_Tetra` method.
        - The final result is the sum of Combinations_int across all files.

        """
        
        if self.Folder:
            Combinations_all = [];
            Tetra_all = [];

            # * Filter the list to include only .txt files
            Files = os.listdir(self.Path);
            print(f"Processing files... Total: {len(Files)}");
    
            with multiprocessing.Pool(processes=self.Num_processes) as pool:
                # * Use starmap to pass both file paths and STORAGELIST to process_octovoxel
                Results = pool.starmap_async(self.process_octovoxel, [(File, self.STORAGELIST, Depth, Height, Width) for File in Files]);

                Data = Results.get();

                # * Extract Results and Tetra from the list of results
                Combinations, Euler = zip(*Data);
        
                '''Result_combination = np.sum(Combination, axis=0);
                Result_Tetra = np.sum(Tetra, axis=0);'''
                
                for _, (C, E) in enumerate(zip(Combinations, Euler)):

                    Combinations_all.append(C);
                    Tetra_all.append(E);

            return Combinations_all, Tetra_all, self.Descriptor 
        else:
            return None



