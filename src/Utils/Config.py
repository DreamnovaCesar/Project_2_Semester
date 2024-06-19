import numpy as np
#import tensorflow as tf

class Config:
    """
    A class that defines data-related constants and configuration options for regression models.
    """

    Meta_path = r"src\Data\Metaheuristic";

    Euler_csv_files = {
    'DS_0_3'    : r'src/Data/Combinations/Folder_Combinations_And_Euler_Async_Images_DS_0_3_3D.csv', 
    'DS_0_4'    : r'src/Data/Combinations/Folder_Combinations_And_Euler_Async_Images_DS_0_4_3D.csv', 
    'DS_0_8'    : r'src/Data/Combinations/Folder_Combinations_And_Euler_Async_Images_DS_0_8_3D.csv', 
    'DS_0_16'   : r'src/Data/Combinations/Folder_Combinations_And_Euler_Async_Images_DS_0_16_3D.csv', 
    'DS_0_32'   : r'src/Data/Combinations/Folder_Combinations_And_Euler_Async_Images_DS_0_32_3D.csv', 
    'DS_0_64'   : r'src/Data/Combinations/Folder_Combinations_And_Euler_Async_Images_DS_0_64_3D.csv', 
    #'DS_1_3'    : r'src/Data/Combinations/Folder_Combinations_And_Euler_Async_Images_DS_1_3_3D.csv', 
    #'DS_1_4'    : r'src/Data/Combinations/Folder_Combinations_And_Euler_Async_Images_DS_1_4_3D.csv', 
    #'DS_1_8'    : r'src/Data/Combinations/Folder_Combinations_And_Euler_Async_Images_DS_1_8_3D.csv',
    #'DS_1_16'   : r'src/Data/Combinations/Folder_Combinations_And_Euler_Async_Images_DS_1_16_3D.csv', 
    #'DS_1_32'   : r'src/Data/Combinations/Folder_Combinations_And_Euler_Async_Images_DS_1_32_3D.csv', 
    #'DS_1_64'   : r'src/Data/Combinations/Folder_Combinations_And_Euler_Async_Images_DS_1_64_3D.csv',
    #'DS_2_64'   : r'src/Data/Combinations/Folder_Combinations_And_Euler_Async_Images_DS_2_64_3D.csv',
    #'DS_3_64'   : r'src/Data/Combinations/Folder_Combinations_And_Euler_Async_Images_DS_3_64_3D.csv',
    #'DS_4_64'   : r'src/Data/Combinations/Folder_Combinations_And_Euler_Async_Images_DS_4_64_3D.csv',
    #'DS_5_64'   : r'src/Data/Combinations/Folder_Combinations_And_Euler_Async_Images_DS_5_64_3D.csv',
    #'DS_6_64'   : r'src/Data/Combinations/Folder_Combinations_And_Euler_Async_Images_DS_6_64_3D.csv'
    };