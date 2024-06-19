import pandas as pd
import logging

from typing import List, Optional

class DataProcessor:
    """
    A class for reading CSV data files and extracting features and labels.

    Parameters
    ----------
    File_path : str
        The path to the CSV data file to be processed.

    Attributes
    ----------
    __File_path : str
        The path to the CSV data file to be processed.

    Examples
    --------
    Example 1:
    ```
    # Initialize the DataProcessor with a file path and extract data
    file_path = "your_csv_file.csv"  # Replace with your CSV file path
    data_processor = DataProcessor(file_path)
    try:
        data_processor.read_csv_and_extract_features_labels()
    except Exception as e:
        logging.error(f"An error occurred while processing the CSV file: {str(e)}")
    
    # Access X and y
    X = data_processor.X
    y = data_processor.y
    ```

    Example 2:
    ```
    # Another way to use the DataProcessor class
    data_processor = DataProcessor("another_csv_file.csv")
    try:
        data_processor.read_csv_and_extract_features_labels()
    except Exception as e:
        logging.error(f"An error occurred while processing the CSV file: {str(e)}")
    
    # Access X and y
    X = data_processor.X
    y = data_processor.y
    ```

    Methods
    -------
    read_csv_and_extract_features_labels()
        Read the CSV data file specified in the 'file_path' attribute and extract
        features and labels.

        Returns
        -------
        X : numpy.ndarray
            A NumPy array containing the extracted features.
        
        y : numpy.ndarray
            A NumPy array containing the extracted labels.
    """

    def __init__(self, File_path) -> None:
        self.__File_path = File_path;
        self.X = None;
        self.y = None;

    # * Class description
    def __str__(self) -> str:
        """
        Return a string description of the object.

        Returns:
            str: String description of the object.
        """
        return  f'''A class {self.__class__.__name__} for reading CSV data files and extracting features and labels.''';

    # * Deleting (Calling destructor)
    def __del__(self) -> None:
        """
        Destructor called when the object is deleted.
        """
        print(f'Destructor called, {self.__class__.__name__} class destroyed.');
    
    @property
    def Get_file_path(self) -> str:
        """Getter method for the `Text` property."""
        return self.__File_path;

    @property
    def Get_file_path_dataframe(self) -> str:
        
        try:
            # * Read the CSV file
            Data = pd.read_csv(self.__File_path);

        except Exception as e:
            logging.error(f"An error occurred while processing the CSV file: {str(e)}");

        return Data;
    
    def read_csv_and_extract_features_labels(self, Columns_to_drop : Optional[List[str]] = None, Column_Y : Optional[str] = None):
        """
        Read the CSV data file specified in the 'file_path' attribute and extract
        features and labels.

        Returns
        -------
        X : numpy.ndarray
            A NumPy array containing the extracted features.
        
        y : numpy.ndarray
            A NumPy array containing the extracted labels.
        """
        try:
            if isinstance(self.__File_path, str):
                # * If File_path is a string, read the CSV file
                Data = pd.read_csv(self.__File_path)
            elif isinstance(self.__File_path, pd.DataFrame):
                # * If File_path is already a DataFrame, use it directly
                Data = self.__File_path;
            else:
                raise ValueError("File_path must be either a string (file path) or a pandas DataFrame.");
            
            # * Check if Columns_to_drop is not None and Column_Y is not None
            if Columns_to_drop is not None and Column_Y is not None:

                # * Drop specified columns
                Data_X = Data[Columns_to_drop];
                # * Extract features (X) and labels (y)
                self.X = Data_X.iloc[:, :].values;
                self.y = Data[Column_Y].values;

                #print(f" Columns : {Data_X}");

                print(f" Len X : {len(self.X[0])}");
                print(f" Len y : {len(self.y)}");

                # * Extract column headers
                Columns = Data_X.columns.tolist();

                return self.X, self.y, Columns
        
            else:

                # * Extract features (X) from all columns except the last one
                self.X = Data.iloc[:, 1:-1].values;

                # * Extract y from the last column
                self.y = Data.iloc[:, -1].values;

                print(f" Len X Else : {len(self.X[0])}");
                print(f" Len y Else : {len(self.y)}");
            
            return self.X, self.y
            
        except Exception as e:
            logging.error(f"An error occurred while processing the CSV file: {str(e)}");
