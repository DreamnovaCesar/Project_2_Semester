import pandas as pd

def save_to_csv(Path: str, Dataframe_evolution: pd.DataFrame) -> None:
    """
    Save the evolution of individuals and their scores to a CSV file.

    Parameters
    ----------
    Path : str
        The name of the CSV file.
    Dataframe_evolution : pd.DataFrame
        DataFrame containing the evolution of individuals and their scores.
    """

    Dataframe_evolution.to_csv(f"{Path}", index=False);