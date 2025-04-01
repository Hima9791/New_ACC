import os
import pandas as pd
from config import MAPPING_FILE_LOCAL, MULTIPLIER_MAPPING

def save_mapping_to_disk(df: pd.DataFrame, filename: str = MAPPING_FILE_LOCAL):
    """
    Saves the provided mapping DataFrame to an Excel file.
    
    Parameters:
        df (pd.DataFrame): The mapping DataFrame.
        filename (str): The file name to save the mapping to.
    """
    df.to_excel(filename, index=False, engine='openpyxl')


def read_mapping_file(mapping_file_path: str = MAPPING_FILE_LOCAL):
    """
    Reads the mapping file from disk and validates its contents.
    
    Checks:
      - The file exists.
      - The required columns {'Base Unit Symbol', 'Multiplier Symbol'} exist.
      - All multiplier symbols in the file are defined in MULTIPLIER_MAPPING.
      
    Returns:
        tuple: (base_units, multipliers_dict)
            base_units (set): Unique base unit symbols from the file.
            multipliers_dict (dict): The multiplier mapping (from config).
    
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If required columns are missing or if undefined multipliers are found.
    """
    if not os.path.exists(mapping_file_path):
        raise FileNotFoundError(f"Error: '{mapping_file_path}' not found.")
    
    df = pd.read_excel(mapping_file_path)
    required_cols = {"Base Unit Symbol", "Multiplier Symbol"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"'{mapping_file_path}' must have columns: {required_cols}")
    
    # Extract unique base units from the file.
    base_units = set(str(x).strip() for x in df["Base Unit Symbol"].dropna().unique())
    
    # Validate multipliers.
    file_multipliers = set(str(x).strip() for x in df["Multiplier Symbol"].dropna().unique())
    undefined = file_multipliers - set(MULTIPLIER_MAPPING.keys())
    if undefined:
        raise ValueError(f"Undefined multipliers in '{mapping_file_path}': {undefined}")
    
    return base_units, MULTIPLIER_MAPPING
