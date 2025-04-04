#**`detailed_pipeline.py`**

#```python
#############################################
# MODULE: DETAILED ANALYSIS PIPELINE
# Purpose: Implements the second pipeline logic:
#          - Performs detailed classification and analysis.
#          - Extracts units, numeric values, normalized values.
#          - Generates summary columns (consistency, min/max, etc.).
#############################################

import pandas as pd
import re
import traceback
import streamlit as st # For error/warning/debug messages

# Import necessary functions and constants from other modules
from mapping_utils import read_mapping_file
from analysis_helpers import (
    classify_value_type_detailed,
    replace_numbers_keep_sign_all,
    resolve_compound_unit,
    analyze_value_units,
    extract_numeric_info_for_value,
    safe_str # Keep safe_str if used directly here, or ensure it's used within called helpers
)


# --- detailed_analysis_pipeline function (core logic) ---
def detailed_analysis_pipeline(df, base_units, multipliers_dict):
    """
    Performs detailed analysis on each row of the input DataFrame.
    Adds columns for classification, normalization, unit analysis, numeric extraction, etc.

    Args:
        df (pd.DataFrame): Input DataFrame (must contain 'Value' column).
        base_units (set): Set of known base units.
        multipliers_dict (dict): Dictionary mapping multiplier symbols to factors.

    Returns:
        pd.DataFrame: DataFrame with original data and added analysis columns.
                      Returns an empty DataFrame on critical errors.
    """
    st.write("DEBUG: Starting Detailed Analysis Pipeline...")
    results = [] # List to store row-by-row results as dicts

    # --- Input Validation ---
    if not isinstance(df, pd.DataFrame):
        st.error("Detailed Analysis Error: Input must be a pandas DataFrame.")
        return pd.DataFrame()
    if 'Value' not in df.columns:
        st.error("Detailed Analysis Error: Input DataFrame missing 'Value' column.")
        return pd.DataFrame()

    # Ensure 'Value' column is string type for consistent processing
    df_copy = df.copy() # Work on a copy to avoid modifying original DataFrame passed in
    df_copy['Value'] = df_copy['Value'].astype(str)


    # --- Process each row ---
    total_rows = len(df_copy)
    for index, row in df_copy.iterrows():
        val_str = row.get('Value', '').strip()

        # Initialize result dict for this row, include original index if needed later
        row_results = {"OriginalIndex": index}
        # Copy original non-'Value' columns to the result dict first
        for col in df_copy.columns:
            if col != 'Value': # Exclude 'Value' itself initially, add it from val_str
                 row_results[col] = row[col]
        row_results["Value"] = val_str # Add the cleaned 'Value' string

        # Handle empty value rows - add default/empty analysis columns
        if not val_str:
            row_results.update({
                'Classification': "Empty", 'Identifiers': "", 'SubValueCount': 0,
                'ConditionCount': 0, 'HasRangeInMain': False, 'HasMultiValueInMain': False,
                'HasRangeInCondition': False, 'HasMultipleConditions': False, 'MainItemCount': 0,
                'DetailedValueType': "Empty", 'Normalized Unit': "", 'Absolute Unit': "",
                'MainUnits': "", 'MainDistinctUnitCount': 0, 'MainUnitsConsistent': True,
                'ConditionUnits': "", 'ConditionDistinctUnitCount': 0, 'ConditionUnitsConsistent': True,
                'OverallUnitConsistency': True, 'ParsingErrorFlag': False,
                'SubValueUnitVariationSummary': "Main: None; Condition: None",
                 'MainNumericValues': "", 'ConditionNumericValues': "", 'MainMultipliers': "",
                 'ConditionMultipliers': "", 'MainBaseUnits': "", 'ConditionBaseUnits': "",
                 'NormalizedMainValues': "", 'NormalizedConditionValues': "",
                 'MinNormalizedValue': None, 'MaxNormalizedValue': None,
                 'SingleUnitForAllSubs': True, 'AllDistinctUnitsUsed': ""
            })
            results.append(row_results)
            continue # Move to next row

        # --- Perform Analysis Steps for non-empty values ---
        try:
            # 1. Classification and Structure Analysis (using detailed classifier)
            (cls, ids, sv_count, final_cond_item_count,
             rng_main, multi_main, rng_cond, multi_cond,
             final_main_item_count) = classify_value_type_detailed(val_str)

            row_results['Classification'] = cls
            row_results['Identifiers'] = ids
            row_results['SubValueCount'] = sv_count
            row_results['ConditionCount'] = final_cond_item_count
            row_results['HasRangeInMain'] = rng_main
            row_results['HasMultiValueInMain'] = multi_main
            row_results['HasRangeInCondition'] = rng_cond
            row_results['HasMultipleConditions'] = multi_cond
            row_results['MainItemCount'] = final_main_item_count # Add main item count

            # Create DetailedValueType string for summary
            if cls and cls not in ["Empty", "Invalid/Empty Structure", "Classification Error"]:
                dvt = f"{cls} [M:{final_main_item_count}][C:{final_cond_item_count}]"
                if sv_count > 1 : dvt += f" (x{sv_count})" # Indicate multiple sub-values
            else:
                dvt = cls if cls else "Unknown" # Show classification result directly
            row_results['DetailedValueType'] = dvt

            # 2. Normalization (Replace numbers with '$')
            # Apply to the original value string
            row_results['Normalized Unit'] = replace_numbers_keep_sign_all(val_str)
            # Optional: Add the version that keeps numbers in parentheses if needed
            # row_results['Normalized Unit (Parens Intact)'] = replace_numbers_keep_sign_outside_parens(val_str)

            # 3. Absolute Unit Resolution (Resolve to base units in structure)
            row_results["Absolute Unit"] = resolve_compound_unit(val_str, base_units, multipliers_dict)

            # 4. Unit Analysis (Main vs Condition, consistency)
            unit_analysis_results = analyze_value_units(val_str, base_units, multipliers_dict)
            row_results["MainUnits"] = ", ".join(safe_str(u) for u in unit_analysis_results["main_units"]) # Show all units found (incl None)
            row_results["MainDistinctUnitCount"] = len(unit_analysis_results["main_distinct_units"]) # Count of valid distinct units
            row_results["MainUnitsConsistent"] = unit_analysis_results["main_units_consistent"]
            row_results["ConditionUnits"] = ", ".join(safe_str(u) for u in unit_analysis_results["condition_units"])
            row_results["ConditionDistinctUnitCount"] = len(unit_analysis_results["condition_distinct_units"])
            row_results["ConditionUnitsConsistent"] = unit_analysis_results["condition_units_consistent"]
            row_results["OverallUnitConsistency"] = unit_analysis_results["overall_consistent"]

            # Unit variation summary string
            main_units_str = "None"
            if unit_analysis_results["main_distinct_units"]:
                main_units_sorted = sorted(list(unit_analysis_results["main_distinct_units"]))
                main_units_str = "Uniform: " + main_units_sorted[0] if unit_analysis_results["main_units_consistent"] else "Mixed: " + ", ".join(main_units_sorted)
            cond_units_str = "None"
            if unit_analysis_results["condition_distinct_units"]:
                cond_units_sorted = sorted(list(unit_analysis_results["condition_distinct_units"]))
                cond_units_str = "Uniform: " + cond_units_sorted[0] if unit_analysis_results["condition_units_consistent"] else "Mixed: " + ", ".join(cond_units_sorted)
            row_results["SubValueUnitVariationSummary"] = f"Main: {main_units_str}; Condition: {cond_units_str}"

            # 5. Numeric Value Extraction & Normalization (to base units)
            numeric_info = extract_numeric_info_for_value(val_str, base_units, multipliers_dict)
            row_results["MainNumericValues"] = ", ".join(safe_str(x) for x in numeric_info["main_numeric"])
            row_results["ConditionNumericValues"] = ", ".join(safe_str(x) for x in numeric_info["condition_numeric"])
            row_results["MainMultipliers"] = ", ".join(safe_str(x) for x in numeric_info["main_multipliers"])
            row_results["ConditionMultipliers"] = ", ".join(safe_str(x) for x in numeric_info["condition_multipliers"])
            row_results["MainBaseUnits"] = ", ".join(safe_str(x) for x in numeric_info["main_base_units"]) # Includes None
            row_results["ConditionBaseUnits"] = ", ".join(safe_str(x) for x in numeric_info["condition_base_units"]) # Includes None
            row_results["NormalizedMainValues"] = ", ".join(safe_str(x) for x in numeric_info["normalized_main"])
            row_results["NormalizedConditionValues"] = ", ".join(safe_str(x) for x in numeric_info["normalized_condition"])

            # Parsing error flag (if any part failed numeric/unit extraction)
            parsing_error = any(numeric_info["main_errors"]) or any(numeric_info["condition_errors"])
            row_results["ParsingErrorFlag"] = parsing_error

            # 6. Summaries and Derived Metrics
            # Collect all valid normalized numeric values and valid base units found
            all_normalized_numeric = [v for v in numeric_info["normalized_main"] + numeric_info["normalized_condition"] if isinstance(v, (int, float))]
            all_base_units_found = [u for u in numeric_info["main_base_units"] + numeric_info["condition_base_units"] if u is not None] # Filter None

            # Min/Max normalized value across main and condition
            min_val = min(all_normalized_numeric) if all_normalized_numeric else None
            max_val = max(all_normalized_numeric) if all_normalized_numeric else None
            row_results["MinNormalizedValue"] = min_val
            row_results["MaxNormalizedValue"] = max_val

            # Overall unit consistency check (alternative way using numeric_info)
            distinct_units_all_found = set(all_base_units_found)
            is_single_unit_overall = (len(distinct_units_all_found) <= 1)
            row_results["SingleUnitForAllSubs"] = is_single_unit_overall # True if 0 or 1 distinct unit found
            row_results["AllDistinctUnitsUsed"] = ", ".join(sorted(list(distinct_units_all_found))) if distinct_units_all_found else ""

            # Append the completed results for this row
            results.append(row_results)

        except Exception as e:
             st.error(f"Error analyzing row {index}, Value: '{val_str}': {e}")
             st.error(traceback.format_exc()) # Log detailed traceback
             # Append row with error information
             error_results = {"OriginalIndex": index, "Value": val_str}
             # Copy original columns
             for col in df_copy.columns:
                 if col != 'Value': error_results[col] = row[col]
             # Add error markers
             error_results.update({
                 'Classification': "Analysis Error",
                 'DetailedValueType': f"Error: {e}",
                 'ParsingErrorFlag': True
                 # Fill other columns with defaults?
             })
             results.append(error_results)

        # Progress indicator (optional)
        if (index + 1) % 100 == 0:
             st.write(f"DEBUG: Detailed analysis progress: {index + 1}/{total_rows} rows")


    st.write(f"DEBUG: Detailed Analysis Pipeline finished. Generated {len(results)} result rows.")

    # --- Create final DataFrame ---
    if not results:
        st.warning("Detailed analysis produced no results.")
        # Return empty frame with original columns + analysis columns if possible?
        # Define expected analysis columns
        analysis_cols = [
             'Classification', 'Identifiers', 'SubValueCount', 'ConditionCount',
             'HasRangeInMain', 'HasMultiValueInMain', 'HasRangeInCondition', 'HasMultipleConditions', 'MainItemCount',
             'DetailedValueType', 'Normalized Unit', 'Absolute Unit',
             'MainUnits', 'MainDistinctUnitCount', 'MainUnitsConsistent', 'ConditionUnits',
             'ConditionDistinctUnitCount', 'ConditionUnitsConsistent', 'OverallUnitConsistency',
             'ParsingErrorFlag', 'SubValueUnitVariationSummary', 'MainNumericValues', 'ConditionNumericValues',
             'MainMultipliers', 'ConditionMultipliers', 'MainBaseUnits', 'ConditionBaseUnits',
             'NormalizedMainValues', 'NormalizedConditionValues', 'MinNormalizedValue', 'MaxNormalizedValue',
             'SingleUnitForAllSubs', 'AllDistinctUnitsUsed'
        ]
        original_cols = df.columns.tolist()
        # Ensure 'Value' is present if it was in original
        if 'Value' not in original_cols: original_cols.append('Value')
        final_cols = original_cols + [col for col in analysis_cols if col not in original_cols]
        return pd.DataFrame(columns=final_cols)


    analysis_df = pd.DataFrame(results)

    # Drop the temporary 'OriginalIndex' column if it exists
    if "OriginalIndex" in analysis_df.columns:
        analysis_df = analysis_df.drop(columns=["OriginalIndex"])

    return analysis_df


# --- Wrapper function (Entry point) ---
# MODIFIED detailed_analysis function
def detailed_analysis(input_df: pd.DataFrame, mapping_file: str, output_file: str):
    """
    Performs detailed analysis on the input DataFrame using the mapping file.
    Saves the results to the specified output Excel file.

    Args:
        input_df (pd.DataFrame): DataFrame containing the data (must have 'Value' column).
        mapping_file (str): Path to the local 'mapping.xlsx' file.
        output_file (str): Path to save the resulting Excel file.

    Returns:
        str or None: The path to the output file on success, None on failure.
    """
    st.write("DEBUG: Starting detailed_analysis function...")
    try:
        # Read mapping file - raises FileNotFoundError or ValueError on issues
        base_units, multipliers_dict = read_mapping_file(mapping_file)
        st.write(f"DEBUG: Using {len(base_units)} base units and {len(multipliers_dict)} multipliers for detailed analysis.")
    except (FileNotFoundError, ValueError) as e:
        st.error(f"Error reading mapping file '{mapping_file}': {e}")
        return None # Indicate failure: Cannot proceed without mapping
    except Exception as e:
        st.error(f"Unexpected error reading mapping file '{mapping_file}': {e}")
        return None

    # --- Input Validation ---
    if not isinstance(input_df, pd.DataFrame):
         st.error("Detailed Analysis Error: Input must be a pandas DataFrame.")
         return None
    if 'Value' not in input_df.columns:
        st.error("Detailed Analysis Error: Input DataFrame must contain a column named 'Value'.")
        return None

    # --- Run the main analysis pipeline ---
    try:
        # Pass a copy of the input df to avoid modification side effects
        analysis_df = detailed_analysis_pipeline(input_df.copy(), base_units, multipliers_dict)
    except Exception as e:
         st.error(f"Error during detailed analysis pipeline execution: {e}")
         st.error(traceback.format_exc())
         return None # Indicate failure

    # Check if the analysis produced a result
    if analysis_df is None: # Check if pipeline function itself failed critically
         st.error("Detailed analysis pipeline returned None. Aborting save.")
         return None
    if analysis_df.empty:
         # Decide if saving an empty file is desired or if it indicates an issue
         st.warning("Detailed analysis resulted in an empty DataFrame. Saving empty file.")
         # Proceed to save the empty DataFrame.

    # --- Save the results ---
    try:
        analysis_df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"[✓] Detailed analysis saved to '{output_file}'.")
        st.write(f"DEBUG: Detailed analysis saved to '{output_file}'. Shape: {analysis_df.shape}")
        return output_file # Return path on success
    except Exception as e:
        st.error(f"Error writing detailed analysis to '{output_file}': {e}")
        return None # Indicate failure
