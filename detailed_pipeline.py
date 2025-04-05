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
    classifications   = []
    identifiers_list  = []
    sub_val_counts    = []
    condition_counts  = []
    any_range_main    = []
    any_multi_main    = []
    any_range_cond    = []
    any_multi_cond    = []
    detailed_value_types = []

    for val in df['Value']:
        val_str = str(val)
        (cls, ids, sv_count, final_cond_item_count,
         rng_main, multi_main, rng_cond, multi_cond,
         final_main_item_count) = classify_value_type_detailed(val_str)

        classifications.append(cls)
        identifiers_list.append(ids)
        sub_val_counts.append(sv_count)
        condition_counts.append(final_cond_item_count)
        any_range_main.append(rng_main)
        any_multi_main.append(multi_main)
        any_range_cond.append(rng_cond)
        any_multi_cond.append(multi_cond)
        if cls:
            dvt = f"{cls} [{final_main_item_count}][{final_cond_item_count}] x{sv_count}"
        else:
            dvt = ""
        detailed_value_types.append(dvt)

    df['Classification'] = classifications
    df['Identifiers'] = identifiers_list
    df['SubValueCount'] = sub_val_counts
    df['ConditionCount'] = condition_counts
    df['HasRangeInMain'] = any_range_main
    df['HasMultiValueInMain'] = any_multi_main
    df['HasRangeInCondition'] = any_range_cond
    df['HasMultipleConditions'] = any_multi_cond
    df['DetailedValueType'] = detailed_value_types

    # Resolve absolute units (if "Normalized Unit" exists)
    if "Normalized Unit" in df.columns:
        unit_source = df["Normalized Unit"]
    else:
        unit_source = df["Value"]
    resolved_units = []
    for x in unit_source:
        x_str = str(x)
        resolved_units.append(resolve_compound_unit(x_str, base_units, multipliers_dict))
    df["Absolute Unit"] = resolved_units

    # Analyze units in main vs condition
    main_units_list = []
    main_distinct_count_list = []
    main_consistent_list = []
    condition_units_list = []
    condition_distinct_count_list = []
    condition_consistent_list = []
    main_sub_analysis_list = []
    condition_sub_analysis_list = []

    for val in df['Value']:
        val_str = str(val)
        ua = analyze_value_units(val_str, base_units, multipliers_dict)
        main_units_list.append(", ".join(safe_str(x) for x in ua["main_units"]))
        main_distinct_count_list.append(len(ua["main_distinct_units"]))
        main_consistent_list.append(ua["main_units_consistent"])
        condition_units_list.append(", ".join(safe_str(x) for x in ua["condition_units"]))
        condition_distinct_count_list.append(len(ua["condition_distinct_units"]))
        condition_consistent_list.append(ua["condition_units_consistent"])
        main_sub_analysis_list.append(str(ua["main_sub_analysis"]))
        condition_sub_analysis_list.append(str(ua["condition_sub_analysis"]))

    df["MainUnits"] = main_units_list
    df["MainDistinctUnitCount"] = main_distinct_count_list
    df["MainUnitsConsistent"] = main_consistent_list
    df["ConditionUnits"] = condition_units_list
    df["ConditionDistinctUnitCount"] = condition_distinct_count_list
    df["ConditionUnitsConsistent"] = condition_consistent_list
    df["MainSubAnalysis"] = main_sub_analysis_list
    df["ConditionSubAnalysis"] = condition_sub_analysis_list

    # Numeric values, multipliers, base units, etc.
    main_numeric_values_list = []
    condition_numeric_values_list = []
    main_multiplier_list = []
    condition_multiplier_list = []
    main_base_units_list = []
    condition_base_units_list = []
    normalized_main_values_list = []
    normalized_condition_values_list = []
    overall_unit_consistency_list = []
    parsing_error_flag_list = []
    sub_value_variation_summary_list = []
    min_value_list = []
    max_value_list = []
    single_unit_list = []
    distinct_units_all_list = []

    for val in df['Value']:
        val_str = str(val)
        num_info = extract_numeric_info_for_value(val_str, base_units, multipliers_dict)
        main_numeric_values_list.append(", ".join(safe_str(x) for x in num_info["main_numeric"]))
        condition_numeric_values_list.append(", ".join(safe_str(x) for x in num_info["condition_numeric"]))
        main_multiplier_list.append(", ".join(safe_str(x) for x in num_info["main_multipliers"]))
        condition_multiplier_list.append(", ".join(safe_str(x) for x in num_info["condition_multipliers"]))
        main_base_units_list.append(", ".join(safe_str(x) for x in num_info["main_base_units"]))
        condition_base_units_list.append(", ".join(safe_str(x) for x in num_info["condition_base_units"]))
        normalized_main_values_list.append(", ".join(safe_str(x) for x in num_info["normalized_main"]))
        normalized_condition_values_list.append(", ".join(safe_str(x) for x in num_info["normalized_condition"]))

        ua = analyze_value_units(val_str, base_units, multipliers_dict)
        overall_consistency = ua["main_units_consistent"] and ua["condition_units_consistent"]
        overall_unit_consistency_list.append(overall_consistency)

        parsing_error = any(num_info["main_errors"]) or any(num_info["condition_errors"])
        parsing_error_flag_list.append(parsing_error)

        # Summarize unit variation
        main_variation = "None"
        if ua["main_distinct_units"]:
            if len(ua["main_distinct_units"]) == 1:
                main_variation = "Uniform: " + safe_str(ua["main_distinct_units"][0])
            else:
                main_variation = "Mixed"
        condition_variation = "None"
        if ua["condition_distinct_units"]:
            if len(ua["condition_distinct_units"]) == 1:
                condition_variation = "Uniform: " + safe_str(ua["condition_distinct_units"][0])
            else:
                condition_variation = "Mixed"
        sub_value_variation_summary_list.append(f"Main: {main_variation}; Condition: {condition_variation}")

        # Min/Max normalized
        all_normalized_values = []
        all_units_used = []
        for i, val_num in enumerate(num_info["normalized_main"]):
            if not num_info["main_errors"][i] and (val_num is not None):
                all_normalized_values.append(val_num)
                all_units_used.append(num_info["main_base_units"][i])
        for i, val_num in enumerate(num_info["normalized_condition"]):
            if not num_info["condition_errors"][i] and (val_num is not None):
                all_normalized_values.append(val_num)
                all_units_used.append(num_info["condition_base_units"][i])
        if all_normalized_values:
            min_val = min(all_normalized_values)
            max_val = max(all_normalized_values)
        else:
            min_val = None
            max_val = None
        distinct_units_all = set(u for u in all_units_used if u and u.lower() != "none")
        is_single_unit = (len(distinct_units_all) <= 1)
        min_value_list.append(min_val)
        max_value_list.append(max_val)
        single_unit_list.append(is_single_unit)
        distinct_units_all_list.append(", ".join(distinct_units_all) if distinct_units_all else "")

    df["MainNumericValues"] = main_numeric_values_list
    df["ConditionNumericValues"] = condition_numeric_values_list
    df["MainMultipliers"] = main_multiplier_list
    df["ConditionMultipliers"] = condition_multiplier_list
    df["MainBaseUnits"] = main_base_units_list
    df["ConditionBaseUnits"] = condition_base_units_list
    df["NormalizedMainValues"] = normalized_main_values_list
    df["NormalizedConditionValues"] = normalized_condition_values_list
    df["OverallUnitConsistency"] = overall_unit_consistency_list
    df["ParsingErrorFlag"] = parsing_error_flag_list
    df["SubValueUnitVariationSummary"] = sub_value_variation_summary_list
    df["MinNormalizedValue"] = min_value_list
    df["MaxNormalizedValue"] = max_value_list
    df["SingleUnitForAllSubs"] = single_unit_list
    df["AllDistinctUnitsUsed"] = distinct_units_all_list

    return df

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
