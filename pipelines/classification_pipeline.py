# pipelines/classification_pipeline.py

import pandas as pd
import re
from mapping_manager import read_mapping_file
from config import (
    MAPPING_FILE_LOCAL,
    PROCESSED_OUTPUT_FILE,
    USER_INPUT_FILE,
    CLASSIFIED_OUTPUT_FILE,
    QA_NEW_FILE,
    FINAL_COMBINED_FILE
)

#############################################
# Basic Classification Patterns & Helpers
#############################################

PATTERNS_12_FIXED = [
    (re.compile(r"^[+-]?\d+(?:\.\d+)?$"), "Number"),
    (re.compile(r"^[+-]?\d+(?:\.\d+)?[a-zA-Zµ]+$"), "Single Value"),
    (re.compile(r"^[+-]?\d+(?:\.\d+)?[a-zA-Zµ]+ @ [+-]?\d+(?:\.\d+)?[a-zA-Zµ]+$"), "Single Value with Single Condition"),
    (re.compile(r"^[+-]?\d+(?:\.\d+)?[a-zA-Zµ]+ @ [+-]?\d+(?:\.\d+)?[a-zA-Zµ]+ to [+-]?\d+(?:\.\d+)?[a-zA-Zµ]+$"), "Single Value with Range Condition"),
    (re.compile(r"^[+-]?\d+(?:\.\d+)? @ [+-]?\d+(?:\.\d+)?[a-zA-Zµ]+$"), "Number with Single Condition"),
    (re.compile(r"^[+-]?\d+(?:\.\d+)?[a-zA-Zµ]+ to [+-]?\d+(?:\.\d+)?[a-zA-Zµ]+ @ [+-]?\d+(?:\.\d+)?[a-zA-Zµ]+$"), "Range Value Single Condition"),
    (re.compile(r"^[+-]?\d+(?:\.\d+)?[a-zA-Zµ]+ to [+-]?\d+(?:\.\d+)?[a-zA-Zµ]+ @ [+-]?\d+(?:\.\d+)?[a-zA-Zµ]+ to [+-]?\d+(?:\.\d+)?[a-zA-Zµ]+$"), "Range Value with Range Condition"),
    (re.compile(r"^[+-]?\d+(?:\.\d+)?[a-zA-Zµ]+ to [+-]?\d+(?:\.\d+)?[a-zA-Zµ]+$"), "Range Values"),
    (re.compile(r"^[+-]?\d+(?:\.\d+)?[a-zA-Zµ]+ @ [+-]?\d+(?:\.\d+)?[a-zA-Zµ]+(?:, [+-]?\d+(?:\.\d+)?[a-zA-Zµ]+)+$"), "Single Value Multi Condition"),
    (re.compile(r"^(?:[+-]?\d+(?:\.\d+)?[a-zA-Zµ]+(?:, [+-]?\d+(?:\.\d+)?[a-zA-Zµ]+)+) @ [+-]?\d+(?:\.\d+)?[a-zA-Zµ]+$"), "Multi Value with Single Condition"),
    (re.compile(r"^[+-]?\d+(?:\.\d+)?[a-zA-Zµ]+ to [+-]?\d+(?:\.\d+)?[a-zA-Zµ]+ @ [+-]?\d+(?:\.\d+)?[a-zA-Zµ]+(?:, [+-]?\d+(?:\.\d+)?[a-zA-Zµ]+)+$"), "Range Value with Multi Condition"),
    (re.compile(r"^(?:[+-]?\d+(?:\.\d+)?[a-zA-Zµ]+(?:, [+-]?\d+(?:\.\d+)?[a-zA-Zµ]+)+)$"), "Multi Values"),
]

PATTERNS_12_FLEX = [
    (re.compile(r"^-?\d+(?:\.\d+)?$"), "Number"),
    (re.compile(r"^-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+(?:\s*\([^)]+\))?$"), "Single Value"),
    (re.compile(r"^-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+(?:\s*\([^)]+\))?\s*@\s*-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+(?:\s*\([^)]+\))?$"), "Single Value with Single Condition"),
    (re.compile(r"^-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+(?:\s*\([^)]+\))?\s*@\s*-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+(?:\s*\([^)]+\))?\s+to\s+-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+(?:\s*\([^)]+\))?$"), "Single Value with Range Condition"),
    (re.compile(r"^-?\d+(?:\.\d+)?\s*@\s*-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+(?:\s*\([^)]+\))?$"), "Number with Single Condition"),
    (re.compile(r"^-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+(?:\s*\([^)]+\))?\s+to\s+-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+(?:\s*\([^)]+\))?\s*@\s*-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+(?:\s*\([^)]+\))?$"), "Range Value Single Condition"),
    (re.compile(r"^-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+(?:\s*\([^)]+\))?\s+to\s+-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+(?:\s*\([^)]+\))?\s*@\s*-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+(?:\s*\([^)]+\))?\s+to\s+-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+(?:\s*\([^)]+\))?$"), "Range Value with Range Condition"),
    (re.compile(r"^-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+(?:\s*\([^)]+\))?\s+to\s+-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+(?:\s*\([^)]+\))?$"), "Range Values"),
    (re.compile(r"^-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+(?:\s*\([^)]+\))?\s*@\s*-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+(?:\s*\([^)]+\))?(?:\s*,\s*-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+(?:\s*\([^)]+\))?)+$"), "Single Value Multi Condition"),
    (re.compile(r"^(?:-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+(?:\s*\([^)]+\))?(?:\s*,\s*-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+(?:\s*\([^)]+\))?)+)\s*@\s*-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+(?:\s*\([^)]+\))?$"), "Multi Value with Single Condition"),
    (re.compile(r"^-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+(?:\s*\([^)]+\))?\s+to\s+-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+(?:\s*\([^)]+\))?\s*@\s*-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+(?:\s*\([^)]+\))?(?:\s*,\s*-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+(?:\s*\([^)]+\))?)+$"), "Range Value with Multi Condition"),
    (re.compile(r"^(?:-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+(?:\s*\([^)]+\))?(?:\s*,\s*-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+(?:\s*\([^)]+\))?)+)$"), "Multi Values"),
]

def detect_value_type_basic(text: str, pattern_list) -> str:
    text = text.strip()
    for regex, value_type in pattern_list:
        if regex.fullmatch(text):
            return value_type
    return "Unknown"

def priority_classification(cell_value: str) -> str:
    fixed_type = detect_value_type_basic(cell_value, PATTERNS_12_FIXED)
    if fixed_type != "Unknown":
        return f"{fixed_type} (standard)"
    flex_type = detect_value_type_basic(cell_value, PATTERNS_12_FLEX)
    if flex_type != "Unknown":
        return f"{flex_type} (flex)"
    return "Unknown"

def extract_identifiers(s: str) -> str:
    ids = re.findall(r'\(([^)]*)\)', s)
    return ', '.join(ids) if ids else ''

def replace_numbers_keep_sign_all(s: str) -> str:
    return re.sub(r'([+-]?)\d+(\.\d+)?', r'\1$', s)

def replace_numbers_keep_sign_outside_parens(s: str) -> str:
    result = []
    in_paren = False
    i = 0
    while i < len(s):
        char = s[i]
        if char == '(':
            in_paren = True
            result.append(char)
            i += 1
        elif char == ')':
            in_paren = False
            result.append(char)
            i += 1
        else:
            if in_paren:
                result.append(char)
                i += 1
            else:
                match_ = re.match(r'([+-]?)\d+(\.\d+)?', s[i:])
                if match_:
                    sign = match_.group(1)
                    result.append(sign + '$')
                    i += len(match_.group(0))
                else:
                    result.append(char)
                    i += 1
    return "".join(result)

#############################################
# Pipeline Functions
#############################################

def basic_classification(input_path: str, output_path: str) -> str:
    """
    Reads an Excel file from input_path, applies basic classification on the 'Value' column,
    and writes the result to output_path.
    """
    df = pd.read_excel(input_path, sheet_name=0)
    column_to_classify = "Value"
    if column_to_classify not in df.columns:
        raise KeyError(f"Column '{column_to_classify}' not found in the Excel file.")
    
    df['Value Type'] = df[column_to_classify].astype(str).apply(priority_classification)
    df['Identifiers'] = df[column_to_classify].astype(str).apply(extract_identifiers)
    df['Normalized Unit'] = df[column_to_classify].astype(str).apply(replace_numbers_keep_sign_all)
    df['Normalized Unit_edit'] = df[column_to_classify].astype(str).apply(replace_numbers_keep_sign_outside_parens)
    df.to_excel(output_path, index=False)
    print(f"[✓] Basic classification saved to '{output_path}'")
    return output_path

def detailed_analysis(input_path: str, mapping_file: str, output_file: str):
    """
    Performs detailed analysis on an Excel file using mapping_file for unit definitions,
    and writes the analysis output to output_file.
    """
    try:
        base_units, multipliers_dict = read_mapping_file(mapping_file)
    except Exception as e:
        print(e)
        return
    try:
        df = pd.read_excel(input_path)
    except Exception as e:
        print(f"Error reading '{input_path}': {e}")
        return
    if 'Value' not in df.columns:
        print("Error: The input Excel file must contain a column named 'Value'.")
        return
    df = detailed_analysis_pipeline(df, base_units, multipliers_dict)
    try:
        df.to_excel(output_file, index=False)
        print(f"[✓] Detailed analysis saved to '{output_file}'.")
    except Exception as e:
        print(f"Error writing to '{output_file}': {e}")

def detailed_analysis_pipeline(df, base_units, multipliers_dict):
    """
    Processes the DataFrame to add detailed analysis columns based on the 'Value' column.
    This includes classification, unit resolution, and numeric extraction.
    """
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

    # Resolve absolute units using 'Normalized Unit' if available; else use 'Value'
    unit_source = df["Normalized Unit"] if "Normalized Unit" in df.columns else df["Value"]
    resolved_units = []
    for x in unit_source:
        x_str = str(x)
        resolved_units.append(resolve_compound_unit(x_str, base_units, multipliers_dict))
    df["Absolute Unit"] = resolved_units

    # Analyze units in main and condition parts
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
        main_units_list.append(", ".join(str(x) for x in ua["main_units"]))
        main_distinct_count_list.append(len(ua["main_distinct_units"]))
        main_consistent_list.append(ua["main_units_consistent"])
        condition_units_list.append(", ".join(str(x) for x in ua["condition_units"]))
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

    # Extract numeric values, multipliers, and errors
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
        main_numeric_values_list.append(", ".join(str(x) for x in num_info["main_numeric"]))
        condition_numeric_values_list.append(", ".join(str(x) for x in num_info["condition_numeric"]))
        main_multiplier_list.append(", ".join(str(x) for x in num_info["main_multipliers"]))
        condition_multiplier_list.append(", ".join(str(x) for x in num_info["condition_multipliers"]))
        main_base_units_list.append(", ".join(str(x) for x in num_info["main_base_units"]))
        condition_base_units_list.append(", ".join(str(x) for x in num_info["condition_base_units"]))
        normalized_main_values_list.append(", ".join(str(x) for x in num_info["normalized_main"]))
        normalized_condition_values_list.append(", ".join(str(x) for x in num_info["normalized_condition"]))
        ua = analyze_value_units(val_str, base_units, multipliers_dict)
        overall_consistency = ua["main_units_consistent"] and ua["condition_units_consistent"]
        overall_unit_consistency_list.append(overall_consistency)
        parsing_error = any(num_info["main_errors"]) or any(num_info["condition_errors"])
        parsing_error_flag_list.append(parsing_error)
        main_variation = "None"
        if ua["main_distinct_units"]:
            if len(ua["main_distinct_units"]) == 1:
                main_variation = "Uniform: " + str(ua["main_distinct_units"][0])
            else:
                main_variation = "Mixed"
        condition_variation = "None"
        if ua["condition_distinct_units"]:
            if len(ua["condition_distinct_units"]) == 1:
                condition_variation = "Uniform: " + str(ua["condition_distinct_units"][0])
            else:
                condition_variation = "Mixed"
        sub_value_variation_summary_list.append(f"Main: {main_variation}; Condition: {condition_variation}")
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
        distinct_units_all = set(u for u in all_units_used if u and str(u).lower() != "none")
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

def combine_results(processed_file: str = PROCESSED_OUTPUT_FILE,
                    analysis_file: str = QA_NEW_FILE,
                    output_file: str = FINAL_COMBINED_FILE):
    """
    Combines results from the fixed processing pipeline and the classification/analysis pipeline.
    Merges data based on the 'Value' column and writes the final output to output_file.
    """
    df_processed = pd.read_excel(processed_file, sheet_name="Combined")
    df_analysis = pd.read_excel(analysis_file)
    df_merged = df_processed.merge(
        df_analysis[["Value", "Normalized Unit", "Absolute Unit", "Classification", "DetailedValueType"]],
        how="left",
        left_on="Main Key",
        right_on="Value",
        suffixes=("", "_analysis")
    )
    df_merged.drop(columns=["Value_analysis"], inplace=True, errors='ignore')
    final_columns = [
        "Main Key",
        "Normalized Unit",
        "Absolute Unit",
        "Classification",
        "DetailedValueType",
        "Value",
        "Category",
        "Attribute",
        "Code",
        "Sheet"
    ]
    existing_cols = [c for c in final_columns if c in df_merged.columns]
    df_merged = df_merged[existing_cols]
    df_merged.to_excel(output_file, index=False)
    print(f"[✓] Final combined file saved to '{output_file}'")

#############################################
# Placeholder Functions for Missing Logic
#############################################
# The following functions are placeholders based on your original code.
# Replace them with your actual implementations as needed.

def classify_value_type_detailed(raw_value: str):
    """
    Placeholder for detailed classification of a value.
    Returns a tuple:
    (final_class, identifiers, sub_value_count, final_condition_item_count,
     has_range_in_main, has_multi_value_in_main, has_range_in_condition, has_multiple_conditions, final_main_item_count)
    """
    return ("DummyClass", "", 1, 1, False, False, False, False, 1)

def resolve_compound_unit(normalized_unit: str, base_units, multipliers_dict):
    """
    Placeholder for resolving compound units.
    """
    return normalized_unit

def analyze_value_units(raw_value: str, base_units, multipliers_dict):
    """
    Placeholder for analyzing unit parts.
    Returns a dictionary with keys:
    'main_units', 'main_distinct_units', 'main_units_consistent',
    'condition_units', 'condition_distinct_units', 'condition_units_consistent',
    'main_sub_analysis', 'condition_sub_analysis'
    """
    return {
        "main_units": [raw_value],
        "main_distinct_units": [raw_value],
        "main_units_consistent": True,
        "condition_units": [],
        "condition_distinct_units": [],
        "condition_units_consistent": True,
        "main_sub_analysis": [],
        "condition_sub_analysis": []
    }

def extract_numeric_info_for_value(raw_value: str, base_units, multipliers_dict):
    """
    Placeholder for extracting numeric information.
    Returns a dictionary with keys:
    'main_numeric', 'condition_numeric',
    'main_multipliers', 'condition_multipliers',
    'main_base_units', 'condition_base_units',
    'normalized_main', 'normalized_condition',
    'main_errors', 'condition_errors'
    """
    return {
        "main_numeric": [raw_value],
        "condition_numeric": [],
        "main_multipliers": [],
        "condition_multipliers": [],
        "main_base_units": [raw_value],
        "condition_base_units": [],
        "normalized_main": [raw_value],
        "normalized_condition": [],
        "main_errors": [False],
        "condition_errors": [False]
    }
