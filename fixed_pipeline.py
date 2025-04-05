#############################################
# MODULE: FIXED PROCESSING PIPELINE
# Purpose: Implements the first pipeline logic:
#          - Classifies input strings based on structure.
#          - Extracts value/unit parts according to classification.
#          - Generates structured output rows with codes (e.g., SV-V, RV-Un).
#############################################

import pandas as pd
import re
import gc
import io # Make sure io is imported
import streamlit as st # For error/warning/debug messages
import traceback

# Import necessary functions and constants from other modules
from mapping_utils import read_mapping_file, LOCAL_BASE_UNITS
# Import specific helpers needed by this pipeline's functions
from analysis_helpers import (
    classify_value_type_detailed, split_outside_parens, fix_exceptions,
    MULTIPLIER_MAPPING, extract_numeric_and_unit_analysis,
    remove_parentheses_detailed, extract_identifiers_detailed
)


# --- Helper functions specific to Fixed Pipeline output generation ---

def extract_block_texts(main_key, category_name):
    """
    Extracts logical text blocks from a value string based on its classification category.
    These blocks correspond to the parts that will receive codes (e.g., min value, max value, condition).

    Args:
        main_key (str): The (potentially cleaned) value string.
        category_name (str): The classification string (e.g., "Range Value with Single Condition").

    Returns:
        list[str]: A list of extracted text blocks.
    """
    main_key = str(main_key).strip()
    parts = []

    # Handle structures with conditions first
    if " with " in category_name:
        # Split category name, e.g., "Range Value", "Single Condition"
        # main_type, cond_type = category_name.split(" with ", 1) # Not needed directly

        # Split the input string by '@' outside parentheses
        at_split = split_outside_parens(main_key, ['@'])
        main_part = ""
        cond_part = ""
        if len(at_split) >= 1:
             main_part = at_split[0].strip()
        if len(at_split) >= 2:
             cond_part = "@".join(at_split[1:]).strip() # Rejoin if multiple '@'

        # Extract blocks from the main part based on its structure type
        if category_name.startswith("Range Value"):
            # Split main part by ' to ' (case-insensitive)
            range_split = split_outside_parens(main_part, [' to ']) # Crude split, assumes ' to ' is delimiter
            # Refine: Use regex split for ' to ' with spaces?
            range_split = re.split(r'\s+to\s+', main_part, flags=re.IGNORECASE)
            parts.extend(p.strip() for p in range_split if p.strip())
        elif category_name.startswith("Multi Value"):
            # This category indicates multiple comma-separated values *at the top level*.
            # process_single_key should handle splitting these. If this function receives
            # such a category, it implies an issue. Assume it's called on a *chunk* which
            # shouldn't be "Multi Value". Fallback: split by comma? Or treat as single?
            # Let's assume it receives a classification like "Single Value" if called on a chunk.
             st.warning(f"DEBUG: extract_block_texts received main category 'Multi Value' for '{main_key}'. Treating as single block.")
             parts.append(main_part) # Fallback
        else: # Single Value, Number, Complex Single
            parts.append(main_part)

        # Extract blocks from the condition part based on its structure type
        if "Range Condition" in category_name:
             range_split = re.split(r'\s+to\s+', cond_part, flags=re.IGNORECASE)
             parts.extend(p.strip() for p in range_split if p.strip())
        elif "Multiple Conditions" in category_name:
             # Split condition part by comma outside parentheses
             cond_blocks = split_outside_parens(cond_part, [','])
             parts.extend(p.strip() for p in cond_blocks if p.strip())
        elif "Single Condition" in category_name: # Check if cond_part is not empty
            if cond_part:
                 parts.append(cond_part)
        # Else: No condition type specified or condition part was empty

    # Handle structures without conditions
    elif category_name.startswith("Range Value"):
        range_split = re.split(r'\s+to\s+', main_key, flags=re.IGNORECASE)
        parts.extend(p.strip() for p in range_split if p.strip())
    elif category_name.startswith("Multi Value"):
        # As above, this category should ideally be handled by the caller splitting chunks.
        st.warning(f"DEBUG: extract_block_texts received category 'Multi Value' for '{main_key}'. Treating as single block.")
        parts.append(main_key) # Fallback
    elif category_name.startswith("Single Value") or category_name.startswith("Number") or category_name.startswith("Complex Single"):
        parts.append(main_key)
    elif category_name.startswith("Multiple"): # Handle "Multiple (2x)..." types
        # Again, caller should split. This is a fallback.
        st.warning(f"DEBUG: extract_block_texts received category '{category_name}'. Treating as single block.")
        parts.append(main_key)
    elif category_name == "Empty" or category_name.startswith("Invalid"):
        pass # Return empty list
    else: # Unknown category
        st.warning(f"DEBUG: Unknown category '{category_name}' in extract_block_texts. Treating '{main_key}' as single block.")
        parts.append(main_key)

    # Final filter for any empty strings resulting from splits
    return [p for p in parts if p]

####identifier update




def parse_value_unit_identifier(raw_chunk, base_units, multipliers_dict):
    """
    Parses a raw text block (expected to be "number + unit" or just "unit") into:
      - a numeric part (as a string)
      - a recognized base unit (could be "A" or "A (Tc)" if both are in base_units)
    preserving parentheses if the leftover text after numeric extraction matches
    a known base unit exactly.

    Returns: (value_str, base_unit_str)

    Example:
      raw_chunk = "357 A (Tc)"
      base_units = {"A", "A (Tc)"}
      => returns ("357", "A (Tc)")

      raw_chunk = "50 A (Tc) typical"
      => leftover is "A (Tc) typical" (not exactly "A (Tc)"), so parentheses are removed
         or the robust parser tries to parse it. 
    """

    # Step 0: Trim whitespace
    raw_chunk_stripped = raw_chunk.strip()
    if not raw_chunk_stripped:
        return ("", "")

    # ---------------------------------------------------------------
    # Step 1: Extract any leading numeric portion (including optional
    # multiplier symbol, e.g. "k", "M", or "µ/μ") with a regex.
    #   - Example: "357 A (Tc)" => numeric_part="357", leftover="A (Tc)"
    #   - If no numeric portion is found, leftover=raw_chunk_stripped
    # ---------------------------------------------------------------
    match = re.match(r'^([+-]?\d+(?:\.\d+)?(?:[kMµμ]?))\s+(.*)$', raw_chunk_stripped)
    if match:
        numeric_part = match.group(1).strip()  # e.g. "357"
        leftover = match.group(2).strip()      # e.g. "A (Tc)"
    else:
        # No numeric portion found => the entire chunk is leftover
        numeric_part = ""
        leftover = raw_chunk_stripped

    # ---------------------------------------------------------------
    # Step 2: Conditionally remove parentheses from leftover
    #   - If leftover is exactly in base_units, skip removing parentheses
    #   - Else, remove them
    # ---------------------------------------------------------------
    if leftover in base_units:
        cleaned_leftover = leftover
    else:
        # If leftover has extra text or isn't exactly recognized,
        # remove extraneous parentheses. E.g. "A (Tc) typical" => "A  typical"
        cleaned_leftover = remove_parentheses_detailed(leftover).strip()

    # ---------------------------------------------------------------
    # Step 3: Combine numeric + leftover for robust parsing
    #   "357" + "A (Tc)" => "357 A (Tc)"
    # ---------------------------------------------------------------
    if numeric_part and cleaned_leftover:
        chunk_for_parsing = f"{numeric_part} {cleaned_leftover}"
    else:
        # If either is empty, just use the one that isn't
        chunk_for_parsing = numeric_part or cleaned_leftover

    if not chunk_for_parsing:
        return ("", "")

    # ---------------------------------------------------------------
    # Step 4: Pass to your robust parser (extract_numeric_and_unit_analysis)
    # ---------------------------------------------------------------
    num_val, multi_sym, base_unit, norm_val, err_flag = extract_numeric_and_unit_analysis(
        chunk_for_parsing, base_units, multipliers_dict
    )
    
    # Optional debug
    # print("DEBUG:", {
    #    "raw_chunk": raw_chunk,
    #    "numeric_part": numeric_part,
    #    "leftover": leftover,
    #    "cleaned_leftover": cleaned_leftover,
    #    "chunk_for_parsing": chunk_for_parsing,
    #    "base_unit_after_parse": base_unit,
    #    "err_flag": err_flag
    # })

    # ---------------------------------------------------------------
    # Step 5: Convert to final (value_string, base_unit_string)
    # ---------------------------------------------------------------
    if err_flag:
        # If parsing failed, decide how to handle. 
        # Possibly just return chunk_for_parsing as 'value' to indicate an error.
        return (chunk_for_parsing, "")

    value_for_output = ""
    base_unit_for_output = ""

    # If we got a numeric value
    if num_val is not None:
        # If it's an integer or a float .is_integer() => return an integer string
        if (isinstance(num_val, int) or (isinstance(num_val, float) and num_val.is_integer())):
            value_for_output = str(int(num_val))  # e.g. 357 -> "357"
        else:
            value_for_output = str(num_val)       # e.g. 0.8 -> "0.8"

        # If we recognized a multiplier (like 'k'), append it
        if multi_sym and multi_sym != "1":
            value_for_output += multi_sym
        
        # Use the recognized base unit if found
        if base_unit:
            base_unit_for_output = base_unit
        else:
            base_unit_for_output = ""
    
    # If there's no numeric_val but a recognized base unit
    elif base_unit:
        value_for_output = ""
        base_unit_for_output = base_unit
    
    else:
        # Fallback if nothing recognized but no error
        # Return chunk_for_parsing as the "value"
        return (chunk_for_parsing, "")

    return (value_for_output, base_unit_for_output)




def get_code_prefixes_for_category(category_name):
    """
    Maps a classification category string to a list of dictionaries,
    where each dictionary defines the codes and attributes for a logical block
    (e.g., main value, condition, range min, range max).

    Args:
        category_name (str): The classification string.

    Returns:
        list[dict]: List of block definitions. Each dict has 'prefix', 'codes', 'attributes'.
                    Example: {"prefix": "SV-", "codes": ["SV-V", "SV-U"], "attributes": ["Value", "Unit"]}
    """
    # This mapping needs to align with the category names produced by classify_value_type_detailed
    # and the blocks expected by extract_block_texts.
    # Prefixes: SN=Number, SV=SingleValue, CX=Complex, RV=RangeValue (Main)
    #           SC=SingleCondition, RC=RangeCondition, MC=MultiCondition
    # Suffixes: V=Value, U=Unit, Vn=ValueMin, Un=UnitMin, Vx=ValueMax, Ux=UnitMax
    #           V1/U1, V2/U2 for multiple conditions

    # --- Structures without Conditions ---
    if category_name == "Number":
        return [{"prefix": "SN-", "codes": ["SN-V", "SN-U"], "attributes": ["Value", "Unit"]}]
    elif category_name == "Single Value":
        return [{"prefix": "SV-", "codes": ["SV-V", "SV-U"], "attributes": ["Value", "Unit"]}]
    elif category_name == "Complex Single":
         return [{"prefix": "CX-", "codes": ["CX-V", "CX-U"], "attributes": ["Value", "Unit"]}]
    elif category_name == "Range Value": # e.g., "10A to 20A"
        return [
            {"prefix": "RV-", "codes": ["RV-Vn", "RV-Un"], "attributes": ["Value", "Unit"]}, # Min/Start
            {"prefix": "RV-", "codes": ["RV-Vx", "RV-Ux"], "attributes": ["Value", "Unit"]}  # Max/End
        ]
    # --- Structures with Single Condition ---
    elif category_name == "Number with Single Condition":
        return [
            {"prefix": "SN-", "codes": ["SN-V", "SN-U"], "attributes": ["Value", "Unit"]}, # Number
            {"prefix": "SC-", "codes": ["SC-V", "SC-U"], "attributes": ["Value", "Unit"]}  # Condition
        ]
    elif category_name == "Single Value with Single Condition":
        return [
            {"prefix": "SV-", "codes": ["SV-V", "SV-U"], "attributes": ["Value", "Unit"]}, # Value
            {"prefix": "SC-", "codes": ["SC-V", "SC-U"], "attributes": ["Value", "Unit"]}  # Condition
        ]
    elif category_name == "Complex Single with Single Condition": # Added for completeness
         return [
             {"prefix": "CX-", "codes": ["CX-V", "CX-U"], "attributes": ["Value", "Unit"]}, # Complex Value
             {"prefix": "SC-", "codes": ["SC-V", "SC-U"], "attributes": ["Value", "Unit"]}  # Condition
         ]
    elif category_name == "Range Value with Single Condition":
        return [
            {"prefix": "RV-", "codes": ["RV-Vn", "RV-Un"], "attributes": ["Value", "Unit"]}, # Range Min
            {"prefix": "RV-", "codes": ["RV-Vx", "RV-Ux"], "attributes": ["Value", "Unit"]}, # Range Max
            {"prefix": "SC-", "codes": ["SC-V", "SC-U"], "attributes": ["Value", "Unit"]}   # Condition
        ]
    # --- Structures with Range Condition ---
    elif category_name == "Number with Range Condition":
         return [
             {"prefix": "SN-", "codes": ["SN-V", "SN-U"], "attributes": ["Value", "Unit"]}, # Number
             {"prefix": "RC-", "codes": ["RC-Vn", "RC-Un"], "attributes": ["Value", "Unit"]}, # Cond Range Min
             {"prefix": "RC-", "codes": ["RC-Vx", "RC-Ux"], "attributes": ["Value", "Unit"]}  # Cond Range Max
         ]
    elif category_name == "Single Value with Range Condition":
         return [
             {"prefix": "SV-", "codes": ["SV-V", "SV-U"], "attributes": ["Value", "Unit"]}, # Value
             {"prefix": "RC-", "codes": ["RC-Vn", "RC-Un"], "attributes": ["Value", "Unit"]}, # Cond Range Min
             {"prefix": "RC-", "codes": ["RC-Vx", "RC-Ux"], "attributes": ["Value", "Unit"]}  # Cond Range Max
         ]
    elif category_name == "Complex Single with Range Condition": # Added
          return [
              {"prefix": "CX-", "codes": ["CX-V", "CX-U"], "attributes": ["Value", "Unit"]}, # Complex Value
              {"prefix": "RC-", "codes": ["RC-Vn", "RC-Un"], "attributes": ["Value", "Unit"]}, # Cond Range Min
              {"prefix": "RC-", "codes": ["RC-Vx", "RC-Ux"], "attributes": ["Value", "Unit"]}  # Cond Range Max
          ]
    elif category_name == "Range Value with Range Condition":
         return [
             {"prefix": "RV-", "codes": ["RV-Vn", "RV-Un"], "attributes": ["Value", "Unit"]}, # Value Range Min
             {"prefix": "RV-", "codes": ["RV-Vx", "RV-Ux"], "attributes": ["Value", "Unit"]}, # Value Range Max
             {"prefix": "RC-", "codes": ["RC-Vn", "RC-Un"], "attributes": ["Value", "Unit"]}, # Cond Range Min
             {"prefix": "RC-", "codes": ["RC-Vx", "RC-Ux"], "attributes": ["Value", "Unit"]}  # Cond Range Max
         ]
    # --- Structures with Multiple Conditions ---
    # These categories signal that generate_mapping needs dynamic MC- codes.
    # The base structure (Number, SV, RV) is defined first.
    elif category_name == "Number with Multiple Conditions":
         return [{"prefix": "SN-", "codes": ["SN-V", "SN-U"], "attributes": ["Value", "Unit"]}] # Base Number part
    elif category_name == "Single Value with Multiple Conditions":
         return [{"prefix": "SV-", "codes": ["SV-V", "SV-U"], "attributes": ["Value", "Unit"]}] # Base SV part
    elif category_name == "Complex Single with Multiple Conditions": # Added
          return [{"prefix": "CX-", "codes": ["CX-V", "CX-U"], "attributes": ["Value", "Unit"]}] # Base CX part
    elif category_name == "Range Value with Multiple Conditions":
         return [
             {"prefix": "RV-", "codes": ["RV-Vn", "RV-Un"], "attributes": ["Value", "Unit"]}, # Base RV part (Min)
             {"prefix": "RV-", "codes": ["RV-Vx", "RV-Ux"], "attributes": ["Value", "Unit"]}  # Base RV part (Max)
         ]

    # --- Multi Value (Top Level) ---
    # These categories (e.g., "Multiple (2x) Single Value") should be handled by the caller (process_single_key)
    # which splits into chunks. This function should ideally receive the chunk's category.
    # Add a fallback if called directly with a "Multiple..." category.
    elif category_name.startswith("Multiple"):
         st.warning(f"DEBUG: get_code_prefixes_for_category received top-level Multiple category '{category_name}'. Using fallback.")
         # Fallback: return generic codes; caller should prefix M1-, M2- etc.
         return [{"prefix": "MULTI-", "codes": ["MULTI-V", "MULTI-U"], "attributes": ["Value", "Unit"]}]

    # --- Empty / Invalid ---
    elif category_name in ["Empty", "Invalid/Empty Structure", "Classification Error", "Unknown Chunk", "Unknown", "Processing Error"]:
         return [{"prefix": "ERR-", "codes": ["ERR-VAL"], "attributes": ["Value"]}] # Special error code

    # --- Fallback for completely unknown categories ---
    else:
        st.warning(f"Unknown category '{category_name}' in get_code_prefixes_for_category. Using default UNK- codes.")
        return [{"prefix": "UNK-", "codes": ["UNK-V", "UNK-U"], "attributes": ["Value", "Unit"]}]


def fill_mapping_for_part(part_tuple, block_info):
    """
    Fills the mapping dictionary for a single parsed part (value, unit tuple)
    using the code/attribute definitions from block_info.

    Args:
        part_tuple (tuple): (value_string, base_unit_string) from parse_value_unit_identifier.
        block_info (dict): Definition for this block (prefix, codes, attributes).

    Returns:
        dict: A dictionary segment for the final mapping (e.g., {"SV-V": {"value": "10"}, "SV-U": {"value": "V"}}).
    """
    (val_str, base_unit_str) = part_tuple
    result = {}
    codes = block_info.get("codes", [])
    attributes = block_info.get("attributes", [])
    prefix = block_info.get("prefix", "ERR")

    try:
        # Find the indices for 'Value' and 'Unit' attributes
        value_idx = attributes.index("Value") if "Value" in attributes else -1
        unit_idx = attributes.index("Unit") if "Unit" in attributes else -1

        # Assign value if attribute and code exist
        if value_idx != -1 and value_idx < len(codes):
            value_code = codes[value_idx]
            result[value_code] = {"value": val_str}
        elif not codes: # Handle cases like ERR-VAL where only one code exists
             if prefix.startswith("ERR"):
                  result[codes[0]] = {"value": val_str} # Put the raw value in the error code
             else:
                   st.warning(f"Block info for prefix '{prefix}' missing Value code/attribute.")

        # Assign unit if attribute and code exist
        if unit_idx != -1 and unit_idx < len(codes):
            unit_code = codes[unit_idx]
            # Store empty string if base_unit_str is empty, don't store None
            result[unit_code] = {"value": base_unit_str if base_unit_str else ""}
        # Don't warn if unit attribute doesn't exist (e.g. for ERR-VAL)
        # elif "Unit" in attributes:
        #      st.warning(f"Block info for prefix '{prefix}' missing Unit code.")

    except Exception as e:
         st.error(f"Error filling mapping for block {block_info} with part {part_tuple}: {e}")
         # Create generic codes as fallback
         result[f"{prefix}-V_ERR"] = {"value": val_str}
         result[f"{prefix}-U_ERR"] = {"value": base_unit_str}

    return result

def generate_mapping(parsed_parts, category_name):
    """
    Generates the complete code-to-value mapping dictionary for a given value string,
    based on its parsed parts and classification category. Handles dynamic codes
    for multiple conditions.

    Args:
        parsed_parts (list[tuple]): List of (value_str, base_unit_str) tuples from extract_block_texts -> parse_value_unit_identifier.
        category_name (str): The classification of the value string.

    Returns:
        dict: The complete code mapping, e.g., {"RV-Vn": {"value": "10"}, "RV-Un": {"value": "A"}, ...}.
    """
    # Get the base structure definition (codes/attributes for main value, maybe range parts)
    base_blocks = get_code_prefixes_for_category(category_name)
    mapping = {}
    part_idx = 0
    block_idx = 0
    mc_counter = 1 # Counter for dynamically generated Multi-Condition codes (MC-V1, MC-U1, ...)

    # Iterate through the parsed parts (text blocks identified by extract_block_texts)
    while part_idx < len(parsed_parts):
        part_tuple = parsed_parts[part_idx] #(val_str, unit_str)

        # Case 1: We still have predefined blocks from the category definition
        if block_idx < len(base_blocks):
            block_info = base_blocks[block_idx]
            # Fill mapping for this part using the current block definition
            new_map = fill_mapping_for_part(part_tuple, block_info)
            mapping.update(new_map)
            part_idx += 1
            block_idx += 1
        # Case 2: Ran out of predefined blocks, but the category involves "Multiple Conditions"
        elif "Multiple Conditions" in category_name:
            # Dynamically generate codes for this extra condition part
            dynamic_block_info = {
                "prefix": "MC-", # Multi-Condition prefix
                "codes": [f"MC-V{mc_counter}", f"MC-U{mc_counter}"], # Dynamic codes MC-V1, MC-U1, etc.
                "attributes": ["Value", "Unit"]
            }
            new_map = fill_mapping_for_part(part_tuple, dynamic_block_info)
            mapping.update(new_map)
            part_idx += 1
            mc_counter += 1 # Increment for the next potential condition
        # Case 3: Ran out of blocks and not expecting multiple conditions
        else:
            # This suggests extract_block_texts returned more parts than expected by get_code_prefixes_for_category.
            st.warning(f"Warning: More parsed parts ({len(parsed_parts)}) than expected blocks ({len(base_blocks)}) for category '{category_name}'. Ignoring extra part: {part_tuple}")
            part_idx += 1 # Skip this unexpected extra part

    # Handle cases where there were FEWER parts than blocks (e.g., optional condition missing)
    # - The mapping will simply lack entries for the unused blocks, which is generally acceptable.

    return mapping


def get_desired_order():
    """Returns a list defining the preferred order of codes in the output."""
    # Update this list based on the codes defined in get_code_prefixes_for_category
    # Include potential codes for all structures and dynamic MC codes up to a reasonable limit.
    return [
        # Number
        "SN-V", "SN-U",
        # Single Value
        "SV-V", "SV-U",
        # Complex Single
        "CX-V", "CX-U",
        # Range Value (main) - Min then Max
        "RV-Vn", "RV-Un", "RV-Vx", "RV-Ux",
        # Single Condition
        "SC-V", "SC-U",
        # Range Condition - Min then Max
        "RC-Vn", "RC-Un", "RC-Vx", "RC-Ux",
        # Multi-Condition (dynamic) - Add first few expected ones
        "MC-V1", "MC-U1",
        "MC-V2", "MC-U2",
        "MC-V3", "MC-U3",
        "MC-V4", "MC-U4", # Add more if needed
        # MULTI (placeholder for base type in a multiple structure - might not appear directly if handled by caller)
        # "MULTI-V", "MULTI-U", # Comment out if not expected in final output rows
        # Unknown/Error Codes
        "UNK-V", "UNK-U",
        "ERR-VAL",
        # Add error codes from fill_mapping_for_part if needed
        # "ERR-V_ERR", "ERR-U_ERR"
    ]


def display_mapping(mapping_dict, desired_order, category, main_key):
    """
    Formats the generated code-to-value mapping dictionary into a list of
    dictionary rows suitable for creating the output DataFrame.

    Args:
        mapping_dict (dict): The code map generated by generate_mapping.
        desired_order (list): Preferred order of codes.
        category (str): The classification category for this main_key.
        main_key (str): The original input value string (used for the 'Main Key' column).

    Returns:
        list[dict]: A list of rows, each row being a dictionary with keys
                    "Main Key", "Category", "Attribute", "Code", "Value".
    """
    def get_attribute_from_code(code):
        # Determines Attribute ("Value" or "Unit") based on code suffix convention
        if code.endswith("-V") or code.endswith("-Vn") or code.endswith("-Vx") or re.search(r'-V\d+$', code) or code == "ERR-VAL":
             return "Value"
        elif code.endswith("-U") or code.endswith("-Un") or code.endswith("-Ux") or re.search(r'-U\d+$', code):
             return "Unit"
        # Add fallbacks for complex/unknown/multi codes if needed
        elif code.startswith("CX-") and code.endswith("-V"): return "Value"
        elif code.startswith("CX-") and code.endswith("-U"): return "Unit"
        elif code.startswith("UNK-") and code.endswith("-V"): return "Value"
        elif code.startswith("UNK-") and code.endswith("-U"): return "Unit"
        elif code.startswith("MULTI-") and code.endswith("-V"): return "Value"
        elif code.startswith("MULTI-") and code.endswith("-U"): return "Unit"
        return "Info" # Default attribute if not clearly Value or Unit


    output_rows = []
    processed_codes = set()

    # Add rows for codes present in the mapping, following the desired order
    for code in desired_order:
        if code in mapping_dict:
            attr = get_attribute_from_code(code)
            # Ensure value exists and handle potential None/NaN from parsing steps
            value = mapping_dict[code].get("value", "") # Default to empty string
            if pd.isna(value): value = "" # Convert NaN to empty string

            row = {
                "Main Key": main_key, # Original input string
                "Category": category, # Classification name
                "Attribute": attr,    # Value or Unit (or Info)
                "Code": code,         # The specific code (e.g., SV-V, RC-Un)
                "Value": str(value)   # The parsed value/unit string, ensure string type
            }
            output_rows.append(row)
            processed_codes.add(code)

    # Add any codes found in the mapping but not in the desired order (e.g., MC-V5 if desired_order stops at MC-V4)
    extra_codes = sorted([c for c in mapping_dict if c not in processed_codes])
    for code in extra_codes:
        attr = get_attribute_from_code(code)
        value = mapping_dict[code].get("value", "")
        if pd.isna(value): value = ""

        row = {
            "Main Key": main_key,
            "Category": category,
            "Attribute": attr,
            "Code": code,
            "Value": str(value)
        }
        output_rows.append(row)

    return output_rows


# --- Main Processing Function for Fixed Pipeline ---

# MODIFIED process_single_key
def process_single_key(main_key: str, base_units, multipliers_dict):
    """
    Processes a single 'Value' string entry: classifies it, extracts blocks,
    parses them, generates codes, and returns structured output rows.
    Handles multi-value entries by splitting and prefixing codes (M1-, M2-).

    Args:
        main_key (str): The input value string.
        base_units (set): Known base units.
        multipliers_dict (dict): Multiplier map.

    Returns:
        list[dict]: A list of output rows for this main_key.
    """
    main_key_original = main_key # Keep original for output row
    # Apply initial cleaning/fixing (e.g., space before Ohm)
    main_key_clean = fix_exceptions(str(main_key).strip())

    # --- Use detailed classification logic ---
    try:
        (category, _, sub_value_count, final_cond_item_count, _, _, _, _, final_main_item_count) = classify_value_type_detailed(main_key_clean)
        if not category or category in ["Empty", "Invalid/Empty Structure"]:
            category = "Unknown" # Standardize empty/invalid classifications for this pipeline's output
    except Exception as e:
        st.error(f"Error during detailed classification for '{main_key_clean}': {e}")
        category = "Classification Error"
        sub_value_count = 1 # Assume one item for error handling


    # --- Handle Classification Error or Unknown ---
    if category in ["Unknown", "Classification Error"]:
        error_value = "Could not classify structure." if category=="Unknown" else f"Classification Error: {e}"
        return [{
            "Main Key": main_key_original,
            "Category": category,
            "Attribute": "Info", # Use Info attribute for errors
            "Code": "ERR-CL",     # Specific error code for classification
            "Value": error_value
        }]

    # --- Process based on sub-value count ---
    all_output_rows = []
    # Check if category indicates multiple top-level values OR if sub_value_count > 1
    is_multiple = category.startswith("Multiple") or sub_value_count > 1

    if is_multiple:
        # Split the cleaned key into chunks based on comma outside parentheses
        # This aligns with how classify_value_type_detailed counts sub-values
        chunks = split_outside_parens(main_key_clean, [','])
        # Filter empty chunks just in case
        chunks = [chk.strip() for chk in chunks if chk.strip()]

        if len(chunks) != sub_value_count and sub_value_count > 0 :
            # If the split count doesn't match the classification count, something is complex.
            # Trust the split_outside_parens result for processing, but warn.
            st.warning(f"Sub-value count mismatch for '{main_key_clean}'. Classified count: {sub_value_count}, Comma-split chunks: {len(chunks)}. Processing {len(chunks)} chunks.")
            # Fallback if sub_value_count was 0 but category was 'Multiple': use chunks length
            effective_sub_value_count = len(chunks)
        elif sub_value_count == 0 and chunks: # Handle case where classification might miss but split finds items
             effective_sub_value_count = len(chunks)
             st.warning(f"Classification yielded 0 sub-values for '{main_key_clean}' but split found {len(chunks)}. Processing chunks.")
        else:
             effective_sub_value_count = sub_value_count


        if effective_sub_value_count <= 0: # Safety check
             st.error(f"Failed to find processable chunks for multi-value key: '{main_key_original}'")
             return [{ "Main Key": main_key_original, "Category": "Processing Error", "Attribute": "Info", "Code": "ERR-CHUNK", "Value": "No chunks found for multi-value key."}]


        for idx, chunk in enumerate(chunks):
             chunk_prefix = f"M{idx+1}-" # Prefix like M1-, M2-

             try:
                 # Re-classify the *chunk* to get its specific structure (e.g., Single Value, Range Value)
                 # This is crucial because the overall category might be "Multiple Mixed"
                 (chunk_cat, _, chunk_sv_count, _, _, _, _, _, _) = classify_value_type_detailed(chunk)
                 if not chunk_cat or chunk_cat in ["Empty", "Invalid/Empty Structure"]:
                     chunk_cat = "Unknown Chunk"

                 if chunk_cat != "Unknown Chunk":
                     # Process this single chunk
                     block_texts = extract_block_texts(chunk, chunk_cat) # Use chunk's category
                     parsed_parts = []
                     for bt in block_texts:
                         part_val, part_unit = parse_value_unit_identifier(bt, base_units, multipliers_dict)
                         parsed_parts.append((part_val, part_unit))

                     # Generate mapping for the chunk
                     code_map_chunk = generate_mapping(parsed_parts, chunk_cat)

                     # Apply the M{idx+1}- prefix to all codes in the chunk's map
                     prefixed_code_map = {f"{chunk_prefix}{code}": val for code, val in code_map_chunk.items()}

                     # Get desired order, apply prefix to standard codes for sorting chunk output
                     desired_order_standard = get_desired_order()
                     desired_order_prefixed = [f"{chunk_prefix}{code}" for code in desired_order_standard]

                     # Create rows for this chunk, using the chunk's category for the 'Category' column
                     chunk_rows = display_mapping(prefixed_code_map, desired_order_prefixed, chunk_cat, main_key_original) # Pass original key
                     all_output_rows.extend(chunk_rows)
                 else:
                     # Handle error for this specific chunk
                      error_row = {
                         "Main Key": main_key_original, "Category": f"Error in Chunk {idx+1}",
                         "Attribute": "Info", "Code": f"{chunk_prefix}ERR-CHK", # Prefixed error code
                         "Value": f"Could not classify chunk: {chunk}"
                      }
                      all_output_rows.append(error_row)

             except Exception as e:
                 st.error(f"Error processing chunk '{chunk}' from '{main_key_original}': {e}")
                 error_row = {
                     "Main Key": main_key_original, "Category": f"Error in Chunk {idx+1}",
                     "Attribute": "Info", "Code": f"{chunk_prefix}ERR-PROC", # Prefixed error code
                     "Value": f"Processing error: {e}"
                 }
                 all_output_rows.append(error_row)
        return all_output_rows

    else:
        # --- Single sub-value case ---
        try:
            # Use the overall category determined at the start
            block_texts = extract_block_texts(main_key_clean, category)
            parsed_parts = []
            for bt in block_texts:
                part_val, part_unit = parse_value_unit_identifier(bt, base_units, multipliers_dict)
                parsed_parts.append((part_val, part_unit))

            # Generate mapping based on the overall category
            code_map = generate_mapping(parsed_parts, category)
            # Get desired order for standard codes
            desired_order_standard = get_desired_order()
            # Create rows using the overall category
            single_rows = display_mapping(code_map, desired_order_standard, category, main_key_original) # Use original key
            return single_rows
        except Exception as e:
             st.error(f"Error processing single key '{main_key_original}': {e}")
             return [{
                 "Main Key": main_key_original, "Category": "Processing Error",
                 "Attribute": "Info", "Code": "ERR-PROC",
                 "Value": f"Processing error: {e}"
             }]


# MODIFIED process_fixed_pipeline_bytes
def process_fixed_pipeline_bytes(file_bytes: bytes, mapping_file_path: str):
    """
    Runs the first pipeline (fixed processing) on the input Excel file bytes.
    Reads mapping configuration from the specified local path.
    Produces processed data as structured rows with codes.

    Args:
        file_bytes (bytes): The content of the uploaded Excel file.
        mapping_file_path (str): Path to the local 'mapping.xlsx' file.

    Returns:
        pd.DataFrame or None: DataFrame containing the processed rows, or None on failure.
    """
    print("Running Fixed Processing Pipeline...")
    st.write("DEBUG: Starting Fixed Processing Pipeline...")
    all_processed_rows = []
    # output_filename_temp = 'temp_processed_output.xlsx' # Not saving intermediate file now

    try:
        # Read mapping file from the provided disk path
        # This assumes the file exists and is valid. Error handling inside read_mapping_file.
        base_units_from_file, multipliers_dict = read_mapping_file(mapping_file_path)
        # Combine with any potential built-in base units
        combined_base_units = LOCAL_BASE_UNITS.union(base_units_from_file)
        st.write(f"DEBUG: Using {len(combined_base_units)} base units for fixed pipeline.")

        # Read input excel from bytes
        try:
             xls = pd.ExcelFile(io.BytesIO(file_bytes))
             all_sheets = xls.sheet_names
             print(f"Found sheets: {all_sheets}")
             st.write(f"DEBUG: Input sheets: {all_sheets}")
        except Exception as e:
             st.error(f"Error reading input Excel file: {e}. Is the file format correct?")
             return None # Cannot proceed without reading input


        total_rows_processed = 0
        chunk_size = 500 # Process in chunks for memory efficiency

        # Process each sheet in the input file
        for sheet_name in all_sheets:
            print(f"Processing sheet: '{sheet_name}'")
            st.write(f"DEBUG: Processing sheet: '{sheet_name}'")
            try:
                # Read the current sheet
                sheet_df = pd.read_excel(xls, sheet_name=sheet_name)

                # --- Column Check: Ensure 'Value' column exists ---
                # Use case-insensitive check? Allow user to specify column?
                # For now, strict check for 'Value'.
                value_col_name = 'Value' # Hardcoded for now
                if value_col_name not in sheet_df.columns:
                    st.warning(f"Sheet '{sheet_name}' skipped: Missing required column '{value_col_name}'.")
                    continue # Skip this sheet

                # Extract original columns to merge back later
                original_cols = sheet_df.columns.tolist()
                # Ensure 'Value' is treated as string for processing
                sheet_df[value_col_name] = sheet_df[value_col_name].astype(str)

                sheet_rows_processed = 0
                # Iterate through the DataFrame in chunks
                for i in range(0, len(sheet_df), chunk_size):
                    chunk_df = sheet_df.iloc[i:i + chunk_size]

                    # Process each row within the chunk
                    for row_index, row_series in chunk_df.iterrows():
                        main_key = row_series.get(value_col_name, '').strip()

                        # Skip rows with empty 'Value' after stripping
                        if not main_key:
                            continue

                        # Run the core processing logic for the 'Value' string
                        # This returns a list of dicts, one for each generated code/attribute row
                        result_rows_for_key = process_single_key(main_key, combined_base_units, multipliers_dict)

                        # Add original row data and sheet info to each result row
                        original_row_data = row_series.to_dict()
                        for r_dict in result_rows_for_key:
                            # Start with original data, overwrite/add processed results
                            # Ensure keys from r_dict ('Main Key', 'Category', 'Attribute', 'Code', 'Value')
                            # overwrite any same-named keys from original_row_data if necessary.
                            # (Here, 'Value' from r_dict is the parsed part, 'Value' in original is the input key)
                            final_row = original_row_data.copy()
                            final_row.update(r_dict) # Processed data overwrites/adds columns
                            final_row["Sheet"] = sheet_name # Add sheet name identifier
                            all_processed_rows.append(final_row)

                        sheet_rows_processed += 1 # Count processed input rows

                    # Optional: Trigger garbage collection after processing each chunk
                    gc.collect()

                total_rows_processed += sheet_rows_processed
                st.write(f"DEBUG: Processed {sheet_rows_processed} non-empty rows from sheet '{sheet_name}'.")

            except Exception as e:
                # Catch errors during sheet processing (reading, iterating, single key processing)
                st.error(f"Error processing sheet '{sheet_name}' at row index approx {row_index if 'row_index' in locals() else 'N/A'}: {e}")
                # Optionally log traceback
                # st.error(traceback.format_exc())
                continue # Skip to next sheet on error

        st.write(f"DEBUG: Fixed Processing Pipeline finished. Total input rows processed: {total_rows_processed}. Output rows generated: {len(all_processed_rows)}")

        # Check if any rows were generated
        if not all_processed_rows:
             st.warning("Fixed processing generated no output rows. Check input data and mapping.")
             # Return an empty DataFrame instead of None, perhaps?
             return pd.DataFrame()

        # Create the final DataFrame from the list of all processed rows
        processed_df = pd.DataFrame(all_processed_rows)

        # Optional: Reorder columns? The order depends on original cols + added cols.
        # Handled later in combine_results.

        # Return the DataFrame
        return processed_df

    # Catch errors related to mapping file loading or initial setup
    except FileNotFoundError as e:
         st.error(f"Pipeline Error: {e}. Cannot start fixed processing.")
         # st.stop() # Stop might be too abrupt in a module; return None.
         return None
    except ValueError as e:
         st.error(f"Pipeline Error: {e}. Cannot start fixed processing.")
         return None
    except Exception as e:
        # Catch any other unexpected errors during the pipeline execution
        st.error(f"An unexpected error occurred in the fixed pipeline: {e}")
        st.error(traceback.format_exc()) # Provide traceback for debugging
        return None
