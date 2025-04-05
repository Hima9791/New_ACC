#############################################
# MODULE: ANALYSIS HELPERS
# Purpose: Contains core, reusable functions for parsing,
#          classification, unit/numeric extraction.
#          Used by both fixed and detailed pipelines.
#############################################

import re
import pandas as pd
import streamlit as st # For warnings/debug, consider replacing with logging

# Import constants from mapping_utils
# Assumes mapping_utils.py is in the same directory or accessible via Python path
try:
    from mapping_utils import MULTIPLIER_MAPPING
except ImportError:
    # Fallback if run in a context where mapping_utils isn't directly importable
    # This is less ideal but provides a default.
    st.error("Could not import MULTIPLIER_MAPPING from mapping_utils. Using default empty map.")
    MULTIPLIER_MAPPING = {}
def split_outside_parens_preserve(text, delimiters):
    """
    Splits text by the given delimiters, ignoring delimiters inside parentheses,
    and always preserving the delimiter tokens.
    
    Args:
        text (str): Text to split.
        delimiters (list[str]): List of delimiter strings.
    
    Returns:
        list[str]: List of tokens with delimiters included.
    """
    text = str(text)
    tokens = []
    current = ""
    i = 0
    depth = 0
    sorted_delims = sorted(delimiters, key=len, reverse=True)

    while i < len(text):
        char = text[i]
        if char == '(':
            depth += 1
            current += char
            i += 1
        elif char == ')':
            depth = max(0, depth - 1)
            current += char
            i += 1
        elif depth == 0:
            matched_delim = None
            for delim in sorted_delims:
                if i + len(delim) <= len(text) and text[i:i+len(delim)] == delim:
                    matched_delim = delim
                    break

            if matched_delim:
                if current.strip():
                    tokens.append(current.strip())
                tokens.append(matched_delim)  # Always add the delimiter token
                current = ""
                i += len(matched_delim)
            else:
                current += char
                i += 1
        else:
            current += char
            i += 1

    if current.strip():
        tokens.append(current.strip())

    return tokens

# --- Utility functions (Mostly from original Section 3 & 4 helpers) ---

def extract_numeric_and_unit_analysis(token, base_units, multipliers_dict):
    """
    Analyzes a token to extract numeric value, multiplier symbol, base unit,
    and normalized value. Handles numbers, units, and combinations.
    """
    token = str(token).strip()  # Ensure string and strip whitespace
    if not token:
        return None, None, None, None, False

    # Regex to capture optional sign, numeric part, and the rest (unit, etc.)
    pattern = re.compile(r'^(?P<numeric>[+\-±]?\d*(?:\.\d+)?(?:[eE][+\-]?\d+)?)(?P<rest>.*)$')
    m = pattern.match(token)

    # Handle case where token might be only a unit
    if not m or not m.group("numeric"):
        if token in base_units:
            return None, "1", token, None, False
        else:
            sorted_prefixes = sorted(multipliers_dict.keys(), key=len, reverse=True)
            for prefix in sorted_prefixes:
                if token.startswith(prefix):
                    possible_base = token[len(prefix):].strip()
                    if possible_base in base_units:
                        return None, prefix, possible_base, None, False
        return None, None, None, None, True

    numeric_str_raw = m.group("numeric")
    rest = m.group("rest").strip()

    # NEW: Instead of immediately removing parentheses, first check if the rest starts
    # with a known multiplier prefix and that the leftover is exactly a recognized unit.
    prefix_found = False
    for prefix in sorted(multipliers_dict.keys(), key=len, reverse=True):
        if rest.startswith(prefix):
            possible_base = rest[len(prefix):].strip()
            if possible_base in base_units:
                # Keep the original rest (with the prefix and parentheses intact)
                cleaned_rest = rest
                prefix_found = True
                break
    if not prefix_found:
        if rest in base_units:
            cleaned_rest = rest
        else:
            cleaned_rest = remove_parentheses_detailed(rest)

    try:
        numeric_val = float(numeric_str_raw.replace('±',''))
    except ValueError:
        return None, None, None, None, True

    # Case: Only a number was found.
    if not cleaned_rest:
        return numeric_val, "1", None, numeric_val, False

    # Try to match a known prefix + base unit
    multiplier_symbol = None
    base_unit = None
    multiplier_factor = 1.0
    found_unit_structure = False

    sorted_prefixes = sorted(multipliers_dict.keys(), key=len, reverse=True)
    for prefix in sorted_prefixes:
        if cleaned_rest.startswith(prefix):
            possible_base = cleaned_rest[len(prefix):].strip()
            if possible_base in base_units:
                multiplier_symbol = prefix
                base_unit = possible_base
                multiplier_factor = multipliers_dict[prefix]
                found_unit_structure = True
                break

    if not found_unit_structure:
        # If no prefix was detected, check if cleaned_rest itself is a base unit.
        if cleaned_rest in base_units:
            multiplier_symbol = "1"
            base_unit = cleaned_rest
            found_unit_structure = True
        else:
            return numeric_val, None, None, None, True

    normalized_value = numeric_val * multiplier_factor
    final_multiplier_symbol = multiplier_symbol if multiplier_symbol is not None else "1"
    return numeric_val, final_multiplier_symbol, base_unit, normalized_value, False


def remove_parentheses_detailed(text: str) -> str:
    """Removes content within the outermost parentheses."""
    return re.sub(r'\([^()]*\)', '', str(text)) # Ensure input is string

def extract_identifiers_detailed(text: str):
    """Finds all content within non-nested parentheses."""
    return re.findall(r'\(([^()]*)\)', str(text)) # Finds content inside (...)


def split_outside_parens(text, delimiters):
    text = str(text)  # Ensure string input
    tokens = []
    current = ""
    i = 0
    depth = 0
    # Sort delimiters by length descending to match longest first
    sorted_delims = sorted(delimiters, key=len, reverse=True)

    while i < len(text):
        char = text[i]
        if char == '(':
            depth += 1
            current += char
            i += 1
        elif char == ')':
            depth = max(0, depth - 1)
            current += char
            i += 1
        elif depth == 0:
            matched_delim = None
            for delim in sorted_delims:
                if i + len(delim) <= len(text) and text[i:i+len(delim)] == delim:
                    matched_delim = delim
                    break

            if matched_delim:
                if current.strip():
                    tokens.append(current.strip())
                # Do not append the delimiter token; use it only as a boundary.
                current = ""
                i += len(matched_delim)
            else:
                current += char
                i += 1
        else:
            current += char
            i += 1

    if current.strip():
        tokens.append(current.strip())

    return [token for token in tokens if token]



def extract_numeric_info(part_text, base_units, multipliers_dict):
    """
    Extracts numeric information (values, multipliers, units, normalized)
    from a string potentially containing single, range, or multiple values.

    Args:
        part_text (str): The text segment to analyze (e.g., "10k to 20k Ohm", "5A", "1, 2, 3").
        base_units (set): Set of known base units.
        multipliers_dict (dict): Dictionary of multipliers.

    Returns:
        dict: Dictionary containing lists of extracted info and the detected type ('single', 'range', 'multiple', 'none').
              Keys: "numeric_values", "multipliers", "base_units", "normalized_values", "error_flags", "type".
    """
    # First remove content within parentheses for analysis
    text = remove_parentheses_detailed(part_text).strip()

    if not text:
        return {
            "numeric_values": [], "multipliers": [],
            "base_units": [], "normalized_values": [],
            "error_flags": [], "type": "none"
        }

    # Determine structure: range, multiple values, or single
    # Use robust splitting for ' to ' to avoid issues with substrings
    # Regex checks for ' to ' surrounded by whitespace or start/end of string parts
    is_range = bool(re.search(r'(?:^|\s)to(?:\s|$)', text, re.IGNORECASE)) and len(split_outside_parens(text, [','])) == 1 # Avoid "A, B to C" being range

    # Split by comma first if not a simple range
    tokens = split_outside_parens(text, [','])

    if len(tokens) > 1:
        info_type = "multiple"
        # Further split parts containing ' to ' if necessary? No, treat "A, B to C" as multiple.
    elif len(tokens) == 1:
         # Now check the single token for ' to ' range
         range_parts = split_outside_parens(tokens[0], [' to ']) # Crude split for ' to '
         # Refine range check: needs values on both sides?
         if len(range_parts) > 1 and re.search(r'\s+to\s+', tokens[0], re.IGNORECASE):
              info_type = "range"
              tokens = range_parts # Use the parts split by ' to '
         else:
              info_type = "single"
              tokens = [tokens[0]] # Keep the single token
    else: # No tokens found after splitting (e.g., empty string after paren removal)
        info_type = "none"
        tokens = []


    # Initialize lists to store results for each token
    numeric_values = []
    multipliers = []
    base_units_list = []
    normalized_values = []
    error_flags = []

    # Process each token found
    for token in tokens:
        token_strip = token.strip()
        if not token_strip: continue # Skip empty tokens resulting from splits

        num_val, multiplier_symbol, base_unit, norm_val, err_flag = extract_numeric_and_unit_analysis(
            token_strip, base_units, multipliers_dict
        )

        numeric_values.append(num_val)
        # Use "1" if no multiplier symbol was identified but parsing was ok
        multipliers.append(multiplier_symbol if multiplier_symbol else ("1" if not err_flag and num_val is not None else None))
         # Base unit might be None if only number, or if error
        base_units_list.append(base_unit if base_unit else None)
        normalized_values.append(norm_val)
        error_flags.append(err_flag)

    # Return dictionary of results
    return {
        "numeric_values": numeric_values,
        "multipliers": multipliers,
        "base_units": base_units_list,
        "normalized_values": normalized_values,
        "error_flags": error_flags,
        "type": info_type # Indicates if it was parsed as 'single', 'range', or 'multiple' parts
    }

def safe_str(item, placeholder="None"):
    """Converts item to string, using placeholder if None."""
    return str(item) if item is not None else placeholder

def extract_numeric_info_for_value(raw_value, base_units, multipliers_dict):
    """
    Extracts numeric information for a potentially complex value string,
    splitting it into main and condition parts based on '@'.

    Args:
        raw_value (str): The input value string (e.g., "10A @ 5V", "50 Ohm").
        base_units (set): Set of known base units.
        multipliers_dict (dict): Dictionary of multipliers.

    Returns:
        dict: Dictionary containing numeric info for main and condition parts.
              Keys like "main_numeric", "condition_base_units", "normalized_main", etc.
    """
    # This function processes ONE logical value string which might contain '@' internally.
    # It assumes the input `raw_value` represents one entry (potentially complex).
    # Splitting multiple entries (like "10A, 20A") should happen *before* calling this.

    raw_value = str(raw_value).strip() # Ensure string type
    main_part = raw_value
    cond_part = ""

    # Split only on the first '@' found outside parentheses
    at_split = split_outside_parens(raw_value, ['@'])
    if len(at_split) > 1:
        main_part = at_split[0].strip()
        cond_part = "@".join(at_split[1:]).strip() # Rejoin if multiple '@' were present (unlikely?)
    elif len(at_split) == 1:
         main_part = at_split[0].strip() # No '@' found outside parens
         # Check if '@' exists inside parentheses (should be ignored by split_outside_parens)


    # Process the main part and the condition part separately
    main_info = extract_numeric_info(main_part, base_units, multipliers_dict)
    # Process cond_part only if it's not empty
    cond_info = extract_numeric_info(cond_part, base_units, multipliers_dict) if cond_part else {
            "numeric_values": [], "multipliers": [],
            "base_units": [], "normalized_values": [],
            "error_flags": [], "type": "none"
        }

    # Combine results. Note these lists contain results for *all* tokens within main/condition parts
    # e.g., for "10 to 20A @ 5V", main_info will have two entries, cond_info one entry.
    return {
        "main_numeric": main_info["numeric_values"],
        "main_multipliers": main_info["multipliers"],
        "main_base_units": main_info["base_units"],
        "normalized_main": main_info["normalized_values"],
        "main_errors": main_info["error_flags"],
        "main_type": main_info["type"], # Add type ('single', 'range', 'multiple')

        "condition_numeric": cond_info["numeric_values"],
        "condition_multipliers": cond_info["multipliers"],
        "condition_base_units": cond_info["base_units"],
        "normalized_condition": cond_info["normalized_values"],
        "condition_errors": cond_info["error_flags"],
        "condition_type": cond_info["type"], # Add type
    }

# --- change1 seems to have been incorporated into process_unit_token_no_paren ---
def process_unit_token_no_paren(token, base_units, multipliers_dict):
    token = str(token).strip()

    # If starts with '$' → handle as normalized token
    if token.startswith('$'):
        after_dollar = token[1:]  # remove the '$'
        has_space = after_dollar.startswith(" ")
        stripped = after_dollar.strip()

        if stripped == "":
            return "$ " if has_space else "$"

        if stripped in base_units:
            return "$ " + stripped if has_space else "$" + stripped

        sorted_prefixes = sorted(multipliers_dict.keys(), key=len, reverse=True)
        for prefix in sorted_prefixes:
            if stripped.startswith(prefix):
                possible_base = stripped[len(prefix):]
                if possible_base in base_units:
                    return "$ " + possible_base if has_space else "$" + possible_base

        # 👇 Preserve your original error comment
        return f"Error: Undefined unit '{stripped}' (no recognized prefix)"

    # Otherwise, token does NOT start with '$' — do NOT add '$'
    stripped_token = token.strip()
    if stripped_token in base_units:
        return stripped_token  # no '$' added

    sorted_prefixes = sorted(multipliers_dict.keys(), key=len, reverse=True)
    for prefix in sorted_prefixes:
        if stripped_token.startswith(prefix):
            possible_base = stripped_token[len(prefix):]
            if possible_base in base_units:
                return possible_base  # again, no '$'

    return f"Error: Undefined unit '{stripped_token}' (no recognized prefix)"




def analyze_unit_part(part_text, base_units, multipliers_dict):
    """
    Analyzes the unit(s) present in a text segment (which might be single, range, or multiple).
    Identifies distinct base units and checks for consistency.

    Args:
        part_text (str): The text segment (e.g., "10k to 20k Ohm", "5A", "1V, 2V").
        base_units (set): Set of known base units.
        multipliers_dict (dict): Dictionary of multipliers.

    Returns:
        dict: Contains lists/sets of units, consistency flag, count, and type.
              Keys: "units", "distinct_units", "is_consistent", "count", "type".
    """
    # Remove parentheses content first
    text = remove_parentheses_detailed(part_text).strip()
    if not text:
        return {
            "units": [], "distinct_units": set(),
            "is_consistent": True, "count": 0,
            "type": "none"
        }

    # Determine structure and split into tokens using the same logic as extract_numeric_info
    is_range = bool(re.search(r'(?:^|\s)to(?:\s|$)', text, re.IGNORECASE)) and len(split_outside_parens(text, [','])) == 1
    tokens = split_outside_parens(text, [','])

    if len(tokens) > 1:
        part_type = "multiple"
    elif len(tokens) == 1:
         range_parts = split_outside_parens(tokens[0], [' to '])
         if len(range_parts) > 1 and re.search(r'\s+to\s+', tokens[0], re.IGNORECASE):
              part_type = "range"
              tokens = range_parts
         else:
              part_type = "single"
              tokens = [tokens[0]]
    else:
        part_type = "none"
        tokens = []

    units = []
    for token in tokens:
        token_strip = token.strip()
        if not token_strip: continue

        # Extract only the unit part from the token (e.g., "10kOhm" -> "Ohm")
        _, _, base_unit, _, err_flag = extract_numeric_and_unit_analysis(
            token_strip, base_units, multipliers_dict
        )

        if not err_flag and base_unit:
            units.append(base_unit)
        elif token_strip in base_units: # Handle cases like just "V"
             units.append(token_strip)
        else:
             # If no unit was resolved by extraction, add None or placeholder
             units.append(None) # Represent absence of recognized unit

    # Filter out None before creating distinct set and checking consistency
    valid_units = [u for u in units if u is not None]
    distinct_units = set(valid_units)
    # Consistency means 0 or 1 distinct *valid* unit found
    is_consistent = (len(distinct_units) <= 1)
    count = len(tokens) # Count based on number of tokens processed

    return {
        "units": units, # List of resolved base units (or None) for each token
        "distinct_units": distinct_units, # Set of unique *valid* base units found
        "is_consistent": is_consistent, # True if zero or one distinct valid base unit
        "count": count, # Number of tokens processed
        "type": part_type # How the part was structured
    }


def analyze_value_units(raw_value, base_units, multipliers_dict):
    """
    Analyzes units in a potentially complex value string (main @ condition).

    Args:
        raw_value (str): Input value string.
        base_units (set): Known base units.
        multipliers_dict (dict): Multiplier map.

    Returns:
        dict: Aggregated unit analysis results for main and condition parts.
    """
    # Similar structure to extract_numeric_info_for_value
    raw_value = str(raw_value).strip()
    main_part = raw_value
    cond_part = ""

    # Split by '@' outside parentheses
    at_split = split_outside_parens(raw_value, ['@'])
    if len(at_split) > 1:
        main_part = at_split[0].strip()
        cond_part = "@".join(at_split[1:]).strip()
    elif len(at_split) == 1:
        main_part = at_split[0].strip()

    # Analyze units in main and condition parts
    main_analysis = analyze_unit_part(main_part, base_units, multipliers_dict)
    cond_analysis = analyze_unit_part(cond_part, base_units, multipliers_dict) if cond_part else {
            "units": [], "distinct_units": set(),
            "is_consistent": True, "count": 0,
            "type": "none"
        }


    # Combine unit information
    all_main_units = main_analysis["units"] # List including None
    all_condition_units = cond_analysis["units"] # List including None

    main_distinct = main_analysis["distinct_units"] # Set excluding None
    condition_distinct = cond_analysis["distinct_units"] # Set excluding None

    main_consistent = main_analysis["is_consistent"]
    condition_consistent = cond_analysis["is_consistent"]

    # Overall consistency considers *distinct valid* units across both parts
    all_distinct_units = main_distinct.union(condition_distinct)
    # Overall is consistent if 0 or 1 distinct valid unit is found across all parts
    overall_consistent = (len(all_distinct_units) <= 1)


    return {
        "main_units": all_main_units,
        "main_distinct_units": main_distinct,
        "main_units_consistent": main_consistent,
        "main_unit_count": main_analysis["count"], # Count of tokens in main part
        # Keep sub_analysis if needed for detailed debugging, maybe as string
        # "main_sub_analysis": str(main_analysis),

        "condition_units": all_condition_units,
        "condition_distinct_units": condition_distinct,
        "condition_units_consistent": condition_consistent,
        "condition_unit_count": cond_analysis["count"], # Count of tokens in condition part
        # "condition_sub_analysis": str(cond_analysis),

        "all_distinct_units": all_distinct_units, # Set of all unique valid units found
        "overall_consistent": overall_consistent # True if <= 1 unique valid unit across main and condition
    }


def process_unit_token(token, base_units, multipliers_dict):
    pattern = re.compile(
        r'^(?P<lead>\s*)'
        r'(?P<numeric>[+\-±]?\d*(?:\.\d+)?)'
        r'(?P<space1>\s*)'
        r'(?P<unit>.*?)(?P<space2>\s*)'
        r'(?P<paren>\([^)]*\))?'
        r'(?P<trail>\s*)$'
    )
    m = pattern.match(token)
    if not m:
        return token
    lead = m.group('lead') or ""
    numeric = m.group('numeric') or ""
    space1 = m.group('space1') or ""
    unit_part = m.group('unit') or ""
    space2 = m.group('space2') or ""
    paren = m.group('paren') or ""
    trail = m.group('trail') or ""
    core = unit_part.strip()
    processed_core = process_unit_token_no_paren(core, base_units, multipliers_dict)
    # For 'ohm' or other special cases, you could do more logic here if needed.
    return f"{lead}{numeric}{space1}{processed_core}{space2}{paren}{trail}"


def resolve_compound_unit(normalized_unit, base_units, multipliers_dict):
    # Use the local helper that preserves delimiters.
    tokens = split_outside_parens_preserve(normalized_unit, ["to", ",", "@"])
    resolved_parts = []
    for part in tokens:
        if part in ["to", ",", "@"]:
            resolved_parts.append(part)
        elif part.strip() == "":
            continue
        else:
            resolved_parts.append(process_unit_token(part, base_units, multipliers_dict))
    return " ".join(resolved_parts)





def count_main_items(main_str: str) -> int:
    main_str = main_str.strip()
    if not main_str:
        return 0
    if " to " in main_str:
        return 1
    if "," in main_str:
        return len([s for s in main_str.split(',') if s.strip()])
    return 1


def count_conditions(cond_str: str) -> int:
    cond_str = cond_str.strip()
    if not cond_str:
        return 0
    parts = [p.strip() for p in cond_str.split(',') if p.strip()]
    return len(parts)

    # Conditions are typically comma-separated clauses.
    # Each clause might be a single value or a range.
    comma_parts = split_outside_parens(cond_str, [','])
    count = 0
    for part in comma_parts:
        part_strip = part.strip()
        if not part_strip: continue # Skip empty parts
        count += 1

    # If no commas, check if the string is non-empty -> 1 condition clause
    if count == 0 and cond_str:
        return 1

    return count


def classify_condition(cond_str: str) -> str:
    cond_str = cond_str.strip()
    if not cond_str:
        return ""
    if "," in cond_str:
        return "Multiple Conditions"
    if " to " in cond_str:
        return "Range Condition"
    return "Single Condition"


def classify_main(main_str: str) -> str:
    main_str = main_str.strip()
    if not main_str:
        return ""
    if "," in main_str:
        return "Multi Value"
    if " to " in main_str:
        return "Range Value"
    if re.search(r'[a-zA-Zµ°%]', main_str):
        return "Single Value"
    else:
        return "Number"


def classify_sub_value(subval: str):
    """
    Classifies a single sub-value string, which might contain main @ condition.

    Args:
        subval (str): The sub-value string (e.g., one item from a comma-split list).

    Returns:
        tuple: (classification_str, has_range_main, has_multi_main,
                has_range_cond, has_multi_cond, cond_item_count, main_item_count)
    """
    subval = str(subval).strip()
    if not subval:
         return ("Empty", False, False, False, False, 0, 0)

    # Split into main and condition parts based on '@' outside parentheses
    main_part = subval
    cond_part = ""
    at_split = split_outside_parens(subval, ['@'])
    if len(at_split) > 1:
        main_part = at_split[0].strip()
        cond_part = "@".join(at_split[1:]).strip()
    elif len(at_split) == 1:
        main_part = at_split[0].strip()

    # Classify each part
    main_class = classify_main(main_part)
    cond_class = classify_condition(cond_part)

    # Determine characteristics based on classification strings
    has_range_in_main = ("Range Value" in main_class) # Check if "Range" is part of the main classification
    has_multi_value_in_main = (main_class == "Multi Value") # Specific check for Multi Value type

    has_range_in_condition = ("Range Condition" in cond_class) # Check if "Range" is part of the condition classification
    has_multiple_conditions = (cond_class == "Multiple Conditions") # Specific check

    # Count items/conditions
    # Note: count_main_items/count_conditions operate on the text after parenthesis removal
    main_item_count = count_main_items(main_part)
    cond_item_count = count_conditions(cond_part) # Number of condition clauses

    # Combine classification strings for the final label
    if main_class and cond_class:
        classification = f"{main_class} with {cond_class}"
    elif main_class:
        classification = main_class # No condition part or condition was empty
    elif cond_class:
         # This case (condition but no main) seems unlikely for valid data
         classification = f"Condition Only: {cond_class}"
    else:
        # Neither part could be classified (e.g., input was just punctuation after cleaning?)
        classification = "Invalid/Empty Structure"

    return (classification,
            has_range_in_main,
            has_multi_value_in_main,
            has_range_in_condition,
            has_multiple_conditions,
            cond_item_count,
            main_item_count)

def classify_value_type_detailed(raw_value: str):
    found_parens = extract_identifiers_detailed(raw_value)
    identifiers = ', '.join(s.strip('()') for s in found_parens)
    clean_value = remove_parentheses_detailed(raw_value).strip()
    at_count = clean_value.count('@')
    if at_count == 1:
        subvals = [clean_value]
    else:
        subvals = [v.strip() for v in clean_value.split(',') if v.strip()]
    sub_value_count = len(subvals)
    if sub_value_count == 0:
        return ("", identifiers, 0, 0, False, False, False, False, 0)
    sub_classifications = []
    sub_range_in_main = []
    sub_multi_in_main = []
    sub_range_in_condition = []
    sub_multi_cond = []
    sub_cond_counts = []
    sub_main_item_counts = []
    for sv in subvals:
        (cls,
         has_range_m,
         has_multi_m,
         has_range_c,
         has_multi_c,
         cond_count,
         main_item_count) = classify_sub_value(sv)
        sub_classifications.append(cls)
        sub_range_in_main.append(has_range_m)
        sub_multi_in_main.append(has_multi_m)
        sub_range_in_condition.append(has_range_c)
        sub_multi_cond.append(has_multi_c)
        sub_cond_counts.append(cond_count)
        sub_main_item_counts.append(main_item_count)
    if sub_value_count == 1:
        final_class = sub_classifications[0]
    else:
        unique_classes = set(sub_classifications)
        if len(unique_classes) == 1:
            final_class = "Multiple " + next(iter(unique_classes))
        else:
            final_class = "Multiple Mixed Classification"
    unique_cond_counts = set(sub_cond_counts)
    if len(unique_cond_counts) == 1:
        final_cond_item_count = unique_cond_counts.pop()
    else:
        final_cond_item_count = "Mixed"
    unique_main_item_counts = set(sub_main_item_counts)
    if len(unique_main_item_counts) == 1:
        final_main_item_count = unique_main_item_counts.pop()
    else:
        final_main_item_count = "Mixed"
    has_range_in_main = any(sub_range_in_main)
    has_multi_value_in_main = any(sub_multi_in_main)
    has_range_in_condition = any(sub_range_in_condition)
    has_multiple_conditions = any(sub_multi_cond)
    # Additional logic for "Multiple Mixed Classification"
    if final_class == "Multiple Mixed Classification":
        freq_map = {}
        for cls in sub_classifications:
            freq_map[cls] = freq_map.get(cls, 0) + 1
        if len(freq_map) == 2:
            # Example scenario to refine if needed
            pass
    return (final_class,
            identifiers,
            sub_value_count,
            final_cond_item_count,
            has_range_in_main,
            has_multi_value_in_main,
            has_range_in_condition,
            has_multiple_conditions,
            final_main_item_count)


def fix_exceptions(s):
    """Applies specific string formatting fixes, like spacing around units."""
    s = str(s) # Ensure string
    # Add space before Ohm if preceded by a digit or known prefix, but not already spaced
    # Look for (digit OR prefix) immediately followed by Ohm (case-insensitive)
    ohm_pattern = r'([0-9' + "".join(re.escape(p) for p in MULTIPLIER_MAPPING.keys()) + r'])([Oo][Hh][Mm])'
    # Replace with group 1, a space, and group 2, only if no space already exists
    # Using negative lookbehind for space before Ohm is tricky. Simpler: add space, then clean double spaces.
    s = re.sub(ohm_pattern, r'\1 \2', s)
    s = re.sub(r'\s+', ' ', s).strip() # Clean up potential double spaces

    # Add other specific fixes here if needed
    # Example: Ensure space before 'V' if preceded by digit/prefix?
    # voltage_pattern = r'([0-9' + "".join(re.escape(p) for p in MULTIPLIER_MAPPING.keys()) + r'])([Vv])'
    # s = re.sub(voltage_pattern, r'\1 \2', s)
    # s = re.sub(r'\s+', ' ', s).strip()

    return s


def replace_numbers_keep_sign_all(s: str) -> str:
    """Replaces all numbers (incl. scientific) with '$', preserving preceding sign."""
    s = str(s)
    # Regex captures optional sign ([+-]?), then digits with optional decimal/exponent.
    # It replaces the number part with '$', keeping the sign captured in group 1 (\1).
    # Handle standalone signs not followed by digits? No, regex requires digits.
    # Ensure it handles cases like ".5" correctly -> might become just "$"?
    # Pattern: (optional sign) followed by (digits OR .digits OR digits.digits) with optional exponent
    pattern = r'([+-]?)(\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
    return re.sub(pattern, r'\1$', s)


def replace_numbers_keep_sign_outside_parens(s: str) -> str:
    """Replaces numbers with '$' (keeping sign) only outside parentheses."""
    s = str(s) # Ensure string
    result = []
    i = 0
    depth = 0
    # Use the same number pattern as replace_numbers_keep_sign_all
    number_pattern = re.compile(r'([+-]?)(\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')

    while i < len(s):
        char = s[i]

        if char == '(':
            depth += 1
            result.append(char)
            i += 1
        elif char == ')':
            depth = max(0, depth - 1) # Prevent negative depth
            result.append(char)
            i += 1
        elif depth == 0:
            # Outside parentheses: check for a number starting at current position
            match_ = number_pattern.match(s, i) # Match from index i
            if match_:
                sign = match_.group(1) if match_.group(1) else '' # Captured sign or empty
                result.append(sign + '$')
                i = match_.end() # Move index past the matched number
            else:
                # Not a number, just append the character
                result.append(char)
                i += 1
        else:
            # Inside parentheses, just append the character
            result.append(char)
            i += 1

    return "".join(result)
