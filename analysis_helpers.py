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


# --- Utility functions (Mostly from original Section 3 & 4 helpers) ---

def extract_numeric_and_unit_analysis(token, base_units, multipliers_dict):
    """
    Analyzes a token to extract numeric value, multiplier symbol, base unit,
    and normalized value. Handles numbers, units, and combinations.

    Args:
        token (str): The string token to analyze (e.g., "10kOhm", "5V", "100").
        base_units (set): A set of known base unit symbols (e.g., {'V', 'Ohm', 'A'}).
        multipliers_dict (dict): A dictionary mapping multiplier symbols to factors (e.g., {'k': 1000}).

    Returns:
        tuple: (numeric_val, multiplier_symbol, base_unit, normalized_value, error_flag)
               - numeric_val (float/None): The extracted numeric value.
               - multiplier_symbol (str/None): The identified multiplier symbol (e.g., 'k', 'm') or "1" if none.
               - base_unit (str/None): The identified base unit symbol (e.g., 'V', 'Ohm').
               - normalized_value (float/None): Numeric value * multiplier factor.
               - error_flag (bool): True if a parsing error occurred.
    """
    token = str(token).strip() # Ensure string and strip whitespace
    if not token:
        return None, None, None, None, False # Not an error, just empty

    # Regex to capture optional sign, numeric part, and the rest (unit, etc.)
    # Allows scientific notation like 1.23e-4, handles +/-/± signs
    # Handles cases like ".5" -> 0.5
    pattern = re.compile(r'^(?P<numeric>[+\-±]?\d*(?:\.\d+)?(?:[eE][+\-]?\d+)?)(?P<rest>.*)$')
    m = pattern.match(token)

    # Handle cases where the token might be *only* a unit or multiplier+unit
    if not m or not m.group("numeric"): # No numeric part found at the start
        # Check if it's just a known unit or starts with a known prefix followed by a known unit
        is_unit_only = False
        if token in base_units:
             is_unit_only = True
             # Return None for numeric parts, "1" for multiplier, token as base unit
             return None, "1", token, None, False
        else:
             # Check if it starts with a prefix and the rest is a base unit
             sorted_prefixes = sorted(multipliers_dict.keys(), key=len, reverse=True)
             for prefix in sorted_prefixes:
                  if token.startswith(prefix):
                       possible_base = token[len(prefix):].strip()
                       if possible_base in base_units:
                            # Found prefix + base unit only
                            return None, prefix, possible_base, None, False

        # If it wasn't purely a unit, and didn't start with a number, it's likely an error or just text
        # Let's flag as error ONLY if it's not recognized as a unit pattern above
        if not is_unit_only:
            # Consider if just text (e.g., "None", "Typical") should be an error or return specific values
            # For now, treat non-numeric, non-unit as error for this function's purpose
            # st.write(f"DEBUG: Token '{token}' is non-numeric and not recognized as unit.")
            return None, None, None, None, True # True error
        else:
             # This path shouldn't be reached if is_unit_only handled correctly above
             return None, None, None, None, False


    # If we got here, a numeric part was found by the regex
    numeric_str_raw = m.group("numeric")
    rest = m.group("rest").strip()

    # Refine numeric conversion robustness
    numeric_val = None
    numeric_str_clean = numeric_str_raw.replace('±','') # Remove ± for float conversion
    try:
        if numeric_str_clean: # Ensure not empty string after removing sign etc.
            numeric_val = float(numeric_str_clean)
    except ValueError:
        # This indicates an issue with the regex or input format if numeric_str exists but isn't float-able
        return None, None, None, None, True # Error in numeric conversion

    # Case 1: Only a number was found (rest is empty)
    if not rest:
        # Correct: Return numeric value, "1" as multiplier, None as base unit
        return numeric_val, "1", None, numeric_val, False

    # Case 2: Number followed by something - try to parse unit and multiplier from 'rest'
    multiplier_symbol = None # Default to None initially
    base_unit = None       # Default to None initially
    multiplier_factor = 1.0  # Default factor
    found_unit_structure = False

    # Try matching known prefixes followed by known base units
    sorted_prefixes = sorted(multipliers_dict.keys(), key=len, reverse=True)
    for prefix in sorted_prefixes:
        if rest.startswith(prefix):
            possible_base = rest[len(prefix):].strip()
            if possible_base in base_units:
                multiplier_symbol = prefix
                base_unit = possible_base
                multiplier_factor = multipliers_dict[prefix]
                found_unit_structure = True
                break # Found the longest matching prefix + base unit

    # If no prefix structure was found, check if the entire 'rest' is a base unit
    if not found_unit_structure:
        if rest in base_units:
            # No prefix symbol, multiplier remains 1.0
            multiplier_symbol = "1" # Explicitly '1' for no prefix
            base_unit = rest
            found_unit_structure = True
        else:
            # The 'rest' part is not a known base unit and didn't start with a known prefix+base_unit
            # It might be an invalid unit, combined unit (Ohm/V - not handled here), or just text.
            # Flag as error since we couldn't resolve the unit part.
            # st.write(f"DEBUG: Unknown unit part '{rest}' in token '{token}'")
            return numeric_val, None, None, None, True # Error: Unrecognized unit part

    # Calculate normalized value if possible
    normalized_value = None
    if numeric_val is not None and multiplier_factor is not None:
         try:
            normalized_value = numeric_val * multiplier_factor
         except TypeError:
             # This should ideally not happen if numeric_val is float and factor is float
             st.warning(f"DEBUG: Type error during normalization for {numeric_val} * {multiplier_factor}")
             normalized_value = None # Or handle differently

    # Return results: original numeric, multiplier symbol found ("1" if none), base unit found, normalized value, error flag
    # Ensure multiplier_symbol is "1" if None was assigned but unit processing succeeded.
    final_multiplier_symbol = multiplier_symbol if multiplier_symbol is not None else "1"
    return numeric_val, final_multiplier_symbol, base_unit, normalized_value, False


def remove_parentheses_detailed(text: str) -> str:
    """Removes content within the outermost parentheses."""
    return re.sub(r'\([^()]*\)', '', str(text)) # Ensure input is string

def extract_identifiers_detailed(text: str):
    """Finds all content within non-nested parentheses."""
    return re.findall(r'\(([^()]*)\)', str(text)) # Finds content inside (...)


def split_outside_parens(text, delimiters):
    """
    Splits text by delimiters, ignoring delimiters inside parentheses.

    Args:
        text (str): The text to split.
        delimiters (list[str]): A list of delimiter strings.

    Returns:
        list[str]: A list of tokens.
    """
    text = str(text) # Ensure string input
    tokens = []
    current = ""
    i = 0
    depth = 0
    # Sort delimiters by length descending to match longest first (e.g., ' to ' before 'to')
    sorted_delims = sorted(delimiters, key=len, reverse=True)

    while i < len(text):
        char = text[i]

        if char == '(':
            depth += 1
            current += char
            i += 1
        elif char == ')':
            # Ensure depth doesn't go below zero
            depth = max(0, depth - 1)
            current += char
            i += 1
        elif depth == 0:
            # Check if any delimiter matches at the current position
            matched_delim = None
            for delim in sorted_delims:
                # Check bounds before slicing
                if i + len(delim) <= len(text) and text[i : i + len(delim)] == delim:
                    # Basic check succeeded, consider it a match for now
                    # More complex logic could check for surrounding whitespace if delimiters require it (e.g. ' to ')
                    matched_delim = delim
                    break

            if matched_delim:
                # Found a delimiter outside parentheses
                if current.strip(): # Add the part before the delimiter if not empty
                    tokens.append(current.strip())
                # Optionally add the delimiter itself as a token if needed:
                # tokens.append(matched_delim)
                current = "" # Reset current part
                i += len(matched_delim) # Move index past the delimiter
            else:
                # Not a delimiter, add char to current part
                current += char
                i += 1
        else:
            # Inside parentheses, just add the char
            current += char
            i += 1

    # Add the last part if any
    if current.strip():
        tokens.append(current.strip())

    # Filter out empty strings that might result from splitting errors or consecutive delimiters
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
    """
    DEPRECATED? analyze_unit_part and extract_numeric_and_unit_analysis are preferred.
    Attempts to resolve a token (without parentheses) to its base unit.
    """
    token = token.strip()
    st.warning("DEBUG: process_unit_token_no_paren may be deprecated.") # Add warning

    # Handle currency symbol ($) explicitly if needed, otherwise treat like other units.
    # Let's assume '$' is NOT a base unit unless added to mapping.xlsx.

    # Direct match for base unit
    if token in base_units:
        return token # Return the base unit itself

    # Check for prefix + base unit
    sorted_prefixes = sorted(multipliers_dict.keys(), key=len, reverse=True)
    for prefix in sorted_prefixes:
        if token.startswith(prefix):
            possible_base = token[len(prefix):].strip()
            if possible_base in base_units:
                # We found a valid prefix and base unit combination.
                return possible_base # Return the identified base unit

    # If no match, it's an unknown unit or just text without a unit
    # st.write(f"DEBUG: Unit part '{token}' not resolved to a base unit.")
    return None # Indicate no base unit could be resolved


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
    """
    DEPRECATED? Preferred: extract_numeric_and_unit_analysis or analyze_unit_part.
    Processes a token, aiming to return the base unit.
    """
    st.warning("DEBUG: process_unit_token may be deprecated.") # Add warning
    # Let's reuse the robust extraction logic:
    _, _, base_unit, _, err_flag = extract_numeric_and_unit_analysis(token, base_units, multipliers_dict)

    if not err_flag and base_unit:
        # Add special formatting if needed (Example was: Ohm -> $ Ohm) - Removed for neutrality
        return base_unit # Return the resolved base unit
    elif str(token).strip() in base_units:
         return str(token).strip() # It was just a base unit
    else:
        # Handle errors or unknowns
        # Return None or error marker? Let's return None.
        return None # Indicate unresolved


def resolve_compound_unit(normalized_unit_string, base_units, multipliers_dict):
    tokens = split_outside_parens(normalized_unit_string, delimiters=["to", ",", "@"])
    resolved_parts = []
    for part in tokens:
        if part in ["to", ",", "@"]:
            resolved_parts.append(part)
        else:
            if part == "":
                continue
            resolved_parts.append(process_unit_token(part, base_units, multipliers_dict))
    return "".join(resolved_parts)




def count_conditions(cond_str: str) -> int:
    """Counts logical condition clauses in the condition part (post-@)."""
    cond_str = remove_parentheses_detailed(cond_str).strip()
    if not cond_str:
        return 0

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
    """Classifies the structure of the condition part (post-@)."""
    cond_str = remove_parentheses_detailed(cond_str).strip()
    if not cond_str:
        return "" # No condition

    # Split by comma first
    comma_parts = split_outside_parens(cond_str, [','])
    num_comma_parts = len([p for p in comma_parts if p.strip()]) # Count non-empty parts

    # Check if *any* non-empty part contains a range marker ' to '
    has_range = any(re.search(r'\s+to\s+', part, re.IGNORECASE) for part in comma_parts if part.strip())

    if num_comma_parts > 1:
        # If multiple parts, classify as "Multiple". We don't detail if they are mixed range/single here.
        return "Multiple Conditions" # e.g., "5V, 10A" or "5V, 10V to 15V"
    elif num_comma_parts == 1:
        # Single non-empty part
        if has_range:
            return "Range Condition" # e.g., "5V to 10V"
        else:
            return "Single Condition" # e.g., "5V"
    else: # Should only happen if cond_str was effectively empty after processing
        return "" # Treat as no condition


def classify_main(main_str: str) -> str:
    """Classifies the structure of the main part (pre-@)."""
    main_str = remove_parentheses_detailed(main_str).strip()
    if not main_str:
        return "" # Empty main part

    # Split by comma first
    comma_parts = split_outside_parens(main_str, [','])
    num_comma_parts = len([p for p in comma_parts if p.strip()]) # Count non-empty parts

    # Check if *any* non-empty part contains ' to '
    has_range = any(re.search(r'\s+to\s+', part, re.IGNORECASE) for part in comma_parts if part.strip())

    # Check if *any* part contains characters typical of units (letters, µ, °, %)
    # Exclude 'to' itself from this check.
    unit_char_pattern = r'[a-zA-Zµ°%]'
    has_unit_chars = any(re.search(unit_char_pattern, re.sub(r'\s+to\s+', '', part, flags=re.IGNORECASE)) for part in comma_parts if part.strip())


    if num_comma_parts > 1:
        # Multiple comma-separated items in the main part
        return "Multi Value" # e.g., "10A, 20A" or "10, 20 to 30"
    elif num_comma_parts == 1:
        # Single non-empty part
        single_part = comma_parts[0].strip()
        if has_range:
            return "Range Value" # e.g., "10A to 20A" or "10 to 20"
        else:
            # Single part, not a range. Check content.
            if has_unit_chars:
                return "Single Value" # e.g., "10A", "50%", "25°C"
            # Check if it's purely numeric (allowing sign, decimal, exponent)
            elif re.fullmatch(r'[+\-±]?\d*(?:\.\d+)?(?:[eE][+\-]?\d+)?', single_part):
                # Check it's not empty string if regex allows optional parts
                if single_part:
                     return "Number" # e.g., "10", "-5.5", ".5"
                else:
                     return "Unknown Single Structure" # Empty after processing?
            else:
                # Contains other characters, maybe mixed alphanumeric, or unparsed structure?
                return "Complex Single" # E.g. "Typ 5", "Approx 10V" (if V not base unit)
    else:
        return "" # Treat as empty


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
    """
    Performs detailed classification of a raw value string, identifying sub-values,
    their structure (main/condition, range/single/multi), identifiers, and counts.

    Args:
        raw_value (str): The complete value string to classify.

    Returns:
        tuple: (final_class_str, identifiers_str, sub_value_count,
                max_cond_item_count, any_range_main, any_multi_main,
                any_range_cond, any_multi_cond, max_main_item_count)
    """
    raw_value = str(raw_value).strip()
    if not raw_value:
        # Return default values for empty input
        return ("Empty", "", 0, 0, False, False, False, False, 0)

    # Extract identifiers (content in parentheses) first
    found_parens_content = extract_identifiers_detailed(raw_value)
    # Join identifiers with comma-space, ensuring uniqueness and order? Simple join for now.
    identifiers = ', '.join(found_parens_content)

    # --- Determine Sub-Values ---
    # The primary separator for distinct sub-values is COMMA, *unless* that comma
    # is part of a condition definition following an '@'.
    # Heuristic: Split by comma outside parentheses. Then, if a part *doesn't* contain '@'
    # but the *previous* part did, consider merging it back as part of the condition.
    # Example: "A @ X, Y" -> initial split ["A @ X", "Y"]. Since "Y" has no "@" and previous did, merge?
    # Example: "A @ X, B @ Y" -> initial split ["A @ X", "B @ Y"]. Both have "@", keep separate.
    # Example: "A, B" -> initial split ["A", "B"]. Keep separate.
    # Example: "A @ X to Y, Z" -> initial split ["A @ X to Y", "Z"]. Keep separate.

    # Let's stick to the simpler split_outside_parens by comma for now.
    # This assumes commas are primarily for separating independent values/measurements.
    subvals_raw = split_outside_parens(raw_value, [','])

    # Filter out empty strings that might result from splitting ", ,"
    subvals = [sv.strip() for sv in subvals_raw if sv.strip()]

    # Ensure we have at least one subval if the raw_value wasn't empty but split resulted in none
    if not subvals and raw_value:
        subvals = [raw_value] # Treat the whole string as one sub-value

    sub_value_count = len(subvals)
    if sub_value_count == 0: # Should not happen if raw_value was not empty initially
        return ("Invalid/Empty Structure", identifiers, 0, 0, False, False, False, False, 0)

    # --- Analyze each sub-value ---
    sub_results = []
    for sv in subvals:
        sub_results.append(classify_sub_value(sv))

    # --- Aggregate results ---
    all_classifications = [res[0] for res in sub_results]
    # Filter out potential "Empty" or "Invalid" classifications if others exist?
    valid_classifications = [c for c in all_classifications if c not in ["Empty", "Invalid/Empty Structure"]]

    final_class = ""
    if sub_value_count == 1:
        final_class = all_classifications[0] # Use the classification of the single sub-value
    else:
        # Multiple sub-values
        unique_valid_classes = set(valid_classifications)
        if len(unique_valid_classes) == 1:
            # All valid sub-values have the same structure
            base_class = next(iter(unique_valid_classes))
            final_class = f"Multiple ({sub_value_count}x) {base_class}"
        elif len(unique_valid_classes) > 1:
            # Mixed structures among sub-values
            final_class = f"Multiple Mixed ({sub_value_count}x)"
            # Optionally list the types: f"Multiple Mixed ({sub_value_count}x: {', '.join(sorted(unique_valid_classes))})"
        else:
            # All sub-values were empty or invalid
            final_class = "Multiple Invalid/Empty"


    # Aggregate boolean flags (True if *any* sub-value had the feature)
    has_range_in_main_overall = any(res[1] for res in sub_results)
    has_multi_value_in_main_overall = any(res[2] for res in sub_results)
    has_range_in_condition_overall = any(res[3] for res in sub_results)
    has_multiple_conditions_overall = any(res[4] for res in sub_results)

    # Aggregate counts (use max)
    final_cond_item_count_agg = max([res[5] for res in sub_results]) if sub_results else 0
    final_main_item_count_agg = max([res[6] for res in sub_results]) if sub_results else 0


    return (final_class, # The overall classification string
            identifiers, # Comma-separated string of content in ()
            sub_value_count, # How many comma-separated sub-values were detected
            final_cond_item_count_agg, # Max condition clauses found in any sub-value
            has_range_in_main_overall, # Any sub-value had range in its main part?
            has_multi_value_in_main_overall, # Any sub-value had multi-value in its main part?
            has_range_in_condition_overall, # Any sub-value had range in its condition?
            has_multiple_conditions_overall, # Any sub-value had multiple condition clauses?
            final_main_item_count_agg) # Max main items found in any sub-value


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
