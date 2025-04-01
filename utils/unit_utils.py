# utils/unit_utils.py

import re

def process_unit_token_no_paren(token, base_units, multipliers_dict):
    """
    Processes a unit token (without parentheses) by checking for built-in
    units or known multiplier prefixes.
    """
    token = token.strip()
    if token.startswith('$'):
        after_dollar = token[1:]
        # Check if there is a space right after "$"
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
        return f"Error: Undefined unit '{stripped}' (no recognized prefix)"
    else:
        stripped_token = token.strip()
        if stripped_token in base_units:
            return "$" + stripped_token
        sorted_prefixes = sorted(multipliers_dict.keys(), key=len, reverse=True)
        for prefix in sorted_prefixes:
            if stripped_token.startswith(prefix):
                possible_base = stripped_token[len(prefix):]
                if possible_base in base_units:
                    return "$" + possible_base
        return f"Error: Undefined unit '{stripped_token}' (no recognized prefix)"

def process_unit_token(token, base_units, multipliers_dict):
    """
    Processes a unit token that may include numeric parts and parentheses.
    It extracts numeric and unit components, processes the core unit, and
    then reconstructs the token with original spacing.
    """
    pattern = re.compile(
        r'^(?P<lead>\s*)'
        r'(?P<numeric>[+\-±]?\d*(?:\.\d+)?)(?P<space1>\s*)'
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
    # Handle ohm spacing if needed
    if "ohm" in core.lower():
        if processed_core.startswith("$") and not processed_core.startswith("$ "):
            processed_core = "$ " + processed_core[1:].lstrip()
        if numeric and not space1:
            space1 = " "
    return f"{lead}{numeric}{space1}{processed_core}{space2}{paren}{trail}"

def split_outside_parens(text, delimiters):
    """
    Splits the text on provided delimiters that occur outside of any parentheses.
    Returns a list of tokens and delimiters.
    """
    tokens = []
    current = ""
    i = 0
    depth = 0
    sorted_delims = sorted(delimiters, key=len, reverse=True)
    while i < len(text):
        ch = text[i]
        if ch == '(':
            depth += 1
            current += ch
            i += 1
        elif ch == ')':
            depth = max(depth - 1, 0)
            current += ch
            i += 1
        elif depth == 0:
            matched = None
            for delim in sorted_delims:
                if text[i:i+len(delim)] == delim:
                    matched = delim
                    break
            if matched:
                if current:
                    tokens.append(current)
                tokens.append(matched)
                current = ""
                i += len(matched)
            else:
                current += ch
                i += 1
        else:
            current += ch
            i += 1
    if current:
        tokens.append(current)
    return tokens

def resolve_compound_unit(normalized_unit, base_units, multipliers_dict):
    """
    Resolves a compound unit string by splitting it outside parentheses using
    delimiters ("to", ",", "@"), processing each token, and concatenating the result.
    """
    tokens = split_outside_parens(normalized_unit, delimiters=["to", ",", "@"])
    resolved_parts = []
    for part in tokens:
        if part in ["to", ",", "@"]:
            resolved_parts.append(part)
        else:
            if part == "":
                continue
            resolved_parts.append(process_unit_token(part, base_units, multipliers_dict))
    return "".join(resolved_parts)

def remove_parentheses_detailed(text: str) -> str:
    """
    Removes all substrings enclosed in parentheses from the text.
    """
    return re.sub(r'\([^)]*\)', '', text)

def extract_identifiers_detailed(text: str):
    """
    Extracts and returns all substrings enclosed in parentheses (including the parentheses)
    from the text.
    """
    return re.findall(r'\([^)]*\)', text)

def fix_exceptions(s: str) -> str:
    """
    Applies known fixes to string exceptions, such as ensuring proper spacing
    between numbers and unit parts (e.g., '200Ohm' becomes '200 Ohm').
    """
    return re.sub(r'(\d)([a-zA-Zµ]*Ohm)', r'\1 \2', s)

def count_main_items(main_str: str) -> int:
    """
    Counts the number of main items in the string. If " to " is present,
    returns 1; if commas are present, returns the count of items separated by commas.
    """
    main_str = main_str.strip()
    if not main_str:
        return 0
    if " to " in main_str:
        return 1
    if "," in main_str:
        return len([s for s in main_str.split(',') if s.strip()])
    return 1

def count_conditions(cond_str: str) -> int:
    """
    Counts the number of conditions in the string by splitting on commas.
    """
    cond_str = cond_str.strip()
    if not cond_str:
        return 0
    parts = [p.strip() for p in cond_str.split(',') if p.strip()]
    return len(parts)

def classify_condition(cond_str: str) -> str:
    """
    Classifies the condition part of a value string as 'Multiple Conditions',
    'Range Condition', or 'Single Condition'.
    """
    cond_str = cond_str.strip()
    if not cond_str:
        return ""
    if "," in cond_str:
        return "Multiple Conditions"
    if " to " in cond_str:
        return "Range Condition"
    return "Single Condition"

def classify_main(main_str: str) -> str:
    """
    Classifies the main part of a value string as 'Multi Value', 'Range Value',
    'Single Value', or 'Number' based on delimiters and content.
    """
    main_str = main_str.strip()
    if not main_str:
        return ""
    if "," in main_str:
        return "Multi Value"
    if " to " in main_str:
        return "Range Value"
    if re.search(r'[a-zA-Zµ]', main_str):
        return "Single Value"
    else:
        return "Number"

def classify_sub_value(subval: str):
    """
    Classifies a sub-value that may contain both a main part and a condition part (separated by '@').
    Returns a tuple with:
      - Combined classification (e.g., "Single Value with Single Condition")
      - Booleans indicating the presence of a range in main, multi values in main,
        range in condition, and multiple conditions.
      - The count of condition items and main items.
    """
    parts = subval.split('@', maxsplit=1)
    main_part = parts[0].strip()
    cond_part = parts[1].strip() if len(parts) > 1 else ""
    main_class = classify_main(main_part)
    cond_class = classify_condition(cond_part)
    cond_count = count_conditions(cond_part)
    classification = f"{main_class} with {cond_class}" if main_class and cond_class else main_class or ""
    has_range_in_main = (main_class == "Range Value")
    has_multi_value_in_main = (main_class == "Multi Value")
    has_range_in_condition = (cond_class == "Range Condition")
    has_multiple_conditions = (cond_class == "Multiple Conditions")
    main_item_count = count_main_items(main_part)
    return (classification, has_range_in_main, has_multi_value_in_main,
            has_range_in_condition, has_multiple_conditions, cond_count, main_item_count)
