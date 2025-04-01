# pipelines/fixed_pipeline.py

import pandas as pd
import re
import os
import gc
import io

from mapping_manager import read_mapping_file  # to read the local mapping file
from config import LOCAL_BASE_UNITS

#############################################
# Global Patterns & Utility Functions
#############################################

PATTERNS_12 = [
    (re.compile(r"^-?\d+(?:\.\d+)?(?:\s*\([^)]+\))?$"), "Number"),
    (re.compile(r"^-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+\s*(?:\([^)]+\))?$"), "Single Value"),
    (re.compile(
        r"^-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+\s*(?:\([^)]+\))?\s*@\s*-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+\s*(?:\([^)]+\))?$"
    ), "Single Value with Single Condition"),
    (re.compile(
        r"^-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+\s*(?:\([^)]+\))?\s*@\s*-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+\s*(?:\([^)]+\))?\s+to\s+-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+\s*(?:\([^)]+\))?$"
    ), "Single Value with Range Condition"),
    (re.compile(
        r"^-?\d+(?:\.\d+)?\s*(?:\([^)]+\))?\s*@\s*-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+\s*(?:\([^)]+\))?$"
    ), "Number with Single Condition"),
    (re.compile(
        r"^-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+\s*(?:\([^)]+\))?\s+to\s+-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+\s*(?:\([^)]+\))?\s*@\s*-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+\s*(?:\([^)]+\))?$"
    ), "Range Value Single Condition"),
    (re.compile(
        r"^-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+\s*(?:\([^)]+\))?\s+to\s+-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+\s*(?:\([^)]+\))?\s*@\s*-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+\s*(?:\([^)]+\))?\s+to\s+-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+\s*(?:\([^)]+\))?$"
    ), "Range Value with Range Condition"),
    (re.compile(
        r"^-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+\s*(?:\([^)]+\))?\s+to\s+-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+\s*(?:\([^)]+\))?$"
    ), "Range Values"),
    (re.compile(
        r"^-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+\s*(?:\([^)]+\))?\s*@\s*-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+\s*(?:\([^)]+\))?(?:\s*,\s*-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+\s*(?:\([^)]+\))?)+$"
    ), "Single Value Multi Condition"),
    (re.compile(
        r"^(?:-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+\s*(?:\([^)]+\))?(?:\s*,\s*-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+\s*(?:\([^)]+\)))*?)\s*@\s*-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+(?:\s*\([^)]+\))?$"
    ), "Multi Value with Single Condition"),
    (re.compile(
        r"^-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+\s*(?:\([^)]+\))?\s+to\s+-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+\s*(?:\([^)]+\))?\s*@\s*-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+(?:\s*\([^)]+\))?(?:\s*,\s*-?\d+(?:\.\d+)?\s*[a-zA-Zµ]+\s*(?:\([^)]+\))?)+$"
    ), "Range Value with Multi Condition"),
]

def detect_value_type(input_string: str) -> str:
    text = input_string.strip()
    for pattern, cat_name in PATTERNS_12:
        if pattern.match(text):
            return cat_name
    if "," in text:
        chunks = [c.strip() for c in text.split(",")]
        chunk_categories = []
        for c in chunks:
            cat_found = None
            for pat, single_cat in PATTERNS_12:
                if pat.match(c):
                    cat_found = single_cat
                    break
            if not cat_found:
                chunk_categories = []
                break
            chunk_categories.append(cat_found)
        if len(chunk_categories) > 1 and len(set(chunk_categories)) == 1:
            return "Multiple " + chunk_categories[0]
    return "Unknown"

def fix_exceptions(s: str) -> str:
    return re.sub(r'(\d)([a-zA-Zµ]*Ohm)', r'\1 \2', s)

#############################################
# Text Parsing & Mapping Generation Functions
#############################################

def extract_block_texts(main_key: str, category: str):
    main_key = main_key.strip()
    if category in ["Single Value with Single Condition", "Number with Single Condition"]:
        if "@" in main_key:
            return [x.strip() for x in main_key.split("@", 1)]
        else:
            return [main_key]
    elif category == "Single Value with Range Condition":
        if "@" in main_key:
            left, right = [x.strip() for x in main_key.split("@", 1)]
            right_parts = [x.strip() for x in right.split("to", 1)]
            return [left] + right_parts
        else:
            return [main_key]
    elif category == "Range Value Single Condition":
        if "@" in main_key:
            left, right = [x.strip() for x in main_key.split("@", 1)]
            left_parts = [x.strip() for x in left.split("to", 1)]
            return left_parts + [right]
        else:
            return [main_key]
    elif category == "Range Value with Range Condition":
        if "@" in main_key:
            left, right = [x.strip() for x in main_key.split("@", 1)]
            left_parts = [x.strip() for x in left.split("to", 1)]
            right_parts = [x.strip() for x in right.split("to", 1)]
            return left_parts + right_parts
        else:
            return [main_key]
    elif category == "Range Values":
        return [x.strip() for x in main_key.split("to")]
    elif category == "Single Value Multi Condition":
        if "@" in main_key:
            value_part, conds_part = [x.strip() for x in main_key.split("@", 1)]
            conds = [x.strip() for x in conds_part.split(",")]
            return [value_part] + conds
        else:
            return [main_key]
    elif category == "Multi Value with Single Condition":
        if "@" in main_key:
            values_part, cond_part = [x.strip() for x in main_key.split("@", 1)]
            values = [x.strip() for x in values_part.split(",")]
            return values + [cond_part]
        else:
            return [main_key]
    elif category == "Range Value with Multi Condition":
        if "@" in main_key:
            range_part, conds_part = [x.strip() for x in main_key.split("@", 1)]
            range_parts = [x.strip() for x in range_part.split("to", 1)]
            conds = [x.strip() for x in conds_part.split(",")]
            return range_parts + conds
        else:
            return [main_key]
    elif category == "Multi Values":
        return [x.strip() for x in main_key.split(",")]
    else:
        return [main_key]

def parse_value_unit_identifier(raw_chunk: str, base_units, multipliers_dict):
    """
    Parses a raw string (e.g. '50 mOhm') into a tuple (value_with_prefix, base_unit).
    """
    # Remove parentheses (e.g., "10k(typ)" -> "10k")
    chunk_no_paren = re.sub(r'\([^)]*\)', '', raw_chunk).strip()
    match = re.match(r'^([+\-]?\d+(?:\.\d+)?)(.*)$', chunk_no_paren)
    if not match:
        raise ValueError(f"Undefined unit in '{raw_chunk}' (no recognized prefix).")
    numeric_str_matched = match.group(1)
    remainder = match.group(2).strip()
    numeric_end_index = raw_chunk.find(numeric_str_matched) + len(numeric_str_matched)
    value_with_prefix = raw_chunk[:numeric_end_index]
    try:
        float(numeric_str_matched)
    except ValueError:
        raise ValueError(f"Invalid numeric in '{raw_chunk}'")
    prefix_sym = "1"
    base_unit = ""
    if remainder:
        if remainder in base_units:
            base_unit = remainder
        else:
            sorted_prefixes = sorted(multipliers_dict.keys(), key=len, reverse=True)
            found_prefix = False
            for prefix in sorted_prefixes:
                if remainder.startswith(prefix):
                    candidate_base = remainder[len(prefix):]
                    if candidate_base in base_units:
                        prefix_sym = prefix
                        base_unit = candidate_base
                        found_prefix = True
                        break
            if not found_prefix:
                raise ValueError(f"Undefined unit in '{raw_chunk}' (no recognized prefix).")
    if prefix_sym != "1":
        prefix_start_index = raw_chunk.find(prefix_sym, numeric_end_index)
        if prefix_start_index != -1:
            prefix_end_index = prefix_start_index + len(prefix_sym)
            value_with_prefix = raw_chunk[:prefix_end_index]
    return (value_with_prefix.strip(), base_unit)

def fill_mapping_for_part(part_tuple, block_info):
    (val_with_prefix, base_unit) = part_tuple
    result = {}
    for attr, code in zip(block_info["attributes"], block_info["codes"]):
        if attr == "Value":
            result[code] = {"value": val_with_prefix}
        elif attr == "Unit":
            result[code] = {"value": base_unit}
        else:
            result[code] = {"value": ""}
    return result

def get_code_prefixes_for_category(category_name: str):
    if category_name == "Number":
        return [{"prefix": "SN-", "codes": ["SN-V0", "SN-U0"], "attributes": ["Value", "Unit"]}]
    elif category_name == "Single Value":
        return [{"prefix": "SV-", "codes": ["SV-V0", "SV-U0"], "attributes": ["Value", "Unit"]}]
    elif category_name == "Single Value with Single Condition":
        return [
            {"prefix": "SV-", "codes": ["SV-V0", "SV-U0"], "attributes": ["Value", "Unit"]},
            {"prefix": "SC-", "codes": ["SC-V0", "SC-U0"], "attributes": ["Value", "Unit"]}
        ]
    elif category_name == "Single Value with Range Condition":
        return [
            {"prefix": "SV-", "codes": ["SV-V0", "SV-U0"], "attributes": ["Value", "Unit"]},
            {"prefix": "RC-", "codes": ["RC-Vn0", "RC-Un0"], "attributes": ["Value", "Unit"]},
            {"prefix": "RC-", "codes": ["RC-Vx0", "RC-Ux0"], "attributes": ["Value", "Unit"]}
        ]
    elif category_name == "Number with Single Condition":
        return [
            {"prefix": "SN-", "codes": ["SN-V0", "SN-U0"], "attributes": ["Value", "Unit"]},
            {"prefix": "SC-", "codes": ["SC-V0", "SC-U0"], "attributes": ["Value", "Unit"]}
        ]
    elif category_name == "Range Value Single Condition":
        return [
            {"prefix": "RV-", "codes": ["RV-Vn0", "RV-Un0"], "attributes": ["Value", "Unit"]},
            {"prefix": "RV-", "codes": ["RV-Vx0", "RV-Ux0"], "attributes": ["Value", "Unit"]},
            {"prefix": "SC-", "codes": ["SC-V0", "SC-U0"], "attributes": ["Value", "Unit"]}
        ]
    elif category_name == "Range Value with Range Condition":
        return [
            {"prefix": "RV-", "codes": ["RV-Vn0", "RV-Un0"], "attributes": ["Value", "Unit"]},
            {"prefix": "RV-", "codes": ["RV-Vx0", "RV-Ux0"], "attributes": ["Value", "Unit"]},
            {"prefix": "RC-", "codes": ["RC-Vn0", "RC-Un0"], "attributes": ["Value", "Unit"]},
            {"prefix": "RC-", "codes": ["RC-Vx0", "RC-Ux0"], "attributes": ["Value", "Unit"]}
        ]
    elif category_name == "Range Values":
        return [
            {"prefix": "RV-", "codes": ["RV-Vn0", "RV-Un0"], "attributes": ["Value", "Unit"]},
            {"prefix": "RV-", "codes": ["RV-Vx0", "RV-Ux0"], "attributes": ["Value", "Unit"]}
        ]
    elif category_name == "Single Value Multi Condition":
        return [{"prefix": "SV-", "codes": ["SV-V0", "SV-U0"], "attributes": ["Value", "Unit"]}]
    elif category_name == "Multi Value with Single Condition":
        return []
    elif category_name == "Range Value with Multi Condition":
        return [
            {"prefix": "RV-", "codes": ["RV-Vn0", "RV-Un0"], "attributes": ["Value", "Unit"]},
            {"prefix": "RV-", "codes": ["RV-Vx0", "RV-Ux0"], "attributes": ["Value", "Unit"]}
        ]
    elif category_name == "Multi Values":
        return []
    elif category_name == "Number with Multiple Conditions":
        return [{"prefix": "SN-", "codes": ["SN-V0", "SN-U0"], "attributes": ["Value", "Unit"]}]
    else:
        return [{"prefix": "SN-", "codes": ["SN-V0", "SN-U0"], "attributes": ["Value", "Unit"]}]

def generate_mapping(parsed_parts, category_name: str):
    blocks = get_code_prefixes_for_category(category_name)
    mapping = {}
    mc_counter = 1
    for i, part_tuple in enumerate(parsed_parts):
        if i < len(blocks):
            block_info = blocks[i]
            new_map = fill_mapping_for_part(part_tuple, block_info)
            mapping.update(new_map)
        else:
            block_info = {
                "prefix": "MC-",
                "codes": [f"MC-V{mc_counter}", f"MC-U{mc_counter}"],
                "attributes": ["Value", "Unit"]
            }
            new_map = fill_mapping_for_part(part_tuple, block_info)
            mapping.update(new_map)
            mc_counter += 1
    return mapping

def display_mapping(mapping_dict, desired_order, category, main_key):
    def get_attribute_from_code(code):
        suffix = code.split('-', 1)[1] if '-' in code else code
        if 'V' in suffix:
            return "Value"
        elif 'U' in suffix:
            return "Unit"
        return "Unknown"
    output = []
    used_codes = set()
    for code in desired_order:
        if code in mapping_dict:
            attr = get_attribute_from_code(code)
            row = {
                "Main Key": main_key,
                "Category": category,
                "Attribute": attr,
                "Code": code,
                "Value": mapping_dict[code]["value"]
            }
            output.append(row)
            used_codes.add(code)
    extras = [c for c in mapping_dict if c not in used_codes]
    extras.sort()
    for code in extras:
        attr = get_attribute_from_code(code)
        row = {
            "Main Key": main_key,
            "Category": category,
            "Attribute": attr,
            "Code": code,
            "Value": mapping_dict[code]["value"]
        }
        output.append(row)
    return output

def get_desired_order():
    return [
        "SN-V0", "SN-U0",
        "SV-V0", "SV-U0",
        "RV-Vn0", "RV-Un0", "RV-Vx0", "RV-Ux0",
        "RC-Vn0", "RC-Un0", "RC-Vx0", "RC-Ux0",
        "SC-V0", "SC-U0",
        "MC-V1", "MC-U1", "MC-V2", "MC-U2", "MC-V3", "MC-U3"
    ]

#############################################
# Main Pipeline Processing Functions
#############################################

def process_single_key(main_key: str, base_units, multipliers_dict):
    main_key_clean = fix_exceptions(main_key.strip())
    category = detect_value_type(main_key_clean)
    if category.startswith("Multiple "):
        chunks = [c.strip() for c in main_key_clean.split(",")]
        all_rows = []
        for idx, chunk in enumerate(chunks):
            try:
                chunk_cat = detect_value_type(chunk)
                block_texts = extract_block_texts(chunk, chunk_cat)
                parsed_parts = []
                for bt in block_texts:
                    part_val, part_unit = parse_value_unit_identifier(bt, base_units, multipliers_dict)
                    parsed_parts.append((part_val, part_unit))
                code_map = generate_mapping(parsed_parts, chunk_cat)
                new_code_map = {}
                for code, val in code_map.items():
                    new_code_map[f"M{idx+1}-{code}"] = val
                desired_order_chunk = [f"M{idx+1}-{c}" for c in get_desired_order()]
                rows = display_mapping(new_code_map, desired_order_chunk, chunk_cat, main_key_clean)
                all_rows.extend(rows)
            except Exception as e:
                error_row = {
                    "Main Key": main_key_clean,
                    "Category": "Error",
                    "Attribute": "Value",
                    "Code": "ERR",
                    "Value": str(e)
                }
                all_rows.append(error_row)
        return all_rows
    elif category != "Unknown":
        try:
            block_texts = extract_block_texts(main_key_clean, category)
            parsed_parts = []
            for bt in block_texts:
                part_val, part_unit = parse_value_unit_identifier(bt, base_units, multipliers_dict)
                parsed_parts.append((part_val, part_unit))
            code_map = generate_mapping(parsed_parts, category)
            rows = display_mapping(code_map, get_desired_order(), category, main_key_clean)
            return rows
        except Exception as e:
            return [{
                "Main Key": main_key_clean,
                "Category": "Error",
                "Attribute": "Value",
                "Code": "ERR",
                "Value": str(e)
            }]
    else:
        return [{
            "Main Key": main_key_clean,
            "Category": "Error",
            "Attribute": "Value",
            "Code": "ERR",
            "Value": "Value did not match any known pattern."
        }]

def process_fixed_pipeline_bytes(file_bytes: bytes):
    """
    Processes an input Excel file using the fixed processing pipeline.
    Reads all sheets, processes each row in chunks, and writes a 'Combined'
    output sheet to a new Excel file which is then returned as bytes.
    """
    print("Running Fixed Processing Pipeline...")
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    
    # Read mapping file from disk to get base_units and multipliers_dict
    local_base_units, multipliers_dict = read_mapping_file("mapping.xlsx")
    combined_base_units = LOCAL_BASE_UNITS.union(local_base_units)
    
    output_filename = 'processed_output.xlsx'
    writer = pd.ExcelWriter(output_filename, engine='openpyxl', mode='w')
    current_row = 0
    chunk_size = 500
    
    all_sheets = xls.sheet_names
    print(f"Found sheets: {all_sheets}")
    
    for sheet in all_sheets:
        print(f"Processing sheet: '{sheet}'")
        sheet_df = pd.read_excel(xls, sheet_name=sheet)
        sub_output = []
        for i in range(0, len(sheet_df), chunk_size):
            chunk = sheet_df.iloc[i:i + chunk_size]
            for row_index, row in chunk.iterrows():
                main_key = str(row.get('Value', '')).strip()
                if not main_key:
                    continue
                results = process_single_key(main_key, combined_base_units, multipliers_dict)
                orig_data = row.to_dict()
                for r in results:
                    new_row = orig_data.copy()
                    new_row.update(r)
                    new_row["Sheet"] = sheet
                    sub_output.append(new_row)
        if sub_output:
            out_df = pd.DataFrame(sub_output)
            out_df.to_excel(
                writer,
                sheet_name='Combined',
                startrow=current_row,
                index=False,
                header=(current_row == 0)
            )
            current_row += len(out_df) + (1 if current_row == 0 else 0)
        gc.collect()
    
    writer.close()
    
    with open(output_filename, "rb") as f:
        processed_bytes = f.read()
    
    os.remove(output_filename)
    return processed_bytes
