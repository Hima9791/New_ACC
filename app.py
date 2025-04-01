import streamlit as st
import pandas as pd
from io import BytesIO

# Import functions from the new modular components
from github_helpers import download_mapping_file_from_github, update_mapping_file_on_github
from mapping_manager import save_mapping_to_disk, read_mapping_file
from pipelines.fixed_pipeline import process_fixed_pipeline_bytes
from pipelines.classification_pipeline import basic_classification, detailed_analysis, combine_results

#############################################
# APP UI: ACC Project Entry Point
#############################################

st.title("ACC Project")

# Hide GitHub button and other header elements
hide_button = """
    <style>
    [data-testid="stBaseButton-header"] {
        display: none;
    }
    </style>
"""
st.markdown(hide_button, unsafe_allow_html=True)

# Download and store mapping file from GitHub if not already cached
if "mapping_df" not in st.session_state:
    st.session_state["mapping_df"] = download_mapping_file_from_github()

mapping_df = st.session_state["mapping_df"]

# Validate that required columns exist in the mapping file
required_cols = {"Base Unit Symbol", "Multiplier Symbol"}
if not required_cols.issubset(mapping_df.columns):
    st.error(f"Mapping file must contain columns: {required_cols}")
    st.stop()

# Prepare base units for downstream pipelines (if needed)
base_units = set(str(u).strip() for u in mapping_df["Base Unit Symbol"].dropna().unique())

# Operation selection
operation = st.selectbox("Select Operation", ["Get Pattern", "Manage Units"])

#############################################
# OPERATION: GET PATTERN
#############################################
if operation == "Get Pattern":
    st.header("Get Pattern")
    st.write("1) Upload an Excel with a 'Value' column.")
    
    input_file = st.file_uploader("Upload Input Excel File", type=["xlsx"])
    if input_file:
        # Step 1: Save current mapping to local disk (required by pipelines)
        save_mapping_to_disk(mapping_df, "mapping.xlsx")
        
        # Read the input Excel file as bytes
        file_bytes = input_file.read()
        try:
            # Run Fixed Processing Pipeline (first pipeline)
            processed_bytes = process_fixed_pipeline_bytes(file_bytes)
            # Write processed bytes to a local file for the next steps
            with open("processed_output.xlsx", "wb") as f:
                f.write(processed_bytes)
        except Exception as e:
            st.error(f"Fixed Processing Pipeline error: {e}")
            st.stop()
        
        try:
            # Save user input file locally for classification pipelines
            with open("user_input.xlsx", "wb") as f:
                f.write(file_bytes)
            # Run Basic Classification pipeline to generate an intermediate file
            classified_path = basic_classification("user_input.xlsx", "classified_output.xlsx")
            # Run Detailed Analysis pipeline using the classified file and mapping file
            detailed_analysis(classified_path, "mapping.xlsx", "QA_new.xlsx")
        except Exception as e:
            st.error(f"Detailed Analysis error: {e}")
            st.stop()
        
        try:
            # Combine results from processing and analysis pipelines
            combine_results(
                processed_file="processed_output.xlsx",
                analysis_file="QA_new.xlsx",
                output_file="final_combined.xlsx"
            )
        except Exception as e:
            st.error(f"Combining error: {e}")
            st.stop()
        
        try:
            # Allow user to download the final combined Excel file
            with open("final_combined.xlsx", "rb") as f:
                final_bytes = f.read()
            st.success("Done! Download your final_combined.xlsx file below.")
            st.download_button(
                label="Download final_combined.xlsx",
                data=final_bytes,
                file_name="final_combined.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"Download error: {e}")

#############################################
# OPERATION: MANAGE UNITS
#############################################
elif operation == "Manage Units":
    st.header("Manage Units (GitHub mapping file)")
    
    st.subheader("Current Mapping File")
    st.dataframe(mapping_df)
    
    # Form for adding a new unit
    with st.form("add_unit_form"):
        new_unit = st.text_input("Enter new Base Unit Symbol")
        submit_new = st.form_submit_button("Add New Unit")
    if submit_new:
        if new_unit.strip():
            new_row = {"Base Unit Symbol": new_unit.strip(), "Multiplier Symbol": None}
            st.session_state["mapping_df"] = pd.concat(
                [mapping_df, pd.DataFrame([new_row])],
                ignore_index=True
            )
            st.success(f"New unit '{new_unit.strip()}' added!")
        else:
            st.error("The unit field is required.")
    
    # Deletion of an existing unit
    existing_units = mapping_df["Base Unit Symbol"].dropna().unique().tolist()
    if existing_units:
        to_delete = st.selectbox("Select a unit to delete", ["--Select--"] + existing_units)
        if st.button("Delete Selected Unit"):
            if to_delete == "--Select--":
                st.warning("Please select a valid unit to delete.")
            else:
                before_shape = mapping_df.shape
                st.session_state["mapping_df"] = mapping_df[mapping_df["Base Unit Symbol"] != to_delete]
                after_shape = st.session_state["mapping_df"].shape
                st.success(f"Unit '{to_delete}' has been deleted. (Rows before: {before_shape}, after: {after_shape})")
    else:
        st.info("No units available to delete.")
    
    st.subheader("Updated Mapping File")
    st.dataframe(st.session_state["mapping_df"])
    
    # Option to download the updated mapping file locally
    if st.button("Download Updated Mapping File"):
        towrite = BytesIO()
        st.session_state["mapping_df"].to_excel(towrite, index=False, engine='openpyxl')
        towrite.seek(0)
        st.download_button(
            label="Download mapping.xlsx",
            data=towrite,
            file_name="mapping.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    # Save changes back to GitHub
    if st.button("Save Changes to GitHub"):
        if update_mapping_file_on_github(st.session_state["mapping_df"]):
            st.success("Mapping file updated on GitHub!")
        else:
            st.error("Failed to update mapping file on GitHub.")
