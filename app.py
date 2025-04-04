#############################################
# STREAMLIT SCRIPT - UNIFIED PIPELINE APP
# Main application file for UI and pipeline orchestration.
#############################################

import streamlit as st
import pandas as pd
# import re # No longer needed directly in app.py
import os
# import base64 # No longer needed directly in app.py
# import requests # No longer needed directly in app.py
import gc # Keep for potential explicit calls
from io import BytesIO
import io # Make sure io is imported if used for BytesIO reading/writing directly

# Import functions from custom modules
from github_utils import download_mapping_file_from_github, update_mapping_file_on_github
from mapping_utils import save_mapping_to_disk # read_mapping_file not needed directly by UI
from fixed_pipeline import process_fixed_pipeline_bytes
from detailed_pipeline import detailed_analysis
from result_combiner import combine_results

#############################################
# HIDE GITHUB ICON & OTHER ELEMENTS
#############################################
# (Keep this UI customization in the main app file)
hide_button = """
    <style>
    [data-testid="stBaseButton-header"] {
        display: none;
    }
    </style>
    """
st.markdown(hide_button, unsafe_allow_html=True)


#############################################
# STREAMLIT APP UI & LOGIC
#############################################

st.title("ACC Project - Unified Pipeline")

# Define mapping file name constant
MAPPING_FILENAME = "mapping.xlsx"

# --- Initialize Mapping File ---
# Attempt to load mapping_df from GitHub into session_state on first run or if invalidated.
# Also ensures the mapping file is saved locally for the pipelines to use.
if "mapping_df" not in st.session_state or st.session_state.get("mapping_df") is None:
    st.write("DEBUG: mapping_df not in session state or is None. Attempting download...")
    try:
        # Download from GitHub
        mapping_data = download_mapping_file_from_github()
        if mapping_data is not None:
            st.session_state["mapping_df"] = mapping_data
            # Immediately save to disk after successful download for pipeline use
            save_mapping_to_disk(st.session_state["mapping_df"], MAPPING_FILENAME)
            st.success("Mapping file downloaded and saved locally.")
        else:
             # download_mapping_file_from_github should st.stop() on failure,
             # but handle None case just in case.
             st.error("Failed to download mapping file (returned None). Cannot proceed.")
             st.session_state["mapping_df"] = None # Ensure state reflects failure
             st.stop()
    except Exception as e:
        st.error(f"Error initializing mapping file from GitHub: {e}")
        st.session_state["mapping_df"] = None # Ensure state reflects failure
        st.stop() # Stop execution if mapping fails to initialize


# --- Validate Mapping File (after potential download) ---
# Check if mapping_df is valid in session state before proceeding
if st.session_state.get("mapping_df") is None:
    st.error("Mapping data could not be loaded. Please check GitHub configuration/connection and refresh.")
    # Offer a button to retry?
    if st.button("Retry Download Mapping"):
        if "mapping_df" in st.session_state: del st.session_state["mapping_df"]
        st.rerun()
    st.stop() # Stop if mapping is essential and missing
else:
    # Validate required columns after ensuring df is loaded
    required_cols = {"Base Unit Symbol", "Multiplier Symbol"} # Keep this check
    current_mapping_df = st.session_state["mapping_df"]
    if not required_cols.issubset(current_mapping_df.columns):
        st.error(f"Mapping file from GitHub (or session state) is missing required columns: {required_cols}. Please fix the file via 'Manage Units' or directly on GitHub.")
        # Allow viewing/managing but maybe disable Get Pattern if invalid?
        mapping_valid_for_processing = False
    else:
        mapping_valid_for_processing = True


# --- Main App Navigation ---
operation = st.selectbox("Select Operation", ["Get Pattern", "Manage Units"])

############################
# OPERATION: GET PATTERN
############################
if operation == "Get Pattern":
    st.header("Get Pattern")

    # Re-check mapping file validity specifically for this operation
    if not mapping_valid_for_processing:
         st.error("Cannot run 'Get Pattern' because the mapping file is missing required columns. Please check/fix it via 'Manage Units' or on GitHub.")
         st.stop()
    if not os.path.exists(MAPPING_FILENAME):
        st.error(f"Local mapping file '{MAPPING_FILENAME}' not found. Please ensure it was downloaded/saved (try refreshing or managing units).")
        # Try saving it again?
        if st.session_state.get("mapping_df") is not None:
            save_mapping_to_disk(st.session_state["mapping_df"], MAPPING_FILENAME)
            st.rerun() # Rerun to check existence again
        st.stop()


    st.write("Upload an Excel file containing a 'Value' column for processing.")

    # --- File Uploader ---
    input_file = st.file_uploader("Upload Input Excel File", type=["xlsx"], key="pattern_uploader")

    if input_file:
        # Define output filenames for this run
        # Use dynamic names or clear previous outputs if needed
        user_input_filename_temp = "user_input_temp_for_analysis.xlsx" # Temp file for detailed analysis input
        analysis_output_filename = "detailed_analysis_output.xlsx"
        final_output_filename = "final_combined_output.xlsx"

        # --- Progress Bar and Status ---
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.info("Starting processing...")

        try:
            # --- Read Input File ---
            status_text.info("Reading input file...")
            progress_bar.progress(5)
            input_file_bytes = input_file.read()

            # --- Save input bytes locally (needed for detailed_analysis function) ---
            # TODO: Modify detailed_analysis to accept bytes or DataFrame directly?
            # For now, save then read back.
            try:
                with open(user_input_filename_temp, "wb") as f:
                    f.write(input_file_bytes)
                # Read it back into a DataFrame for the detailed analysis step
                input_df_for_analysis = pd.read_excel(user_input_filename_temp)
                st.write(f"DEBUG: Read user input file into DataFrame. Shape: {input_df_for_analysis.shape}")
                if 'Value' not in input_df_for_analysis.columns:
                     st.error("Input file MUST contain a 'Value' column.")
                     st.stop()
            except Exception as e:
                 st.error(f"Error reading uploaded Excel file: {e}. Ensure format is correct and 'Value' column exists.")
                 st.stop()


            # --- Step 1: Ensure mapping is on disk (already checked, but double-check) ---
            status_text.info("Verifying local mapping file...")
            progress_bar.progress(10)
            if not os.path.exists(MAPPING_FILENAME):
                 st.error(f"Critical Error: Local mapping file '{MAPPING_FILENAME}' disappeared.")
                 st.stop()


            # --- Step 2: Run Fixed Processing Pipeline ---
            status_text.info("Running fixed processing pipeline (generating codes)...")
            progress_bar.progress(20)
            # Pass bytes and mapping path; returns a DataFrame or None
            processed_df = process_fixed_pipeline_bytes(input_file_bytes, MAPPING_FILENAME)

            if processed_df is None: # Check for critical failure
                st.error("Fixed Processing Pipeline failed critically. Aborting.")
                st.stop()
            if processed_df.empty:
                # Allow proceeding, but combine step will likely just have analysis data.
                st.warning("Fixed Processing Pipeline did not generate any output rows. Check input 'Value' column content and mapping.")
                # Create an empty DF with 'Main Key' to allow merge? Or handle in combine?
                # Let combine_results handle the empty processed_df.
            else:
                 progress_bar.progress(40)
                 st.write(f"DEBUG: Fixed processing pipeline complete. Output shape: {processed_df.shape}")


            # --- Step 3: Run Detailed Analysis Pipeline ---
            status_text.info("Running detailed analysis pipeline...")
            progress_bar.progress(50)
            # Pass the input DataFrame read earlier, mapping path, and output path
            analysis_result_path = detailed_analysis(
                input_df=input_df_for_analysis, # Pass the DataFrame
                mapping_file=MAPPING_FILENAME,
                output_file=analysis_output_filename # Saves result to this file
            )

            if analysis_result_path is None:
                st.error("Detailed Analysis Pipeline failed. Aborting.")
                # Clean up temp input file?
                if os.path.exists(user_input_filename_temp): os.remove(user_input_filename_temp)
                st.stop()
            progress_bar.progress(80)
            st.write(f"DEBUG: Detailed analysis complete. Output saved to {analysis_result_path}")


            # --- Step 4: Combine Results ---
            status_text.info("Combining fixed and detailed results...")
            progress_bar.progress(90)
            # Pass the processed DataFrame and the path to the analysis results file
            final_result_path = combine_results(
                processed_df=processed_df, # Pass the DataFrame from fixed pipeline
                analysis_file=analysis_output_filename, # Path to detailed analysis output
                output_file=final_output_filename # Path for final combined output
            )

            if final_result_path is None:
                st.error("Combining results failed. Aborting.")
                # Clean up temp files?
                if os.path.exists(user_input_filename_temp): os.remove(user_input_filename_temp)
                if os.path.exists(analysis_output_filename): os.remove(analysis_output_filename)
                st.stop()

            progress_bar.progress(100)
            status_text.success("Processing Complete!")


            # --- Step 5: Offer Download ---
            try:
                with open(final_result_path, "rb") as fp:
                    final_bytes = fp.read()
                st.download_button(
                    label=f"Download Combined Results ({os.path.basename(final_result_path)})",
                    data=final_bytes,
                    file_name=os.path.basename(final_result_path), # Use the actual output name
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_final"
                )
            except Exception as e:
                st.error(f"Error preparing download link for final results: {e}")

            # --- Clean up temporary files ---
            # Consider keeping them for debugging or making cleanup optional
            cleanup_files = [user_input_filename_temp, analysis_output_filename] # Keep final_output_filename for download
            for f in cleanup_files:
                try:
                    if os.path.exists(f):
                         os.remove(f)
                         st.write(f"DEBUG: Removed temporary file {f}")
                except Exception as e:
                     st.warning(f"Could not remove temporary file {f}: {e}")


        except Exception as e:
            status_text.error(f"An error occurred during the 'Get Pattern' process: {e}")
            import traceback
            st.error(traceback.format_exc()) # Show detailed error in Streamlit UI/logs
            # Attempt cleanup on error
            cleanup_files = [user_input_filename_temp, analysis_output_filename, final_output_filename]
            for f in cleanup_files:
                try:
                    if os.path.exists(f): os.remove(f)
                except Exception as clean_e:
                     st.warning(f"Could not remove temp file {f} during error cleanup: {clean_e}")


############################
# OPERATION: MANAGE UNITS
############################
elif operation == "Manage Units":
    st.header("Manage Units (GitHub mapping file)")

    # Display current mapping from session state
    st.subheader("Current Mapping File (from GitHub / Session State)")
    current_mapping_df = st.session_state.get("mapping_df") # Use .get for safety

    if current_mapping_df is not None:
         # Display the DataFrame using Streamlit's dataframe widget
         st.dataframe(current_mapping_df)

         # Add button to explicitly refresh from GitHub
         if st.button("Refresh Mapping from GitHub"):
              if "mapping_df" in st.session_state: del st.session_state["mapping_df"]
              st.rerun() # Clears state and reruns script from top, triggering download

    else:
         st.warning("Mapping data not currently loaded in session state.")
         if st.button("Retry Download from GitHub"):
              # Clear state and rerun to trigger download attempt
              if "mapping_df" in st.session_state: del st.session_state["mapping_df"]
              st.rerun()


    # --- Add New Unit (Modify Session State) ---
    st.subheader("Add New Base Unit")
    with st.form("add_unit_form"):
        new_unit = st.text_input("Enter new Base Unit Symbol").strip()
        # Optional: Add inputs for other required columns if mapping structure changes
        # new_multiplier = st.text_input("Associated Multiplier Symbol (optional)").strip()
        submitted_new = st.form_submit_button("Add Unit to Local Session")

    if submitted_new and new_unit:
        if current_mapping_df is not None:
            # Check if unit already exists (case-sensitive check)
            if new_unit in current_mapping_df["Base Unit Symbol"].astype(str).values:
                 st.warning(f"Unit '{new_unit}' already exists in the session mapping.")
            else:
                # Create a new row dictionary matching the DataFrame's columns
                new_row_data = {"Base Unit Symbol": new_unit, "Multiplier Symbol": None} # Default multiplier None
                # Add defaults for any other columns the mapping file might have
                for col in current_mapping_df.columns:
                    if col not in new_row_data:
                         new_row_data[col] = None # Or appropriate default

                # Convert the single row dict to a DataFrame to concatenate
                new_row_df = pd.DataFrame([new_row_data], columns=current_mapping_df.columns)

                # Update the DataFrame in session state
                st.session_state["mapping_df"] = pd.concat(
                    [current_mapping_df, new_row_df],
                    ignore_index=True
                )
                st.success(f"New unit '{new_unit}' added to the current session. Save to GitHub to persist.")
                # Use st.rerun() to refresh the UI and show the updated dataframe
                st.rerun()
        else:
            st.error("Mapping data not available in session state to add unit.")
    elif submitted_new: # Handle case where submit clicked but input is empty
        st.error("Base Unit Symbol cannot be empty.")


    # --- Delete Unit (Modify Session State) ---
    st.subheader("Delete Base Unit")
    if current_mapping_df is not None and not current_mapping_df.empty:
        # Get unique, non-empty base units from the current session DataFrame
        try:
             existing_units = sorted(current_mapping_df["Base Unit Symbol"].dropna().astype(str).unique())
             existing_units = [unit for unit in existing_units if unit] # Filter out empty strings if any
        except KeyError:
             st.error("Cannot retrieve units: 'Base Unit Symbol' column not found.")
             existing_units = []

        if existing_units:
            # Use selectbox for deletion choice
            unit_to_delete = st.selectbox(
                "Select a unit to delete from local session",
                options=["--Select--"] + existing_units,
                key="delete_unit_select"
            )

            # Use a button to confirm deletion
            if st.button("Delete Selected Unit from Local Session"):
                if unit_to_delete != "--Select--":
                    original_shape = st.session_state["mapping_df"].shape
                    # Filter out the selected unit
                    st.session_state["mapping_df"] = st.session_state["mapping_df"][
                        st.session_state["mapping_df"]["Base Unit Symbol"] != unit_to_delete
                    ].reset_index(drop=True) # Reset index after deletion
                    new_shape = st.session_state["mapping_df"].shape
                    st.success(f"Unit '{unit_to_delete}' deleted from the current session. (Rows before: {original_shape[0]}, after: {new_shape[0]}). Save to GitHub to persist.")
                    # Rerun to update the displayed dataframe and selectbox options
                    st.rerun()
                else:
                    st.warning("Please select a unit to delete.")
        else:
            st.info("No base units found in the current mapping session to delete.")
    elif current_mapping_df is not None and current_mapping_df.empty:
         st.info("Mapping data in session is currently empty.")
    else:
        # Handled above, but reiterate if needed
        st.info("Mapping data is not loaded.")


    # --- Persist Changes ---
    st.subheader("Persist Changes")
    st.warning("Changes made using the forms above only affect the current browser session.")

    # Option 1: Download the session's current mapping file locally
    if current_mapping_df is not None:
        try:
            # Use BytesIO to create the Excel file in memory for download
            output_buffer = BytesIO()
            current_mapping_df.to_excel(output_buffer, index=False, engine='openpyxl')
            output_buffer.seek(0) # Rewind buffer to the beginning
            st.download_button(
                label="Download Current Mapping File (Local Session)",
                data=output_buffer,
                file_name="session_mapping.xlsx", # Suggest a filename
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_session_mapping"
            )
        except Exception as e:
             st.error(f"Error creating download link for session mapping: {e}")


    # Option 2: Save the session's current mapping back to GitHub
    if st.button("Save Current Session Mapping to GitHub"):
        if current_mapping_df is not None:
            st.info("Attempting to save current session mapping to GitHub...")
            # --- Pre-upload Validation ---
            # Ensure required columns are still present before attempting upload
            required_cols = {"Base Unit Symbol", "Multiplier Symbol"} # Re-check required columns
            if not required_cols.issubset(current_mapping_df.columns):
                 st.error(f"Cannot save to GitHub: Mapping file is missing required columns: {required_cols}. Please add them back or refresh from GitHub.")
            else:
                # Call the GitHub update function from github_utils
                success = update_mapping_file_on_github(current_mapping_df)
                if success:
                    st.success("Mapping file updated on GitHub! Changes will be reflected after the app fully refreshes or restarts.")
                    # Session state was already cleared by update_mapping_file_on_github on success
                    # Trigger a rerun to potentially show updated state if desired, though download is needed next time.
                    # st.rerun() # Optional: forces immediate UI update cycle
                else:
                    st.error("Failed to update mapping file on GitHub. Check console/logs for details (e.g., permissions, token validity, conflicts).")
        else:
            st.error("No mapping data found in the current session to save.")
