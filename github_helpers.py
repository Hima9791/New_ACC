import streamlit as st
import requests
import base64
import os
import pandas as pd

from config import GITHUB_OWNER, GITHUB_REPO, GITHUB_TOKEN, GITHUB_FILE_PATH, MAPPING_FILE_LOCAL

def download_mapping_file_from_github() -> pd.DataFrame:
    """
    Downloads 'mapping.xlsx' from the GitHub repo specified via Streamlit secrets or config,
    then returns a DataFrame parsed from that file.
    """
    st.write("DEBUG: Downloading mapping.xlsx from GitHub...")
    
    # Prefer values from st.secrets, fallback to config.py values
    github_token = st.secrets.get("github", {}).get("token", GITHUB_TOKEN)
    owner = st.secrets.get("github", {}).get("owner", GITHUB_OWNER)
    repo = st.secrets.get("github", {}).get("repo", GITHUB_REPO)
    file_path = st.secrets.get("github", {}).get("file_path", GITHUB_FILE_PATH)

    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content_json = response.json()
        encoded_content = content_json["content"]
        decoded_bytes = base64.b64decode(encoded_content)

        # Save to local file for parsing
        local_file = MAPPING_FILE_LOCAL
        with open(local_file, "wb") as f:
            f.write(decoded_bytes)

        try:
            df = pd.read_excel(local_file)
        except Exception as e:
            st.error(f"Failed to parse downloaded mapping file: {e}")
            st.stop()

        os.remove(local_file)  # Clean up local file
        st.write("DEBUG: Download successful. mapping_df shape:", df.shape)
        return df
    else:
        st.error(f"Failed to download file from GitHub: {response.status_code} {response.text}")
        st.stop()


def update_mapping_file_on_github(mapping_df: pd.DataFrame) -> bool:
    """
    Updates 'mapping.xlsx' on GitHub by saving the current DataFrame to a local file,
    encoding it in base64, and sending a PUT request to update the GitHub file.
    """
    st.write("DEBUG: Attempting to update mapping.xlsx on GitHub.")
    st.write("DEBUG: DataFrame shape before upload:", mapping_df.shape)

    github_token = st.secrets.get("github", {}).get("token", GITHUB_TOKEN)
    owner = st.secrets.get("github", {}).get("owner", GITHUB_OWNER)
    repo = st.secrets.get("github", {}).get("repo", GITHUB_REPO)
    file_path = st.secrets.get("github", {}).get("file_path", GITHUB_FILE_PATH)

    # Save DataFrame to a local file
    temp_file = MAPPING_FILE_LOCAL
    mapping_df.to_excel(temp_file, index=False, engine='openpyxl')

    # Encode local file in base64
    with open(temp_file, "rb") as f:
        content_bytes = f.read()
    encoded_content = base64.b64encode(content_bytes).decode("utf-8")

    # Get current file SHA (if it exists)
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    current_response = requests.get(url, headers=headers)
    sha = None
    if current_response.status_code == 200:
        sha = current_response.json().get("sha")
        st.write("DEBUG: Current file SHA:", sha)
    else:
        st.write("DEBUG: No existing file found. Creating a new one...")

    # Prepare data payload for updating/creating file on GitHub
    data = {
        "message": "Update mapping file via Streamlit app",
        "content": encoded_content
    }
    if sha:
        data["sha"] = sha

    update_response = requests.put(url, headers=headers, json=data)
    os.remove(temp_file)  # Clean up local file

    if update_response.status_code in [200, 201]:
        st.write("DEBUG: Update/creation successful:", update_response.status_code)
        return True
    else:
        st.error(f"Failed to update file on GitHub: {update_response.status_code} {update_response.text}")
        return False
