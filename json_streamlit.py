import streamlit as st
import pandas as pd
import json

def extract_tables_from_csv(file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file)

    # Identify rows where the percentage of non-null values is greater than 70%
    non_null_counts = df.notna().sum(axis=1)
    total_columns = df.shape[1]
    non_null_percentage = (non_null_counts / total_columns) * 100
    rows_to_extract = df[non_null_percentage > 70]

    # Check if any rows were extracted
    if rows_to_extract.empty:
        st.warning("No rows with more than 70% non-null values found.")
        return pd.DataFrame()

    # Set the first row of the extracted rows as the header
    rows_to_extract.columns = rows_to_extract.iloc[0]  # Set the first row as header
    rows_to_extract = rows_to_extract[1:]  # Remove the first row from the data

    # Rename columns with NaN names
    rows_to_extract.columns = [
        f"Unnamed{i+1}" if pd.isna(col) else col
        for i, col in enumerate(rows_to_extract.columns)
    ]

    # Ensure unique column names by appending a suffix to duplicates
    columns = pd.Series(rows_to_extract.columns)
    for dup in columns[columns.duplicated()].unique():
        columns[columns[columns == dup].index.values.tolist()] = [
            f"{dup}_{i+1}" if i != 0 else dup
            for i in range(sum(columns == dup))
        ]
    rows_to_extract.columns = columns  # Update the DataFrame columns

    # Define possible package type columns
    package_type_cols = ['CTNS', 'QTN', 'QTY/CTN', 'Pallet', 'Boxes', 'Euro Pallet', 'Bags', 'Cases']

    # Check if any of the package type columns exist and have non-null values
    if any(col in rows_to_extract.columns and rows_to_extract[col].notna().any() for col in package_type_cols):
        # Create a new column "Package Type" and populate it with matching column names
        rows_to_extract['Package Type'] = rows_to_extract.apply(
            lambda row: next((col for col in package_type_cols if col in rows_to_extract.columns and pd.notna(row[col])), None), axis=1
        )

        # Drop the original columns that were used to populate "Package Type"
        rows_to_extract = rows_to_extract.drop(columns=[col for col in package_type_cols if col in rows_to_extract.columns])
    else:
        st.info("No package type columns found with non-null values. 'Package Type' column will not be created.")

    return rows_to_extract

def convert_to_json_hierarchy(dataframe):
    # Convert the DataFrame to a JSON format
    json_hierarchy = dataframe.to_json(orient='records', indent=2)
    return json_hierarchy

def convert_json_to_dataframe(json_data):
    # Convert the JSON string back to a DataFrame
    dataframe = pd.read_json(json_data)
    return dataframe

# Streamlit app code
st.title("CSV to JSON Converter")

# File uploader
file = st.file_uploader("Upload a CSV file", type=["csv"])

if file is not None:
    extracted_rows = extract_tables_from_csv(file)
    
    if not extracted_rows.empty:
        # Convert the extracted rows to JSON hierarchical structure
        json_output = convert_to_json_hierarchy(extracted_rows)
        
        # Display the JSON output
        st.subheader("Extracted Data in JSON Format")
        st.json(json_output)

        # Convert the JSON back to DataFrame
        dataframe_from_json = convert_json_to_dataframe(json_output)

        # Display the DataFrame output
        st.subheader("Extracted Table Created from JSON")
        st.dataframe(dataframe_from_json)
    else:
        st.info("No valid data to display.")
