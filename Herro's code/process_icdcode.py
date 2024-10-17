import pandas as pd
import gc

# Load data from the CSV file
final_data_with_icd = pd.read_csv('final_data_with_icd.csv')

# Ensure 'previous_icd_codes' is treated as a string
final_data_with_icd['previous_icd_codes'] = final_data_with_icd['previous_icd_codes'].fillna('').astype(str)

# Create a list of all unique ICD codes by splitting the 'previous_icd_codes' column
all_icd_codes = set(
    [icd for sublist in final_data_with_icd['previous_icd_codes'].apply(lambda x: [icd.strip() for icd in x.split(', ')]) for icd in sublist if icd])

# Set the chunk size to process the data in smaller batches
chunk_size = 1000

# Initialize a list to store results from each chunk
icd_chunks = []

# Process the data in chunks to avoid memory overflow
for start in range(0, final_data_with_icd.shape[0], chunk_size):
    # Select the current chunk of data
    chunk = final_data_with_icd.iloc[start:start + chunk_size]

    # Create a DataFrame with zeros for each unique ICD code in the current chunk
    icd_columns_chunk = pd.DataFrame(0, index=chunk.index, columns=sorted(all_icd_codes))

    # Loop through each row in 'previous_icd_codes' in the current chunk
    for i, icd_list in enumerate(chunk['previous_icd_codes'].apply(lambda x: [icd.strip() for icd in x.split(', ')])):
        for icd in icd_list:
            if icd in icd_columns_chunk.columns:
                icd_columns_chunk.at[chunk.index[i], icd] = 1

    # Append the processed chunk to the list
    icd_chunks.append(icd_columns_chunk)

# Concatenate all chunks back together
icd_columns = pd.concat(icd_chunks, axis=0)

# Merge the ICD columns back to the original dataframe
final_data_with_icd_expanded = pd.concat([final_data_with_icd, icd_columns], axis=1)

# Clean up unused variables to free up memory
del icd_chunks, icd_columns_chunk
gc.collect()

# Define lists of unrelated ICD codes to be removed
not_related_disease_icd = ['Y92230', 'Y92239', 'Y929', 'Z23', 'Z66', 'E8770', 'Z9981', 'Y831', 'R531', 'W1830XA',
                           'Z9181', 'W19XXXA', 'Z006', 'W010XXA', 'Y838', 'Z781', 'Y92009', 'Z79899', 'V4986',
                           '4414', 'V1581', 'E8889', 'E8859', 'V1588', 'V153', 'V4501', 'E8798', '4168']
not_in_d_icd = ['E780', 'I482', 'I272', 'M4806']
H_icd = ['I10']

# Merge all unrelated ICD codes into one set
not_used_icd = set(not_related_disease_icd + not_in_d_icd + H_icd)

# Identify and remove unnecessary columns
not_used_cols = list(set(final_data_with_icd_expanded.columns) & not_used_icd)
final_data_with_icd_expanded = final_data_with_icd_expanded.drop(columns=not_used_cols)

# Only perform frequency calculations on numerical columns (ICD columns)
icd_columns_only = final_data_with_icd_expanded.iloc[:, 9:]  # ICD columns start from the 9th column
icd_frequencies = icd_columns_only.mean(axis=0)

# Filter for common ICD codes with a frequency >= 0.01
frequent_icd_codes = icd_frequencies[icd_frequencies >= 0.01].index

# Filter the data based on common ICD codes
final_data_frequent_icds = final_data_with_icd_expanded[
    ['subject_id', 'hadm_id', 'gender', 'age', 'race', 'classification'] + list(frequent_icd_codes)]

# Display the shape of the final filtered dataframe
print(final_data_frequent_icds.shape)

# Display the first 10 rows of the final filtered dataframe
print(final_data_frequent_icds.head(10))

# Save the processed file
final_data_frequent_icds.to_csv('final_data.csv', index=False)

# Confirm the data has been successfully saved
print("Data saved to final_data.csv")
