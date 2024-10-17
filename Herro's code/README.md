## Steps to Run the Project

### 1. Data Preparation (`comp90089a3_dataprocessing.ipynb`)

- First, run the Jupyter notebook `comp90089a3_dataprocessing.ipynb` to process the initial dataset.
- This will prepare the data, and the output will be saved as `final_data_with_icd.csv`.

### 2. ICD Code Processing (`process_icdcode.py`)

- Due to memory limitations in Colab, the next step needs to be executed locally using PyCharm.
- Run the `process_icdcode.py` script. This script cuts the `icd_code` and filters out the ICD codes based on their usage frequency.
- The output of this step is saved as `final_data.csv`.

### 3. Adding Long Titles (`add_longtitle.ipynb`)

- Run the Jupyter notebook `add_longtitle.ipynb` to add long titles to the ICD codes for better matching.
- This step will produce the file `final_data_with_long_titles.csv`, which is the final dataset.

### 4. Model Training (`XGBoost.py`)

- Finally, run the `XGBoost.py` script to train the XGBoost model using the prepared dataset.
- The results of the model training will be stored in the `result` folder.

## Files and Outputs

- `comp90089a3_dataprocessing.ipynb`: Jupyter notebook for data preparation.
- `process_icdcode.py`: Python script to process ICD codes and filter based on usage frequency.
- `add_longtitle.ipynb`: Jupyter notebook to add long titles to ICD codes.
- `XGBoost.py`: Python script to run the XGBoost model and store the results.
- Output files:
  - `final_data_with_icd.csv`: Data prepared from the first step.
  - `final_data.csv`: Filtered ICD code data.
  - `final_data_with_long_titles.csv`: Final dataset with long titles.
  - `result` folder: Stores model training results.

## Prerequisites

Ensure you have the necessary libraries installed, including:
- XGBoost
- pandas
- numpy
- Any additional libraries as listed in the respective notebooks/scripts.

## Notes

- Memory-intensive processes may need to be run locally due to limitations in online environments like Colab.
