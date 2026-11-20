# pandas-project-file-cleaning

# Titanic Dataset Cleaner 🚢

A simple Python data-cleaning script built with `pandas` to clean and preprocess the Titanic dataset (`Titanic-Dataset.csv`).

## What This Script Does
- **Removes unnecessary columns & duplicates:** Drops extra index columns (`Unnamed: 0`) and duplicate rows.
- **Cleans text formatting:** Strips leading and trailing spaces from column names and string columns.
- **Handles missing values (without losing rows):**
  - Fills missing `Age` values with the median age.
  - Fills missing `Embarked` values with `"S"` (Southampton).
  - Fills missing `Cabin` values with `"Unknown"`.
- **Validates data:** Filters out invalid entries for `Age`, `Fare`, `Survived`, and `Pclass`.
- **Exports clean data:** Resets row indices and saves the cleaned dataset as `cleaned_titanic.csv`.

## Requirements
- Python 3.x
- pandas (`pip install pandas`)

## How to Run
1. Place `Titanic-Dataset.csv` and `pandasproject.py` in the same folder.
2. Run the script:
   ```bash
   python pandasproject.py
   ```
3. The cleaned output will be saved as `cleaned_titanic.csv` (891 rows, 0 missing values).
