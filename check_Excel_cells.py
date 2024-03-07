import os
import pandas as pd

def count_unused_cells(file_path):
    try:
        # Read the Excel file into a DataFrame
        df = pd.read_excel(file_path)

        # Count the number of empty cells with formatting
        unused_cell_count = df.isna().sum().sum()

        return unused_cell_count
    except ValueError:
        return -1


def count_unused_cells_in_directory(directory):
    total_unused_cells = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.xlsx'):
                file_path = os.path.join(root, file)
                unused_cells = count_unused_cells(file_path)

                if unused_cells > 10000:
                    print(f"File: {file_path}, Unused Cells: {unused_cells}")
                    total_unused_cells += unused_cells

    print(f"Total Unused Cells in Directory: {total_unused_cells}")

# Usage
count_unused_cells_in_directory(r'C:\Users\Lenovo\OneDrive - MONEYTIME VENTURES\Portfolio')
