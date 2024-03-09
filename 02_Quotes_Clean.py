import pandas as pd

df = pd.read_csv('crypto_data.csv')

# Pivot the DataFrame
df = df.pivot(index='Symbol', columns='Date', values='Close')
df = df.dropna()
df.to_csv('crypto_data_clean.csv', header=True)

#
# new_column_names = df.iloc[0]
# print(new_column_names)
# df.columns = new_column_names
#
#
# # Step 1: Count NaNs in each row
# df['NaN_Count'] = df.isnull().sum(axis=1)
#
# # Step 2: Calculate total columns (subtracting the one we just added for NaN counts)
# total_columns = len(df.columns) - 1
#
# # Step 3: Filter rows with at least one missing value
# rows_with_missing = df[df['NaN_Count'] > 0]
#
# # Step 4: Calculate percentage of NaNs in each row
# rows_with_missing['NaN_Percentage'] = (rows_with_missing['NaN_Count'] / total_columns) * 100
# # print(rows_with_missing)
#
#
#
# # Check if there are any missing values in the DataFrame
# has_missing_values = df.isnull().values.any()
# print(f"Are there any missing values? {has_missing_values}")
#
# # # Select the first row of the DataFrame
# # first_row_df = pd.DataFrame([df.iloc[0]])
# # # Save the first row as a CSV file, where the first row will act as the header
# # first_row_df.to_csv('crypto_data_clean.csv', index=False, header=True)


# df.to_csv('crypto_data_clean.csv', header=True)
# # Count missing values in each column
# # missing_values_count = df.isnull().sum().sum()
# missing_values_count = df.isnull().sum(axis=1)
# print(missing_values_count)
# # Find rows with missing values
# rows_with_missing_values = df[df.isnull().any(axis=1)]
# print(rows_with_missing_values)
# Display the pivoted DataFrame
# print(df)
# Replace missing values with the preceding non-missing value in the DataFrame
# df = df.fillna(method='ffill')
# df = df.fillna(method='bfill')