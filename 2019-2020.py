import pandas as pd

#CSV file
df = pd.read_csv("C:/Users/dave_/OneDrive/Bureau/Food Insecurity/2017-2018/escc-82M0013-F-2017-2018-composante-annuelle_F3 (1).csv")

#Column and value for filters
first_column_to_filter = "GEO_PRV"
first_value_to_keep = [60, 61, 62]
second_column_to_filter = "SDC_015"
second_value_to_keep = 1

#Filtering process
filtered_df = df[(df[first_column_to_filter] == first_value_to_keep) & (df[second_column_to_filter] == second_value_to_keep)]
non_filtered_df = df[(df[first_column_to_filter] != first_value_to_keep) & (df[second_column_to_filter] != second_value_to_keep)]
#Observed variable
target_column = 'FSCDVHF2'  # Replace with the actual column name for y

#Rearrange columns to put observed column at the end
columns = [col for col in filtered_df.columns if col != target_column]  # All columns except 'y'
columns.append(target_column)  # Add 'y' at the end
columns_1 = [col for col in non_filtered_df.columns if col != target_column]  # All columns except 'y'
columns_1.append(target_column)  # Add 'y' at the end

#Reordering data frame
filtered_df = filtered_df[columns]
non_filtered_df = non_filtered_df[columns_1]

#Create new CSV with filtered data frame
filtered_df.to_csv("Inuit_rearranged_file_2019_2020.csv", index=False)
non_filtered_df.to_csv("Rest_rearranged_file_2019_2020.csv", index=False)