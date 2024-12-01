from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename

#Function to select CSV file
def select_file():
    Tk().withdraw()
    print("Please select the CSV file you want to load:")
    file_path = askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        print(f"File selected: {file_path}")
    else:
        print("No file selected. Exiting.")
        exit()
    return file_path

#Call the file path from function "select_file"
file_path = select_file()

#Load the selected file
data = pd.read_csv(file_path)

#List of variables
features_to_keep = [
    'DHHDGHSZ', 'MAC_010', 'GEN_005', 'GEN_015', 'GEN_030', 'SMK_005',
    'DHHGAGE', 'ALCDVTTM', 'INCDGHH', 'WTS_M', 'FSCDVHF2'
]

#Filtering dataset with the selected variables
filtered_data = data[features_to_keep]

#Separate the weight and target variables
weights = filtered_data['WTS_M']
target = filtered_data['FSCDVHF2']

#Exclude 'WTS_M' and 'FSCDVHF2' for transformation
X = filtered_data.drop(columns=['FSCDVHF2', 'WTS_M'])

#Treat all variables in X as categorical
categorical_columns = X.columns.tolist()

#Apply one-hot encoding to all columns in X
transformer = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False), categorical_columns)
    ]
)

#Transform the data
X_transformed = transformer.fit_transform(X)

#Get the correct feature names after encoding
encoded_columns = transformer.named_transformers_['onehot'].get_feature_names_out(categorical_columns)

#Convert the transformed data to a DataFrame
X_transformed_df = pd.DataFrame(X_transformed, columns=encoded_columns)

#Add the weight and target variables as the last columns
X_transformed_df['WTS_M'] = weights.values
X_transformed_df['FSCDVHF2'] = target.values

#Save the transformed dataset to a CSV file
output_path = "One_Hot_Rest_2019_2020.csv"
X_transformed_df.to_csv(output_path, index=False)

