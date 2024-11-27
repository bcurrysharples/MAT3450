from sklearn.ensemble import RandomForestClassifier
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

# Load the selected file
data = pd.read_csv(file_path)

#Function to manually choose variables to keep
def manual_feature_selection(columns):
    print("\nAvailable features:\n", columns)
    print("\nPlease enter the names of the features you want to keep, separated by commas:")
    selected_features = input("Enter features: ").strip().split(",")
    selected_features = [feature.strip() for feature in selected_features if feature.strip() in columns]
    print("\nSelected features:\n", selected_features)
    return selected_features

#Call variables from function
features_to_keep = manual_feature_selection(data.columns)

#Filtering dataset with the selected variables
filtered_data = data[features_to_keep]
filtered_data.to_csv("Inuit_reduced_variables_2019_2020.csv", index=False)

#Categorize variables
if 'FSCDVHF2' in features_to_keep and 'WTS_M' in features_to_keep:
    X = filtered_data.drop(columns=['FSCDVHF2', 'WTS_M'])
    y = filtered_data['FSCDVHF2']
    sample_weights = filtered_data['WTS_M']
else:
    print("\nError: 'FSCDVHF2' and 'WTS_M' must be included for training.")
    exit()

#Train on the entire dataset
model = RandomForestClassifier(random_state=42)
model.fit(X, y, sample_weight=sample_weights)

#Extract "feature importance" scores
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

#Display results
most_important_vars = importances[importances > 0].index.tolist()
print("\nFeature Importance:\n", importances)
print("\nMost Important Variables:\n", most_important_vars)
