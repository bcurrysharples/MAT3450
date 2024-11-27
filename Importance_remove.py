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

#List of variables
initial_features = [
    'DHHDGHSZ', 'MAC_010', 'GEN_005', 'GEN_015', 'GEN_030', 'SMK_005',
    'DHHGAGE', 'ALCDVTTM', 'CAN_015', 'PAA_030', 'SBE_005', 'SWL_015',
    'SWL_040', 'SPS_050', 'UCN_005', 'INS_005', 'INCDGHH', 'WTS_M', 'FSCDVHF2'
]

#Filtering dataset with the selected variables
filtered_data = data[initial_features]

#Function to manually choose variables to remove
def manual_feature_removal(features):
    print("\nAvailable features from the predefined list:\n", features)
    print("\nPlease enter the names of the features you want to remove, separated by commas:")
    features_to_remove = input("Enter features to remove: ").strip().split(",")
    features_to_remove = [feature.strip() for feature in features_to_remove if feature.strip() in features]
    remaining_features = [feature for feature in features if feature not in features_to_remove]
    print("\nRemaining features after removal:\n", remaining_features)
    return remaining_features

#Call variables from function "manual_feature_removal"
final_features = manual_feature_removal(initial_features)

#Categorize variables
if 'FSCDVHF2' in final_features and 'WTS_M' in final_features:
    X = filtered_data[final_features].drop(columns=['FSCDVHF2', 'WTS_M'])
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
