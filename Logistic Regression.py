from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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

#Transform values of target variable to binary
data['FSCDVHF2'] = data['FSCDVHF2'].apply(lambda x: 1 if x in [2, 3] else x)

#Logistic Regression
X = data.drop(columns=['FSCDVHF2', 'WTS_M'])
y = data['FSCDVHF2']
weights = data['WTS_M']

#Split data into training and testing sets
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
    X, y, weights, test_size=0.2, random_state=42
)

#Initialize and train logistic regression model
model = LogisticRegression(max_iter=5000)  # Increase max_iter if necessary
model.fit(X_train, y_train, sample_weight=weights_train)

#Make predictions
y_pred = model.predict(X_test)

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

#After fitting the model
intercept = model.intercept_[0]
coefficients = model.coef_[0]
feature_names = X.columns

#Print the prediction equation
print("Logit(p) = {:.4f}".format(intercept), end="")
for coef, feature in zip(coefficients, feature_names):
    print(" + ({:.4f} * {})".format(coef, feature), end="")

#Call the file path from function "select_file"
file_path_inuit = select_file()

#Load the selected file
data_inuit = pd.read_csv(file_path_inuit)

#Transform values of target variable to binary
data_inuit['FSCDVHF2'] = data_inuit['FSCDVHF2'].apply(lambda x: 1 if x in [2, 3] else x)

#Logistic Regression
X_inuit = data_inuit.drop(columns=['FSCDVHF2', 'WTS_M'])
y_inuit = data_inuit['FSCDVHF2']
weights_inuit = data_inuit['WTS_M']

#Split data into training and testing sets
X_train_inuit, X_test_inuit, y_train_inuit, y_test_inuit, weights_train_inuit, weights_test_inuit = train_test_split(
    X_inuit, y_inuit, weights_inuit, test_size=0.2, random_state=42
)

#Ensure all features from training are present in the new dataset
missing_features = set(X_train.columns) - set(X_train_inuit.columns)
for feature in missing_features:
    X_test_inuit[feature] = 0  # Fill missing features with a default value, e.g., 0
    X_train_inuit[feature] = 0

#Reorder columns to match the training dataset
X_test_inuit = X_test_inuit[X_train.columns]
X_train_inuit = X_test_inuit[X_train.columns]

#Make predictions on the Inuit dataset
y_pred_inuit = model.predict(X_test_inuit)

#Evaluate the model on the Inuit dataset
accuracy_inuit = accuracy_score(y_test_inuit, y_pred_inuit)
conf_matrix_inuit = confusion_matrix(y_test_inuit, y_pred_inuit)
class_report_inuit = classification_report(y_test_inuit, y_pred_inuit, zero_division=1)

print(f"\nInuit Dataset Evaluation:")
print(f"Accuracy: {accuracy_inuit:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix_inuit)
print("\nClassification Report:")
print(class_report_inuit)

#Add a new variable set to 1 for Inuit and 0 for the rest
X_train['new_variable'] = 0
X_test['new_variable'] = 0
X_train_inuit['new_variable'] = 1
X_test_inuit['new_variable'] = 1

#Combine both training datasets
combined_train_x = pd.concat([X_train, X_train_inuit], ignore_index=True)
combined_train_y = pd.concat([y_train, y_train_inuit], ignore_index=True)
combined_train_weights = pd.concat([weights_train, weights_train_inuit], ignore_index=True)

#Initialize and train logistic regression model on the combined dataset
model = LogisticRegression(max_iter=5000)  # Increase max_iter if necessary
model.fit(combined_train_x, combined_train_y, sample_weight=combined_train_weights)

#Combine both testing datasets
combined_test_x = pd.concat([X_test, X_test_inuit], ignore_index=True)
combined_test_y = pd.concat([y_test, y_test_inuit], ignore_index=True)
combined_test_weights = pd.concat([weights_test, weights_test_inuit], ignore_index=True)

#Make predictions on the combined dataset
y_pred_combined = model.predict(combined_test_x)

#Evaluate the model on the combined dataset
combined_accuracy = accuracy_score(combined_test_y, y_pred_combined)
combined_conf_matrix = confusion_matrix(combined_test_y, y_pred_combined)
combined_class_report = classification_report(combined_test_y, y_pred_combined, zero_division=1)

print(f"\nInuit Dataset Evaluation:")
print(f"Accuracy: {combined_accuracy:.2f}")
print("\nConfusion Matrix:")
print(combined_conf_matrix)
print("\nClassification Report:")
print(combined_class_report)