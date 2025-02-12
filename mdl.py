import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Load the dataset
df = pd.read_csv(r'D:\HI448116_Santosh_Karpe\FY25\DOCS\III\Ass\ASA - SK\Dataset and Objective-20250118T164825Z-001\Dataset and Objective\temperature_data (1).csv')

# Prepare features and target variable
X = df.drop(columns='motor_speed')  # Features
y = df['motor_speed']  # Target variable

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
svr_classifier = SVR(kernel='linear')
svr_classifier.fit(X_train_scaled, y_train)

# Save the trained model and scaler as pickle files
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svr_classifier, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and Scaler saved successfully!")
