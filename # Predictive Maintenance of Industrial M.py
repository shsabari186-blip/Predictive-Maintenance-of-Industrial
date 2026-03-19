# Predictive Maintenance of Industrial Machines using Data Mining
# Author: Project Example
# Date: 2026

# -------------------------------
# Step 1: Import Required Libraries
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# Step 2: Create Sample Dataset
# -------------------------------
data = {
    "Temperature":[60,65,70,75,80,85,90,95,100,105],
    "Vibration":[20,25,30,35,40,45,50,55,60,65],
    "Pressure":[30,32,35,37,40,42,45,47,50,52],
    "Speed":[1000,1100,1200,1300,1400,1500,1600,1700,1800,1900],
    "Failure":[0,0,0,0,1,0,1,1,1,1]
}

df = pd.DataFrame(data)
print("=== Machine Dataset ===")
print(df)

# -------------------------------
# Step 3: Data Preprocessing
# -------------------------------
X = df[["Temperature","Vibration","Pressure","Speed"]]
y = df["Failure"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Step 4: Split Data (Train/Test)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# -------------------------------
# Step 5: Build ML Model
# -------------------------------
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)

# -------------------------------
# Step 6: Predict & Evaluate
# -------------------------------
predictions = model.predict(X_test)

print("\n=== Predicted Values ===")
print(predictions)

print("\n=== Actual Values ===")
print(y_test.values)

accuracy = accuracy_score(y_test,predictions)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test,predictions))

# -------------------------------
# Step 7: Visualization
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(df["Temperature"], label="Temperature", marker='o')
plt.plot(df["Vibration"], label="Vibration", marker='x')
plt.plot(df["Pressure"], label="Pressure", marker='s')
plt.title("Machine Sensor Data")
plt.xlabel("Machine Sample")
plt.ylabel("Sensor Values")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# Step 8: New Machine Prediction
# -------------------------------
def predict_machine(temp, vib, press, speed):
    new_data = [[temp, vib, press, speed]]
    new_scaled = scaler.transform(new_data)
    prediction = model.predict(new_scaled)
    if prediction[0]==1:
        print("Warning: Machine Failure Predicted")
    else:
        print("Machine Working Normally")

# Example Prediction
predict_machine(90,50,45,1600)

# -------------------------------
# Step 9: Save Dataset
# -------------------------------
df.to_csv("machine_data.csv", index=False)
print("\nDataset saved as 'machine_data.csv'")

# -------------------------------
# Step 10: Load Dataset
# -------------------------------
data_loaded = pd.read_csv("machine_data.csv")
print("\nLoaded Dataset:")
print(data_loaded)

print("\n=== Predictive Maintenance System Execution Completed ===")