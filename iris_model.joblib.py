# Step 1: Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Step 2: Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 3: Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 5: Save the model using joblib
joblib.dump(model, 'iris_model.joblib')

print("✅ Model trained and saved as 'iris_model.joblib'")