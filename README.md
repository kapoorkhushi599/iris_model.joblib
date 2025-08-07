# iris_model.joblib# iris_model.joblib

This repository contains a trained Random Forest model for classifying the famous Iris dataset. The model is saved as `iris_model.joblib` and can be loaded for inference in your own Python projects.

## How was the model created?

The model was trained using the following steps:

1. **Importing Libraries:**  
   The script uses `scikit-learn` for dataset loading, splitting, and model training, and `joblib` for saving the model.

2. **Loading the Dataset:**  
   The Iris dataset is loaded from scikit-learn's built-in datasets.

3. **Splitting the Data:**  
   The data is split into training and testing sets (80% train, 20% test).

4. **Training the Model:**  
   A Random Forest Classifier is trained on the training data.

5. **Saving the Model:**  
   The trained model is saved as `iris_model.joblib`.

#### Example Code Used

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, 'iris_model.joblib')
```

## How to Use

1. **Install dependencies:**
   ```
   pip install scikit-learn joblib
   ```

2. **Load the model in your Python code:**
   ```python
   import joblib
   model = joblib.load('iris_model.joblib')
   ```

3. **Make predictions:**
   ```python
   # X_new should be a 2D array of shape (n_samples, 4)
   predictions = model.predict(X_new)
   ```

## License

This project is open source and free to use.
