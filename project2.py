print("==================================")
print("Project 2: Data Classification Using AI")
print("Internship: DecodeLabs") 
print("===================================")

# Import required libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the Iris dataset
iris = load_iris()

# Create a DataFrame
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data["Flower Type"] = iris.target

# Display first five records
print("First Five Records:")
print(data.head())

# Display dataset information
print("\nDataset Shape:", data.shape)
print("\nMissing Values:")
print(data.isnull().sum())

# Step 2: Define Features and Target
X = iris.data
y = iris.target

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create the Decision Tree model
model = DecisionTreeClassifier(random_state=42)

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Make predictions
predictions = model.predict(X_test)

# Step 7: Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# Step 8: Predict a new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("\nNew Flower Details:")
print("Sepal Length =", new_flower[0][0])
print("Sepal Width  =", new_flower[0][1])
print("Petal Length =", new_flower[0][2])
print("Petal Width  =", new_flower[0][3])

print("\nPredicted Flower Name:", iris.target_names[prediction[0]])

print("\nProject Completed Successfully!")