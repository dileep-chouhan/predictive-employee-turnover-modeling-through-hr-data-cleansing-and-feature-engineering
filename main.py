import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
n_employees = 500
data = {
    'Age': np.random.randint(20, 60, size=n_employees),
    'YearsExperience': np.random.randint(0, 30, size=n_employees),
    'SatisfactionScore': np.random.randint(1, 11, size=n_employees),
    'WorkLifeBalance': np.random.randint(1, 11, size=n_employees),
    'PromotionLast5Years': np.random.randint(0, 2, size=n_employees), # 0 or 1
    'Salary': np.random.randint(40000, 150000, size=n_employees),
    'Left': np.random.randint(0, 2, size=n_employees) # 0: stayed, 1: left
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Feature Engineering ---
#Handle Missing Values (Synthetic data, so no missing values here, but demonstrating best practice)
#df.fillna(df.mean(), inplace=True) #replace with mean for numerical columns
#Feature Engineering: Create a composite score
df['OverallScore'] = df['SatisfactionScore'] + df['WorkLifeBalance']
# --- 3. Data Analysis and Visualization ---
# Descriptive Statistics
print("Descriptive Statistics:")
print(df.describe())
#Correlation Matrix
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
print("Plot saved to correlation_matrix.png")
#Distribution of Employees who Left
plt.figure(figsize=(8,6))
sns.countplot(x='Left', data=df)
plt.title('Distribution of Employees who Left')
plt.savefig('employee_left_distribution.png')
print("Plot saved to employee_left_distribution.png")
# --- 4. Predictive Modeling ---
X = df.drop('Left', axis=1)
y = df['Left']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
#Save the model (optional)
#import joblib
#joblib.dump(model, 'employee_turnover_model.pkl')