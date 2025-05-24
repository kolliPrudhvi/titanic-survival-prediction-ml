#!/usr/bin/env python
# coding: utf-8

# In[44]:


# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Step 2: Load data
df = pd.read_csv("train.csv")

# Step 3: Drop column with too many missing values
df.drop('Cabin', axis=1, inplace=True, errors='ignore')

# Step 4: Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Step 5: Encode categorical variables

# Encode 'Sex' with map
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Encode 'Embarked' with one-hot encoding
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Step 6: Define features and target
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
X = df[features]
y = df['Survived']

# Step 7: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 9: Evaluate
print("Training Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", model.score(X_test, y_test))


# In[45]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[46]:


y_pred = model.predict(X_test)


# In[47]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test data: {accuracy:.4f}")


# In[48]:


cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


# In[49]:


report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)


# In[ ]:




