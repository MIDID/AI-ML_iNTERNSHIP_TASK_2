# 🧪 Task 2: Exploratory Data Analysis (EDA) – Titanic Dataset

## 👩‍💻 Internship Task – AI & ML Track  
**Objective**: Perform Exploratory Data Analysis on the Titanic dataset using Python libraries like Pandas, Seaborn, and Matplotlib.

---

## 📁 Dataset
- **Source**: [Kaggle - Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- **File Used**: `Titanic-Dataset.csv`

---

## 🔧 Tools & Technologies Used
- Python 🐍
- Google Colab
- Libraries:
  - `pandas` for data manipulation
  - `seaborn`, `matplotlib` for visualizations
  - `numpy` for numerical ops

---

## 📊 What I Did

### 1. Data Inspection
- Loaded dataset using `pandas.read_csv()`
- Viewed structure with `.info()` and `.head()`
- Checked for null values

### 2. Descriptive Statistics
- Used `.describe()` for mean, std, min, etc.
- Analyzed missing values and data types

### 3. Visualizations
- Histogram for `Age` distribution
- Boxplot for `Fare` to identify outliers
- Correlation heatmap
- Pairplot to find relationships
- Countplots for survival based on `Sex` and `Pclass`

### 4. Inferences
- Females and 1st class passengers had higher survival rates
- Some skewness in Age and Fare
- `Pclass` and `Fare` mildly correlated with survival

---

## 📌 Key Visuals Summary

| Plot Type    | Feature(s)       | Insight                             |
|--------------|------------------|--------------------------------------|
| Histogram    | Age              | Age distribution is right-skewed     |
| Boxplot      | Fare             | Presence of fare outliers            |
| Heatmap      | Correlation      | Pclass, Fare vs Survived correlation |
| Countplot    | Sex vs Survived  | Females had higher survival          |
| KDE Plot     | Age vs Survived  | Slight advantage for young survivors|

---

## 🔢 Code Used

```python
# Install required libraries
!pip install seaborn pandas matplotlib --quiet

# 1. Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 2. Load dataset
from google.colab import files
uploaded = files.upload()
df = pd.read_csv('Titanic-Dataset.csv')
df.head()

# 3. Basic Info
df.info()
print(df.describe())
print(df.isnull().sum())

# 4. Visualizations
sns.histplot(df['Age'].dropna(), kde=True, color='skyblue')
plt.title("Age Distribution")
plt.show()

sns.boxplot(x=df['Fare'], color='orange')
plt.title("Fare Boxplot")
plt.show()

# 5. Correlation Matrix
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# 6. Pairplot
sns.pairplot(df[['Age', 'Fare', 'Survived']].dropna())
plt.suptitle("Pairplot of Age, Fare, and Survival", y=1.02)
plt.show()

# 7. Categorical Relationships
sns.countplot(data=df, x='Survived', hue='Sex')
plt.title("Survival by Gender")
plt.show()

sns.countplot(data=df, x='Survived', hue='Pclass')
plt.title("Survival by Passenger Class")
plt.show()

sns.kdeplot(df[df['Survived'] == 1]['Age'].dropna(), label='Survived', fill=True)
sns.kdeplot(df[df['Survived'] == 0]['Age'].dropna(), label='Not Survived', fill=True)
plt.title("Age Distribution by Survival")
plt.legend()
plt.show()
