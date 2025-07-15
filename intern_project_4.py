import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("road_safety.csv")
numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = df[numeric_cols].replace(-1, pd.NA)
gender_map = {1: 'Male', 2: 'Female'}
df['sex_of_driver'] = df['sex_of_driver'].map(gender_map)

x = df['age_of_driver'].dropna()
plt.figure(figsize=(10, 5))
sns.histplot(x, bins=30, kde=True, color='skyblue')
plt.title("Distribution of Driver Age")
plt.xlabel("Age of Driver")
plt.ylabel("Number of Drivers")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='sex_of_driver', palette='Set2', order=['Male', 'Female'])
plt.title("Gender of Drivers Involved in Accidents")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

veh = df['vehicle_type'].value_counts().head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=veh.index.astype(str), y=veh.values, palette='viridis')
plt.title("Top 10 Vehicle Types Involved in Accidents")
plt.xlabel("Vehicle Type Code")
plt.ylabel("Number of Vehicles")
plt.show()

df['journey_purpose_of_driver'] = df['journey_purpose_of_driver'].fillna(0)

purpose_map = {
    1: 'Journey to/from work',
    2: 'Taking pupil to/from school',
    3: 'Other work',
    4: 'Personal business',
    5: 'Leisure',
    6: 'Going home',
    7: 'Other',
    0: 'Unknown'
}
df['journey_purpose_label'] = df['journey_purpose_of_driver'].map(purpose_map)
plt.figure(figsize=(10, 5))
sns.countplot(x='journey_purpose_label', data=df, palette='coolwarm')
plt.title("Journey Purpose of Drivers")
plt.xlabel("Journey Purpose")
plt.ylabel("Count")
plt.show()

