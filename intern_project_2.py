import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Titanic_dataset.csv')

data = df.copy()


data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
data['AgeGroup'] = data['AgeGroup'].map({'Child': 0, 'Teenager': 1, 'Adult': 2, 'Senior': 3})


plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
sns.barplot(x='Sex', y='Survived', data=data, palette='Set1')
plt.title('Survival Rate by Sex')
plt.xlabel('Sex')
plt.ylabel('Survival Rate')

plt.subplot(2, 2, 2)
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5, linecolor='white', fmt='.2f')
plt.title('Correlation Heatmap')

plt.subplot(2, 2, 3)
sns.violinplot(x='Pclass', y='Survived', data=data, palette='Set2', inner='quartile')
plt.title('Survival Distribution by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')

plt.subplot(2, 2, 4)
data['Age'] = data['Age'].fillna(data['Age'].median())

def get_age_group(age):
    if age < 13:
        return 0  # Child
    elif age < 20:
        return 1  # Teenager
    elif age < 60:
        return 2  # Adult
    else:
        return 3  # Senior

data['AgeGroup'] = data['Age'].apply(get_age_group)
agegroup_counts = data['AgeGroup'].value_counts().sort_index()

agegroup_map = {0: 'Child', 1: 'Teenager', 2: 'Adult', 3: 'Senior'}
labels = [agegroup_map[i] for i in agegroup_counts.index]
plt.pie(agegroup_counts,labels=labels,autopct='%1.1f%%',colors=sns.color_palette('Set3'))
plt.title('Passenger Distribution by Age Group')
plt.tight_layout()
plt.show()

gender_counts = data['Sex'].value_counts()
plt.pie(gender_counts, labels=['Male', 'Female'], autopct='%1.1f%%', startangle=90,
    colors=['lightblue', 'pink'],
    wedgeprops={'width': 0.4})
plt.title('Gender Distribution (Donut Chart)')
plt.show()