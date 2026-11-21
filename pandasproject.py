import pandas as pd

# 1. Load the CSV file
df = pd.read_csv("Titanic-Dataset.csv")

# 2. Pre-Cleaning Data Exploration (Minimum functions, maximum column insight)
print("--- Pre-Cleaning Dataset Overview ---")
df.info()
print("\nSummary Statistics (Numeric & Text):\n", df.describe(include="all").round(2))
print("\nDuplicate rows:", df.duplicated().sum())
print("\nFirst 5 rows:\n", df.head())

# 3. Clean column names and text spacing
df = df.drop(columns=["Unnamed: 0"], errors="ignore")
df.columns = df.columns.str.strip()

text_cols = df.select_dtypes(include="object").columns
for col in text_cols:
    df[col] = df[col].str.strip()

# 4. Remove duplicates (keeping first occurrence)
df = df.drop_duplicates(keep="first")

# 5. Fill missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Cabin"] = df["Cabin"].fillna("Unknown")

# 6. Keep only valid values
df = df[df["Age"] > 0]
df = df[df["Fare"] >= 0]
df = df[df["Survived"].isin([0, 1])]
df = df[df["Pclass"].isin([1, 2, 3])]

# 7. Type Conversion (from your notes)
df["Pclass"] = df["Pclass"].astype("category")
df["Age"] = df["Age"].astype("int64")

# 8. Feature Engineering (from your notes)
df["family_size"] = df["SibSp"] + df["Parch"] + 1
df.drop(columns=["SibSp", "Parch"], inplace=True)

# 9. Reset index and verify final dataset
df = df.reset_index(drop=True)

print("\n--- Final Dataset Summary ---")
print("After cleaning:", df.shape)
print("Final columns:", df.columns.tolist())
print("Data types after:\n", df.dtypes)
print("\nSample rows:\n", df[["Name", "Age", "Sex", "family_size"]].head())

# 10. Save the clean file
df.to_csv("cleaned_titanic.csv", index=False)
print("\nCleaning done! Saved to cleaned_titanic.csv")

# 11. Survival Analysis
print("\n--- Survival Analysis ---")

# Filter only passengers who survived (Survived == 1)
survivors = df[df["Survived"] == 1]

# Count how many males and females survived
survival_by_sex = survivors["Sex"].value_counts()
print("Survivors by Gender:\n", survival_by_sex)

# Optional: Print them individually as clean numbers
male_survived = len(survivors[survivors["Sex"] == "male"])
female_survived = len(survivors[survivors["Sex"] == "female"])
print(f"\nMales survived: {male_survived}")
print(f"Females survived: {female_survived}")

# Average age of survivors (vs. non-survivors)
avg_survivor_age = survivors["Age"].mean()
print(f"Average age of survivors: {avg_survivor_age:.2f} years")

# Bonus: Average age grouped by Survived (0 = Did not survive, 1 = Survived)
print("\nAverage age by survival status:\n", df.groupby("Survived")["Age"].mean())


# 12. Survival Rates by Embarked Port (S, C, Q)
print("\n--- Survival by Embarked Port ---")

# Survival rate as a percentage (0% to 100%)
embarked_rate = df.groupby("Embarked")["Survived"].mean() * 100
print("Survival Rate (%) by Port:\n", embarked_rate.round(2))

# Detailed breakdown: Total passengers, Survivors, and Survival %
embarked_summary = df.groupby("Embarked")["Survived"].agg(
    Total_Passengers="count",
    Survived_Count="sum",
    Survival_Rate_Pct=lambda x: round(x.mean() * 100, 2)
)
print("\nDetailed Breakdown:\n", embarked_summary)

# travelling with family  vs solo
print("\n--- Solo Travelers vs. With Family ---")
solo = df[df["family_size"] == 1]
with_family = df[df["family_size"] > 1]

print(f"Solo travelers survival rate: {solo['Survived'].mean() * 100:.2f}%")
print(f"With family survival rate: {with_family['Survived'].mean() * 100:.2f}%")

# Survival Rate: Elderly vs. Children vs. Middle-Aged Men vs. Middle-Aged Women
print("\n--- Survival Rate by Age & Gender Groups ---")

# 1. Children (Age < 15)
children = df[df["Age"] < 15]

# 2. Elderly (Age > 60)
elderly = df[df["Age"] > 60]

# 3. Middle-aged men (Age 30 to 45, male)
mid_men = df[(df["Age"] >= 30) & (df["Age"] <= 45) & (df["Sex"] == "male")]

# 4. Middle-aged women (Age 30 to 45, female)
mid_women = df[(df["Age"] >= 30) & (df["Age"] <= 45) & (df["Sex"] == "female")]

# Print survival rates (%) and counts
print(f"Children (< 15)             : {children['Survived'].mean() * 100:.2f}% ({children['Survived'].sum()}/{len(children)})")
print(f"Elderly (> 60)              : {elderly['Survived'].mean() * 100:.2f}% ({elderly['Survived'].sum()}/{len(elderly)})")
print(f"Middle-aged Men (30-45)     : {mid_men['Survived'].mean() * 100:.2f}% ({mid_men['Survived'].sum()}/{len(mid_men)})")
print(f"Middle-aged Women (30-45)   : {mid_women['Survived'].mean() * 100:.2f}% ({mid_women['Survived'].sum()}/{len(mid_women)})")

# Survival Rate: Family Size Above 3 vs. Below 3
print("\n--- Survival Rate by Family Size (Threshold: 3) ---")

# Check the actual average family size first
print(f"Average family size: {df['family_size'].mean():.2f}")

# Filter groups
above_3 = df[df["family_size"] > 3]
below_3 = df[df["family_size"] < 3]
equal_3 = df[df["family_size"] == 3]  # Optional: exact size 3

print(f"Family size > 3  : {above_3['Survived'].mean() * 100:.2f}% ({above_3['Survived'].sum()}/{len(above_3)})")
print(f"Family size < 3  : {below_3['Survived'].mean() * 100:.2f}% ({below_3['Survived'].sum()}/{len(below_3)})")
print(f"Family size == 3 : {equal_3['Survived'].mean() * 100:.2f}% ({equal_3['Survived'].sum()}/{len(equal_3)})")
