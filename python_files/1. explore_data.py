# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Explore the Midwest Survey dataset
#
# In this notebook, we will explore the **Midwest Survey** dataset from [skrub](https://skrub-data.org/).
#
# This dataset contains survey responses from people across the United States,
# asking them about their perception of the Midwest region.
#
# The goal is to predict the **Census Region** where a respondent lives,
# based on their survey answers.

# %% [markdown]
# ## Load the dataset

# %%
from skrub.datasets import fetch_midwest_survey

dataset = fetch_midwest_survey()

# X contains the features (the survey answers)
X = dataset.X
# y contains the target (the Census Region)
y = dataset.y

# %% [markdown]
# ## Question 1: How many examples are there in the dataset?
#
print("Question 1")

# Use the `.shape` attribute to find out the number of rows and columns.

# %%
# Display the number of rows and columns
print("Number of examples:", X.shape[0])
print("Number of features:", X.shape[1])
print()
# %%
# You can also look at the first few rows of the dataset
X.head()

# %% [markdown]
# ## Question 2: What is the distribution of the target?
#
print("Question 2")

# The target variable `y` tells us the Census Region of each respondent.
# Let's see how many respondents belong to each region.

# %%
# Count how many respondents belong to each region
print("Target distribution:")
print(y.value_counts())

# Visualize the target distribution with a bar plot
# hint: use barh
import matplotlib.pyplot as plt

y.value_counts().plot.barh()
plt.title("Distribution of Census Region")
plt.xlabel("Number of respondents")
plt.tight_layout()
plt.show()

# %% [markdown]
# Is the target balanced (roughly the same number of examples per class) or imbalanced?

##La variable cible est déséquilibrée, car certaines régions (par exemple East North Central) comptent beaucoup plus de répondants que d’autres (comme New England).

# %% [markdown]
# ## Question 3: What are the features that can be used to predict the target?
#

print("Question 3")

# Let's look at the column names and their data types.

# %%
# List all column names
print("Feature names:")
print(list(X.columns))
print("Total number of features:", X.shape[1])
print()
# %%
# Show data types for each column
print("Data types:")
print(X.dtypes)
print()
# %% [markdown]
# How many features are numerical? How many are categorical (text)?
num_features = X.select_dtypes(include="number").shape[1]
cat_features = X.select_dtypes(exclude="number").shape[1]

print("Number of numerical features:", num_features)
print("Number of categorical (text) features:", cat_features)
print()
# %%


# %%
from skrub import TableReport
TableReport(X)

# %% [markdown]
# ## Question 4: Are there any missing values in the dataset?
#

print("Question 4")

# Missing values can cause problems for machine learning models.
# Let's check if there are any.

# %%
# Check for NaN missing values
missing = X.isna().sum()
print("Missing values per column:")
print(missing[missing > 0])
print()
# %% [markdown]
# Missing values can sometimes be encoded differently. Let's look at some columns more closely.

# %%
# Look at unique values for the Household_Income column
# #X["Household_Income"].??
# %%
print("Unique values in Household_Income:")
print(X["Household_Income"].unique())
print()
# Look at unique values for the Education column
print("Unique values in Education:")
print(X["Education"].unique())
print()
# %% [markdown]
# Do you see a special value that could represent missing data?
##Il n’y a pas de valeurs NaN dans le dataset. Cependant, certaines catégories comme « Prefer not to answer » (ou des réponses similaires) peuvent correspondre à des valeurs manquantes implicites.
# %% [markdown]
# ## Question 5: What is the most common answer to "How much do you personally identify as a Midwesterner"?
#

print("Question 5")

# Let's explore this important feature.

# %%
# TODO: display the value counts for the column
# "How_much_do_you_personally_identify_as_a_Midwesterner"
col = "How_much_do_you_personally_identify_as_a_Midwesterner"

counts = X[col].value_counts(dropna=False)

print("Value counts:")
print(counts)
print("Most common answer:", counts.idxmax())
print()
# %%
# TODO: make a bar plot of the results
import matplotlib.pyplot as plt

counts.plot.barh()
plt.title("Identification as a Midwesterner")
plt.xlabel("Number of respondents")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Bonus: Explore another feature
#

print("BONUS")

# Pick another column and explore its distribution.
# For example: `Gender`, `Age`, or one of the
# "Do you consider X state as part of the Midwest" columns.

# %%
# TODO: explore a column of your choice
# Explore the Gender column

print("Distribution of Gender:")
print(X["Gender"].value_counts(dropna=False))
print()

import matplotlib.pyplot as plt

X["Gender"].value_counts(dropna=False).plot.barh()
plt.title("Gender Distribution")
plt.xlabel("Number of respondents")
plt.tight_layout()
plt.show()
