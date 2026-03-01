# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python (pixi)
#     language: python
#     name: cours_ia_cyber_laval_exploration
# ---

# %% [markdown]
# # Compare machine learning models
#
# In this notebook, we will compare 3 pre-trained models that predict the
# **Census Region** of a respondent based on their survey answers.
#
# The 3 models are:
# - **Logistic Regression**: a simple linear model
# - **Random Forest**: a model based on many decision trees
# - **Gradient Boosting**: a model that builds trees sequentially

# %% [markdown]
# ## Load the dataset

# %%
from skrub.datasets import fetch_midwest_survey

dataset = fetch_midwest_survey()
X = dataset.X
y = dataset.y

# %%
# To simplify evaluation, we will group categories in the target to deal with a binary classification problem instead of a multiclass one.
y = y.apply(lambda x: "North Central" if x in ["East North Central", "West North Central"] else "other")

# %%
sample_idx = X.sample(n=1000, random_state=1).index
X_train = X.loc[sample_idx].reset_index(drop=True)
y_train = y.loc[sample_idx].reset_index(drop=True)
X_test = X.drop(sample_idx).reset_index(drop=True)
y_test = y.drop(sample_idx).reset_index(drop=True)

# %% [markdown]
# ## Load the 3 models
#
# The models were saved as `.pkl` files. We use `joblib` to load them.

# %%
import joblib
from midwest_survey_models.transformers import NumericalStabilizer

model_lr = joblib.load("../model_logistic_regression.pkl")
model_rf = joblib.load("../model_random_forest.pkl")
model_gb = joblib.load("../model_gradient_boosting.pkl")

# %% [markdown]
# Let's inspect what each model looks like. They are **pipelines**: they
# first transform the data, then make predictions.

# %%
model_lr

# %%
model_rf

# %%
model_gb

# %% [markdown]
# ## Evaluate the models with cross-validation
#
# To fairly evaluate each model, we use **cross-validation**.
# This means we train and test the model on different parts of the data multiple times, so we can see how well it generalizes.
#
# We use `cross_val_predict` to get predictions for every example in the dataset (each time, the example was in the test set).

# %%
from sklearn.model_selection import cross_val_score

cv_lr = cross_val_score(model_lr, X, y, cv=5)
cv_rf = cross_val_score(model_rf, X, y, cv=5)
cv_gb = cross_val_score(model_gb, X, y, cv=5)

# %%
cv_rf

# %% [markdown]
# ## Question 6: Among the three models, which one has the best recall?
#
# The **classification report** shows precision, recall, and f1-score for each class.
#
# - **Precision**: among all predictions for a class, how many were correct?
# - **Recall**: among all real examples of a class, how many were found?
# - **F1-score**: a balance between precision and recall

# %%
from sklearn.metrics import classification_report

print("=== Logistic Regression ===")
print(classification_report(y, y_pred_lr))

# %%
print("=== Random Forest ===")
print(classification_report(y, y_pred_rf))

# %%
print("=== Gradient Boosting ===")
print(classification_report(y, y_pred_gb))

# %% [markdown]
# Look at the **weighted avg recall** row for each model.
# Which model has the highest recall?

# %% [markdown]
# ## Question 7: Which model has the best practical application?
#
# Let's visualize the confusion matrix for each model.
# This shows us which classes the model confuses with each other.

# %%
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for ax, y_pred, name in zip(
    axes,
    [y_pred_lr, y_pred_rf, y_pred_gb],
    ["Logistic Regression", "Random Forest", "Gradient Boosting"],
):
    ConfusionMatrixDisplay.from_predictions(y, y_pred, ax=ax, xticks_rotation=90)
    ax.set_title(name)

plt.tight_layout()

# %% [markdown]
# Look at the diagonal (correct predictions) and the off-diagonal
# (errors). Which model makes the most meaningful predictions in practice?

# %% [markdown]
# ## Question 8: Which model generalizes the best?
#
# To understand generalization, we compare the **training score** (how well the model fits the data it was trained on) with the **test score** (how well it performs on unseen data).
#
# A big gap between the two means the model is **overfitting**.

# %%
from sklearn.model_selection import cross_validate
import pandas as pd

models = {
    "Logistic Regression": model_lr,
    "Random Forest": model_rf,
    "Gradient Boosting": model_gb,
}

results = []
for name, model in models.items():
    cv_results = cross_validate(
        model, X, y, cv=5, scoring="accuracy",
        return_train_score=True
    )
    results.append({
        "Model": name,
        "Train accuracy (mean)": cv_results["train_score"].mean(),
        "Test accuracy (mean)": cv_results["test_score"].mean(),
        "Gap (train - test)": (
            cv_results["train_score"].mean()
            - cv_results["test_score"].mean()
        ),
    })

pd.DataFrame(results)

# %% [markdown]
# Which model has the smallest gap between train and test accuracy?
# That model generalizes the best.
#
# Which model has the largest gap? That model is likely **overfitting**.

# %%
# TODO: Based on the results above, which model would you choose
# for a real application? Write your answer as a comment below.

# My choice: ...
# Reason: ...

