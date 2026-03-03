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
# We use `cross_val_score` to get the score for every fold in cross-validation.

# %%
from sklearn.model_selection import cross_val_score

cv_lr = cross_val_score(model_lr, X, y, cv=5)
cv_rf = cross_val_score(model_rf, X, y, cv=5)
cv_gb = cross_val_score(model_gb, X, y, cv=5)

# %% [markdown]
# ## Question 6: Among the three models, which one has the best recall?
#
# The **classification report** shows precision, recall, and f1-score for each class.
#
# - **Precision**: among all predictions for a class, how many were correct?
# - **Recall**: among all real examples of a class, how many were found?
# - **F1-score**: a balance between precision and recall
#
# We will define the positive class as "North Central".

# %%
y_pred_lr = model_lr.predict(X_test)

# %%
from skore import EstimatorReport
report = EstimatorReport(estimator = model_lr,
                X_test = X_test,
                y_test = y_test)
report.help()

# %%
report.metrics.summarize(pos_label="North Central").frame()

# %% [markdown]
# Which model has the highest recall?
from sklearn.metrics import recall_score

print("===== QUESTION 6: RECALL =====")

for name, model in [
    ("Logistic Regression", model_lr),
    ("Random Forest", model_rf),
    ("Gradient Boosting", model_gb),
]:
    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred, pos_label="North Central")
    print(f"{name} recall:", recall)

from sklearn.metrics import confusion_matrix


# %% [markdown]
# ## Question 7: Which model has the best practical application?
#
# Let's say that it costs 10 to make a false positive error, while it costs 1 to make a false negative error. Correctly predicting a positive example gains 5, while correctly predicting a negative example gains 2.

# %%

# %% [markdown]
# Which model makes the most meaningful predictions in practice?
print("\n===== QUESTION 7: PRACTICAL SCORE =====")

def practical_score(model, X_test, y_test):
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred, labels=["other", "North Central"]
    ).ravel()

    score = (
        tp * 5 +      # gain true positive
        tn * 2 -      # gain true negative
        fp * 10 -     # cost false positive
        fn * 1        # cost false negative
    )
    return score

for name, model in [
    ("Logistic Regression", model_lr),
    ("Random Forest", model_rf),
    ("Gradient Boosting", model_gb),
]:
    print(f"{name} practical score:", practical_score(model, X_test, y_test))

# %% [markdown]
# ## Question 8: Which model generalizes the best?
#
# To understand generalization, we compare the **training score** (how well the model fits the data it was trained on) with the **test score** (how well it performs on unseen data).
#
# A big gap between the two means the model is **overfitting**.  
#
# We don't want to do this only once, but several times. Use cross-validation for that. You can either use cross-validation from scikit-learn, or the CrossValidationReport from skore.

# %%

# %% [markdown]
# Which model has the smallest gap between train and test accuracy?
# That model generalizes the best.
#
# Which model has the largest gap? That model is likely **overfitting**.
print("\n===== QUESTION 8: GENERALIZATION =====")

for name, model in [
    ("Logistic Regression", model_lr),
    ("Random Forest", model_rf),
    ("Gradient Boosting", model_gb),
]:
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    gap = train_score - test_score
    print(f"{name}:")
    print("  Train accuracy:", train_score)
    print("  Test accuracy:", test_score)
    print("  Gap:", gap)
    print()

# %%
# TODO: Based on the results above, which model would you choose
# for a real application? Write your answer as a comment below.

# My choice: Gradient Boosting
# Reason: Il a le meilleur recall (0.99), le meilleur score pratique (4629),
# et un écart relativement faible entre les performances d'entraînement et de test.
# Il offre donc le meilleur compromis entre performance et capacité de généralisation
# pour une application réelle.
