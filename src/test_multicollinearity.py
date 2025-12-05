#!/usr/bin/env python
# coding: utf-8

# # Checking Multicollinearity Among Predictors
#
# This script measures what is the extent of multicollinearity among the predictor variables we used in our analysis. Find [here](https://online.stat.psu.edu/stat462/node/180/) more information about multicollinarity and its effect on regression analyses.

# ## Imports and Loading

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

tsv_path = "./survey_clean_var.tsv"
df_clean = pd.read_csv(tsv_path, sep="\t", encoding="utf-8")
print(df_clean.columns)

predictor_variables = [
    "EducationGroup",
    "GeographyGroup",
    "GenderGroup",
    "IncomeGroup",
    "AgeGroup",
    "LT_exp",
    "lt_lit",
]
predictors_df = df_clean[predictor_variables]
print(predictors_df.dtypes)
print(predictors_df.shape)

# drop rows with missing values
print("Shape before dropping NA:", predictors_df.shape)
predictors_df = predictors_df.dropna()
print("Shape after dropping NA:", predictors_df.shape)

# remap ordinal variables to integer encoding
predictors_df["IncomeGroup"] = predictors_df["IncomeGroup"].map(
    {"Lower": 1, "Mid": 2, "Higher": 3}
)
predictors_df["AgeGroup"] = predictors_df["AgeGroup"].map(
    {"18-34": 1, "35-54": 2, "55-64": 3, "65+": 4}
)

# one-hot encoding of categorical variables
predictors_df = pd.get_dummies(
    predictors_df,
    columns=["EducationGroup", "GeographyGroup", "GenderGroup"],
    drop_first=True,
)

# Print dtypes of the final columns
print(predictors_df.dtypes)


def calculate_gvif(df):
    """
    Calculates the Generalized Variance Inflation Factor (GVIF) for a DataFrame.

    The function handles numerical, categorical (binary/multiclass), and ordinal variables.
    Categorical variables are expected to be one-hot encoded, with original
    variable names joined by an underscore (e.g., 'color_blue', 'color_red').
    Ordinal variables should be integer-encoded.

    Args:
        df (pd.DataFrame): The input DataFrame with predictor variables.

    Returns:
        pd.DataFrame: A DataFrame with GVIF and degrees of freedom for each variable.
    """
    gvif_data = {}

    # Identify unique variables from one-hot encoded columns
    # e.g., 'color_blue' and 'color_red' belong to 'color'
    def get_base_variable(col, df_cols):
        if "_" in col:
            base_name = col.split("_")[0]
            related_cols = [c for c in df_cols if c.startswith(base_name + "_")]
            if len(related_cols) > 1:
                return base_name
        return col

    variables = sorted(list(set(get_base_variable(c, df.columns) for c in df.columns)))

    print("Variables considered for GVIF calculation:", variables)

    for var in variables:
        # Get all columns related to the current variable (for one-hot encoding)
        is_categorical = False
        target_cols = [c for c in df.columns if c.startswith(var + "_")]
        if not target_cols:
            target_cols = [var]
        else:
            is_categorical = True

        # print("Target cols:", target_cols)
        # print("Is categorical:", is_categorical)

        y = df[target_cols]
        X = df.drop(columns=target_cols)
        X = sm.add_constant(X, has_constant="add")  # Add intercept

        # Determine the type of the target variable
        dtype = df[target_cols[0]].dtype
        print("Dtype of current variable:", dtype)

        try:
            # Ensure design matrix is purely numeric (statsmodels breaks with mixed dtypes -> object ndarray)
            X_numeric = X.copy()
            for c in X_numeric.columns:
                if X_numeric[c].dtype == bool:
                    X_numeric[c] = X_numeric[c].astype(int)
            # Cast everything to float for homogeneity
            X_numeric = X_numeric.astype(float)

            # Use a 1D series when single-column target (many statsmodels classes expect 1D endog)
            # y_series = y.iloc[:, 0] if y.shape[1] == 1 else y
            y_series = y

            # Numerical target (single column, continuous)
            if (
                np.issubdtype(dtype, np.number)
                and not is_categorical
                and df[target_cols[0]].nunique() > 2
            ):
                model = sm.OLS(y_series.astype(float), X_numeric)
                rsquared = model.fit().rsquared
                df_deg = 1  # Degrees of freedom
            # Ordinal target (assumed if integer and not one-hot)
            elif np.issubdtype(dtype, np.integer) and not is_categorical:
                model = OrderedModel(y_series.astype(int), X_numeric, distr="logit")
                rsquared = model.fit(method="bfgs", disp=False).prsquared
                df_deg = 1
            # Categorical target (one-hot encoded or binary dummy)
            else:
                if y.shape[1] == 1:  # single binary / dummy column
                    endog = y_series.astype(int)
                    model = sm.Logit(endog, X_numeric)
                    rsquared = model.fit(disp=False).prsquared
                    df_deg = 1
                else:
                    # Multiclass one-hot (k columns) -> treat as multinomial
                    # Convert to a single categorical series (argmax of dummies)
                    endog = y.values.argmax(axis=1)
                    model = sm.MNLogit(endog, X_numeric)
                    rsquared = model.fit(disp=False).prsquared
                    df_deg = y.shape[1] - 1  # k-1

            gvif = 1 / (1 - rsquared)
            # GVIF^(1/(2*df)) is a scaled version for better comparison
            gvif_scaled = gvif ** (1 / (2 * df_deg))

            gvif_data[var] = {
                "GVIF": gvif,
                "Df": df_deg,
                "GVIF^(1/(2*Df))": gvif_scaled,
            }
        except Exception as e:
            print(f"Could not calculate GVIF for {var}: {e}")

    return pd.DataFrame.from_dict(gvif_data, orient="index")


# --- Example Usage ---

# # 1. Create a sample DataFrame
# data = {
#     'age': [25, 30, 35, 40, 45, 50, 55, 60],
#     'income': [50000, 60000, 75000, 90000, 110000, 130000, 150000, 180000],
#     'experience': [2, 5, 8, 12, 15, 18, 22, 25], # Highly correlated with age
#     'education': [2, 3, 3, 4, 4, 5, 5, 5], # Ordinal: 1=HS, 2=Bach, 3=Mast, 4=PhD
#     'city': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B'] # Categorical
# }
# df_sample = pd.DataFrame(data)

# # 2. One-hot encode the categorical variable
# df_processed = pd.get_dummies(df_sample, columns=['city'], drop_first=True)

# # 3. Calculate and print GVIF
# gvif_results = calculate_gvif(df_processed)
# print(gvif_results)

gvif_results = calculate_gvif(predictors_df)

print("\n")
print("#" * 20)
print("GVIF Results:\n")
print(gvif_results)
print("#" * 20)

print("\nGVIF Results in LaTeX format:\n")
latex_table = gvif_results.to_latex(index=True, float_format="%.3f")
print(latex_table)
