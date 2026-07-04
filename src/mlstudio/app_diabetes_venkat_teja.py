"""app_diabetes_venkat_teja.py - Phase 5 custom project.

A custom project based on the example (app_case.py), applied to a new problem.

CHARACTERIZATION:
    - Dataset: CDC Diabetes Health Indicators (BRFSS 2015, balanced 50/50 sample).
    - Target: Diabetes_binary (0 = no diabetes, 1 = diabetes/prediabetes).
    - The target is a discrete category, so this is a CLASSIFICATION problem.
    - We picked a target to predict, so this is SUPERVISED learning.
    - The example project was supervised REGRESSION (continuous target: score);
      this custom project is supervised CLASSIFICATION (binary target).

Author: Venkat Teja Nallamothu
Date: 2026-07

Process:
    - Load a CSV dataset.
    - Train a supervised classification model (LogisticRegression).
    - Evaluate model performance (accuracy, confusion matrix).
    - Predict one new case.
    - Create useful charts and save them to artifacts/.

Data Source:
- data/raw/diabetes.csv
- Citation: Teboul, A. (2021). Diabetes Health Indicators Dataset.
  Derived from CDC BRFSS 2015. See data/raw/README.md for details.

Terminal command to run this file from the root project folder:

uv run python -m mlstudio.app_diabetes_venkat_teja
"""

# === Section 1a. DECLARE IMPORTS (BRING IN FREE CODE) ===

import logging
import pathlib
from typing import Final

from datafun_toolkit.logger import get_logger, log_header
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# === Section 1b. CONFIGURE LOGGER ONCE PER MODULE ===

LOG: logging.Logger = get_logger("ML", level="DEBUG")
log_header(LOG, "ML")

# === Section 1c. Global Constants and Configuration ===

DATASET_NAME: Final[str] = "diabetes"

# Folder for saved charts and other outputs.

ARTIFACTS_DIR: Final[pathlib.Path] = pathlib.Path("artifacts")

# STEP 1. Pick the target variable we want to predict.
# Diabetes_binary is 0.0 or 1.0 - a discrete category, NOT a continuous number.
# That makes this a supervised CLASSIFICATION problem.

TARGET_COL: Final[str] = "Diabetes_binary"

# STEP 2. Define the column names (features) that may help predict the target.
# A readable subset of the 21 available columns, chosen for interpretability:
# health indicators, lifestyle, and demographics.

FEATURE_COLS: Final[list[str]] = [
    "HighBP",
    "HighChol",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "GenHlth",
    "DiffWalk",
    "Age",
]

# STEP 3. Define the test size and random state for reproducibility.
# This dataset is large (roughly 70,000 rows), so a 20% test set
# still holds back about 14,000 records for evaluation.

TEST_SIZE: Final[float] = 0.20
RANDOM_STATE: Final[int] = 42

# === Section 1d. Pandas Configuration for Display ===

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 120)


# === Section 2. Load the Data ===


def load_data() -> pd.DataFrame:
    """Load the diabetes dataset from the data/raw folder."""
    LOG.info(f"Loading dataset: {DATASET_NAME}")

    df: pd.DataFrame = pd.read_csv(f"data/raw/{DATASET_NAME}.csv")

    LOG.info(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    LOG.debug(f"\n{df.head()}")

    return df


# === Section 3. Inspect Data Shape and Structure ===


def inspect_basic(df: pd.DataFrame) -> None:
    """Inspect basic dataset structure."""
    LOG.info("Column names")
    LOG.debug(f"{list(df.columns)}")

    LOG.info("DataFrame info")
    df.info()

    LOG.info(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    LOG.info("Target class balance (0 = no diabetes, 1 = diabetes/prediabetes)")
    LOG.info(f"\n{df[TARGET_COL].value_counts()}")


# === Section 4. Check Data Quality ===


def check_quality(df: pd.DataFrame) -> None:
    """Check missing values and duplicate rows."""
    LOG.info("Missing values by column")
    LOG.debug(f"\n{df.isna().sum()}")

    duplicate_count: int = df.duplicated().sum()
    LOG.info(f"Duplicate row count: {duplicate_count}")

    # Duplicates are kept: survey rows with identical answers are plausible
    # because every column is a rounded/binned value.


# === Section 5. Create a Clean View ===


def make_clean_view(df: pd.DataFrame) -> pd.DataFrame:
    """Create a cleaned view for modeling."""
    LOG.info("Creating clean modeling view")

    selected_cols: list[str] = FEATURE_COLS + [TARGET_COL]

    # Select only the columns we need.
    df_selected: pd.DataFrame = df[selected_cols]  # type: ignore[assignment]

    # Drop rows with any missing values.
    df_no_missing: pd.DataFrame = df_selected.dropna()

    # Assign a copy of the no-missing DataFrame to df_clean to avoid SettingWithCopyWarning.
    df_clean: pd.DataFrame = df_no_missing.copy()

    LOG.info(f"Clean view: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
    return df_clean


# === Section 6. Train Supervised Model ===


def train_model(df_clean: pd.DataFrame) -> LogisticRegression:
    """Train a supervised classification model."""
    LOG.info("Training LogisticRegression model")

    x = df_clean[FEATURE_COLS]
    y = df_clean[TARGET_COL]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # max_iter raised so the solver converges on unscaled features like BMI.
    model = LogisticRegression(max_iter=2000)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy: float = accuracy_score(y_test, y_pred)
    LOG.info(f"Accuracy on test set: {accuracy:.3f}")

    # The dataset is balanced 50/50, so always guessing one class
    # would only score about 0.50. Accuracy above that shows real signal.

    cm = confusion_matrix(y_test, y_pred)
    LOG.info(
        "Confusion matrix (rows = actual 0/1, columns = predicted 0/1):\n"
        f"{pd.DataFrame(cm, index=['actual 0', 'actual 1'], columns=['pred 0', 'pred 1'])}"
    )

    return model


# === Section 7. Predict One New Case ===


def predict_example(model: LogisticRegression) -> None:
    """Use the trained model to predict one new respondent."""
    LOG.info("Predicting one new case")

    # A hypothetical respondent: high blood pressure and cholesterol,
    # BMI 33, fair general health (4 of 5), age group 9 (60-64).
    new_case = pd.DataFrame(
        [
            {
                "HighBP": 1.0,
                "HighChol": 1.0,
                "BMI": 33.0,
                "Smoker": 0.0,
                "Stroke": 0.0,
                "HeartDiseaseorAttack": 0.0,
                "PhysActivity": 0.0,
                "GenHlth": 4.0,
                "DiffWalk": 0.0,
                "Age": 9.0,
            }
        ]
    )

    predicted_class: float = model.predict(new_case)[0]
    probability: float = model.predict_proba(new_case)[0][1]

    LOG.info(f"New case:\n{new_case}")
    LOG.info(f"Predicted class: {predicted_class:.0f} (1 = diabetes/prediabetes)")
    LOG.info(f"Predicted probability of diabetes: {probability:.2f}")


# === Section 8. Create Visualizations ===


def make_plots(df_clean: pd.DataFrame, model: LogisticRegression) -> None:
    """Create charts for the supervised classification case and save them."""
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    LOG.info("Creating chart: BMI by diabetes status")

    fig, ax = plt.subplots(figsize=(9, 5))

    box_plt: Axes = sns.boxplot(
        data=df_clean,
        x=TARGET_COL,
        y="BMI",
        ax=ax,
    )

    box_plt.set_title("BMI by Diabetes Status (0 = No, 1 = Diabetes/Prediabetes)")
    box_plt.set_xlabel("Diabetes Status")
    box_plt.set_ylabel("BMI")

    bmi_chart_path = ARTIFACTS_DIR / "diabetes_bmi_by_status_venkat_teja.png"
    fig.savefig(bmi_chart_path, dpi=100, bbox_inches="tight")
    LOG.info(f"Saved chart: {bmi_chart_path}")

    LOG.info("Creating chart: model coefficients")

    fig, ax = plt.subplots(figsize=(9, 5))

    coefficient_df = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "coefficient": model.coef_[0],
        }
    ).sort_values("coefficient", ascending=False)

    bar_plt: Axes = sns.barplot(
        data=coefficient_df,
        x="coefficient",
        y="feature",
        ax=ax,
    )

    bar_plt.set_title("Logistic Regression Coefficients (log-odds of diabetes)")
    bar_plt.set_xlabel("Coefficient")
    bar_plt.set_ylabel("Feature")

    coef_chart_path = ARTIFACTS_DIR / "diabetes_coefficients_venkat_teja.png"
    fig.savefig(coef_chart_path, dpi=100, bbox_inches="tight")
    LOG.info(f"Saved chart: {coef_chart_path}")


# === Section 9. Summary and Next Steps ===


def summarize(df: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    """Log a brief summary."""
    LOG.info("========================")
    LOG.info("SUMMARY")
    LOG.info("========================")
    LOG.info(f"Dataset: {DATASET_NAME}")
    LOG.info(f"Original rows: {df.shape[0]}")
    LOG.info(f"Clean rows: {df_clean.shape[0]}")
    LOG.info(f"Features: {FEATURE_COLS}")
    LOG.info(f"Target: {TARGET_COL}")
    LOG.info("Problem type: supervised classification (binary target)")


# === DEFINE THE MAIN FUNCTION THAT CALLS OTHER FUNCTIONS ===


def main() -> None:
    """Main function to run the supervised classification workflow."""
    log_header(LOG, "ML")

    LOG.info("========================")
    LOG.info("START main()")
    LOG.info("========================")

    LOG.info("Load dataset..............")
    df = load_data()

    LOG.info("Inspect dataset...........")
    inspect_basic(df)

    LOG.info("Check data quality........")
    check_quality(df)

    LOG.info("Create clean view.........")
    df_clean = make_clean_view(df)

    LOG.info("Train supervised model....")
    model = train_model(df_clean)

    LOG.info("Predict one case..........")
    predict_example(model)

    LOG.info("Create charts.............")
    make_plots(df_clean, model)

    LOG.info("Summarize workflow........")
    summarize(df, df_clean)

    LOG.info("Charts are saved in artifacts/ and also shown in windows.")
    LOG.info("----- CLOSE the chart windows to CONTINUE -----")

    plt.show()

    LOG.info("Workflow complete")
    LOG.info("========================")
    LOG.info("Executed successfully!")
    LOG.info("========================")


# === CONDITIONAL EXECUTION GUARD ===

if __name__ == "__main__":
    main()
