import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Transcriptomics KEGG Pipeline",
    page_icon="🧬",
    layout="wide"
)


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title("🧬 Transcriptomics & KEGG Pathway Analyzer")

st.write(
    "A pipeline for differential expression analysis, "
    "gene annotation, and KEGG pathway mapping."
)


# ---------------------------------------------------------
# Load expression data
# ---------------------------------------------------------

st.header("1. Expression Data")

try:
    data = pd.read_csv("dummy_data.csv")
except FileNotFoundError:
    st.error(
        "dummy_data.csv was not found. "
        "Please make sure the file is present in the repository."
    )
    st.stop()


st.write("Input expression matrix:")

st.dataframe(
    data,
    use_container_width=True
)


# ---------------------------------------------------------
# Define experimental groups
# ---------------------------------------------------------

control_columns = [
    "Control_1",
    "Control_2",
    "Control_3"
]

treatment_columns = [
    "Treatment_1",
    "Treatment_2",
    "Treatment_3"
]


# ---------------------------------------------------------
# Check required columns
# ---------------------------------------------------------

required_columns = (
    ["Gene_ID"]
    + control_columns
    + treatment_columns
)

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    st.error(
        "The following required columns are missing: "
        + ", ".join(missing_columns)
    )
    st.stop()


# ---------------------------------------------------------
# Calculate expression statistics
# ---------------------------------------------------------

st.header("2. Expression Statistics")


# Control mean
data["Control_Mean"] = data[control_columns].mean(axis=1)


# Treatment mean
data["Treatment_Mean"] = data[treatment_columns].mean(axis=1)


# Fold change
data["Fold_Change"] = (
    data["Treatment_Mean"]
    / data["Control_Mean"]
)


# Avoid invalid values
data["Fold_Change"] = data["Fold_Change"].replace(
    [np.inf, -np.inf],
    np.nan
)


# Log2 fold change
data["Log2_Fold_Change"] = np.log2(
    data["Fold_Change"]
)


# ---------------------------------------------------------
# Calculate p-values
# ---------------------------------------------------------

st.header("3. Differential Expression Analysis")

st.write(
    "For this prototype, a Welch's two-sample t-test is used "
    "to compare the control and treatment replicates."
)


p_values = []


for _, row in data.iterrows():

    control_values = row[control_columns].astype(float).values

    treatment_values = row[treatment_columns].astype(float).values

    test_result = ttest_ind(
        treatment_values,
        control_values,
        equal_var=False
    )

    p_values.append(test_result.pvalue)


data["P_Value"] = p_values


# ---------------------------------------------------------
# Benjamini-Hochberg FDR correction
# ---------------------------------------------------------

def benjamini_hochberg(p_values):
    """
    Calculate Benjamini-Hochberg adjusted p-values.
    """

    p_values = np.asarray(p_values, dtype=float)

    n = len(p_values)

    order = np.argsort(p_values)

    sorted_p_values = p_values[order]

    adjusted = np.empty(n)

    previous = 1.0

    for i in range(n - 1, -1, -1):

        rank = i + 1

        adjusted_value = (
            sorted_p_values[i]
            * n
            / rank
        )

        adjusted_value = min(
            adjusted_value,
            previous
        )

        adjusted[i] = adjusted_value

        previous = adjusted_value

    result = np.empty(n)

    result[order] = adjusted

    return result


data["Adjusted_P_Value"] = benjamini_hochberg(
    data["P_Value"]
)


# ---------------------------------------------------------
# Define significance thresholds
# ---------------------------------------------------------

st.header("4. Significance Classification")

col1, col2 = st.columns(2)


with col1:

    log2fc_threshold = st.number_input(
        "Absolute log₂ Fold Change threshold",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )


with col2:

    adjusted_p_threshold = st.number_input(
        "Adjusted p-value threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.01
    )


# ---------------------------------------------------------
# Classify genes
# ---------------------------------------------------------

def classify_gene(row):

    log2fc = row["Log2_Fold_Change"]

    adjusted_p = row["Adjusted_P_Value"]

    if (
        adjusted_p < adjusted_p_threshold
        and log2fc >= log2fc_threshold
    ):
        return "Upregulated"

    elif (
        adjusted_p < adjusted_p_threshold
        and log2fc <= -log2fc_threshold
    ):
        return "Downregulated"

    else:
        return "Not significant"


data["Regulation"] = data.apply(
    classify_gene,
    axis=1
)


# ---------------------------------------------------------
# Display differential expression results
# ---------------------------------------------------------

st.header("5. Differential Expression Results")


results_columns = [
    "Gene_ID",
    "Control_Mean",
    "Treatment_Mean",
    "Fold_Change",
    "Log2_Fold_Change",
    "P_Value",
    "Adjusted_P_Value",
    "Regulation"
]


results = data[results_columns].copy()


# Round numerical columns
numeric_columns = [
    "Control_Mean",
    "Treatment_Mean",
    "Fold_Change",
    "Log2_Fold_Change",
    "P_Value",
    "Adjusted_P_Value"
]

results[numeric_columns] = results[numeric_columns].round(4)


st.dataframe(
    results,
    use_container_width=True
)


# ---------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------

st.header("6. Differential Expression Summary")


upregulated = (
    data["Regulation"] == "Upregulated"
).sum()


downregulated = (
    data["Regulation"] == "Downregulated"
).sum()


not_significant = (
    data["Regulation"] == "Not significant"
).sum()


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Upregulated genes",
        upregulated
    )


with col2:

    st.metric(
        "Downregulated genes",
        downregulated
    )


with col3:

    st.metric(
        "Not significant",
        not_significant
    )


# ---------------------------------------------------------
# Significant genes
# ---------------------------------------------------------

st.header("7. Significant Genes")


significant_genes = data[
    data["Regulation"] != "Not significant"
].copy()


if significant_genes.empty:

    st.info(
        "No genes meet the current significance thresholds."
    )

else:

    significant_columns = [
        "Gene_ID",
        "Log2_Fold_Change",
        "P_Value",
        "Adjusted_P_Value",
        "Regulation"
    ]

    significant_display = significant_genes[
        significant_columns
    ].copy()

    significant_display[
        [
            "Log2_Fold_Change",
            "P_Value",
            "Adjusted_P_Value"
        ]
    ] = significant_display[
        [
            "Log2_Fold_Change",
            "P_Value",
            "Adjusted_P_Value"
        ]
    ].round(4)

    st.dataframe(
        significant_display,
        use_container_width=True
    )


# ---------------------------------------------------------
# Explanation
# ---------------------------------------------------------

st.header("8. Interpretation")

st.markdown(
    """
### What the analysis means

**Log₂ Fold Change**

- Positive value → higher expression in treatment
- Negative value → lower expression in treatment
- Larger absolute value → larger expression difference

**P-value**

The p-value measures evidence against the null hypothesis
that the two groups have the same mean expression.

**Adjusted P-value**

Multiple genes are tested simultaneously, so raw p-values
are adjusted using the **Benjamini-Hochberg FDR method**.

**Regulation**

A gene is classified as:

- **Upregulated** → positive log₂FC above the threshold and significant adjusted p-value
- **Downregulated** → negative log₂FC below the threshold and significant adjusted p-value
- **Not significant** → does not satisfy both criteria
"""
)


# ---------------------------------------------------------
# Download results
# ---------------------------------------------------------

st.header("9. Download Results")


csv_data = results.to_csv(index=False)


st.download_button(
    label="⬇️ Download Differential Expression Results",
    data=csv_data,
    file_name="differential_expression_results.csv",
    mime="text/csv"
)
