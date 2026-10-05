import streamlit as st
import pandas as pd
import numpy as np
import requests
import time

from scipy.stats import ttest_ind


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Transcriptomics KEGG Pipeline",
    page_icon="🧬",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🧬 Transcriptomics & KEGG Pathway Analyzer")

st.write(
    "A pipeline for differential expression analysis, "
    "gene annotation, and KEGG pathway mapping."
)


# =========================================================
# LOAD EXPRESSION DATA
# =========================================================

st.header("1. Expression Data")

try:
    data = pd.read_csv("dummy_data.csv")

except FileNotFoundError:
    st.error(
        "dummy_data.csv was not found. "
        "Please make sure the file is present in the repository."
    )
    st.stop()


st.dataframe(
    data,
    use_container_width=True
)


# =========================================================
# DEFINE EXPERIMENTAL GROUPS
# =========================================================

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


# =========================================================
# CHECK REQUIRED COLUMNS
# =========================================================

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


# =========================================================
# EXPRESSION STATISTICS
# =========================================================

st.header("2. Expression Statistics")


# Control mean
data["Control_Mean"] = data[
    control_columns
].mean(axis=1)


# Treatment mean
data["Treatment_Mean"] = data[
    treatment_columns
].mean(axis=1)


# Fold change
data["Fold_Change"] = (
    data["Treatment_Mean"]
    / data["Control_Mean"]
)


# Prevent infinity
data["Fold_Change"] = data[
    "Fold_Change"
].replace(
    [np.inf, -np.inf],
    np.nan
)


# Log2 fold change
data["Log2_Fold_Change"] = np.log2(
    data["Fold_Change"]
)


# =========================================================
# DIFFERENTIAL EXPRESSION
# =========================================================

st.header("3. Differential Expression Analysis")

st.write(
    "For this prototype, Welch's two-sample t-test "
    "is used to compare control and treatment replicates."
)


p_values = []


for _, row in data.iterrows():

    control_values = row[
        control_columns
    ].astype(float).values

    treatment_values = row[
        treatment_columns
    ].astype(float).values

    test_result = ttest_ind(
        treatment_values,
        control_values,
        equal_var=False
    )

    p_values.append(
        test_result.pvalue
    )


data["P_Value"] = p_values


# =========================================================
# BENJAMINI-HOCHBERG FDR
# =========================================================

def benjamini_hochberg(p_values):

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    n = len(p_values)

    order = np.argsort(
        p_values
    )

    sorted_p_values = p_values[
        order
    ]

    adjusted = np.empty(n)

    previous = 1.0

    for i in range(
        n - 1,
        -1,
        -1
    ):

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


data["Adjusted_P_Value"] = (
    benjamini_hochberg(
        data["P_Value"]
    )
)


# =========================================================
# SIGNIFICANCE THRESHOLDS
# =========================================================

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


# =========================================================
# CLASSIFY GENES
# =========================================================

def classify_gene(row):

    log2fc = row[
        "Log2_Fold_Change"
    ]

    adjusted_p = row[
        "Adjusted_P_Value"
    ]

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


# =========================================================
# DIFFERENTIAL EXPRESSION RESULTS
# =========================================================

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


results = data[
    results_columns
].copy()


numeric_columns = [
    "Control_Mean",
    "Treatment_Mean",
    "Fold_Change",
    "Log2_Fold_Change",
    "P_Value",
    "Adjusted_P_Value"
]


results[numeric_columns] = (
    results[numeric_columns].round(4)
)


st.dataframe(
    results,
    use_container_width=True
)


# =========================================================
# SUMMARY
# =========================================================

st.header("6. Differential Expression Summary")


upregulated = (
    data["Regulation"]
    == "Upregulated"
).sum()


downregulated = (
    data["Regulation"]
    == "Downregulated"
).sum()


not_significant = (
    data["Regulation"]
    == "Not significant"
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


# =========================================================
# SIGNIFICANT GENES
# =========================================================

st.header("7. Significant Genes")


significant_genes = data[
    data["Regulation"]
    != "Not significant"
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

    significant_display = (
        significant_genes[
            significant_columns
        ].copy()
    )

    significant_display[
        [
            "Log2_Fold_Change",
            "P_Value",
            "Adjusted_P_Value"
        ]
    ] = (
        significant_display[
            [
                "Log2_Fold_Change",
                "P_Value",
                "Adjusted_P_Value"
            ]
        ].round(4)
    )

    st.dataframe(
        significant_display,
        use_container_width=True
    )


# =========================================================
# NCBI ANNOTATION FUNCTIONS
# =========================================================

NCBI_BASE_URL = (
    "https://eutils.ncbi.nlm.nih.gov/"
    "entrez/eutils/"
)


def search_ncbi_gene(gene_id):

    """
    Search NCBI Gene using the supplied gene identifier.
    """

    params = {
        "db": "gene",
        "term": f'"{gene_id}"[All Fields]',
        "retmode": "json",
        "retmax": 10
    }

    response = requests.get(
        NCBI_BASE_URL + "esearch.fcgi",
        params=params,
        timeout=20
    )

    response.raise_for_status()

    result = response.json()

    ids = result.get(
        "esearchresult",
        {}
    ).get(
        "idlist",
        []
    )

    return ids


def get_ncbi_gene_summary(gene_uid):

    """
    Retrieve an NCBI Gene summary.
    """

    params = {
        "db": "gene",
        "id": gene_uid,
        "retmode": "json"
    }

    response = requests.get(
        NCBI_BASE_URL + "esummary.fcgi",
        params=params,
        timeout=20
    )

    response.raise_for_status()

    result = response.json()

    document = result.get(
        "result",
        {}
    ).get(
        str(gene_uid),
        {}
    )

    return document


def annotate_gene_ncbi(gene_id):

    """
    Search and annotate one gene using NCBI.
    """

    try:

        ids = search_ncbi_gene(
            gene_id
        )

        if not ids:

            return {
                "NCBI_Gene_ID": "Not found",
                "Gene_Name": "Not found",
                "Gene_Description": "Not found",
                "Organism": "Not found",
                "Chromosome": "Not available",
                "NCBI_Status": "Not found"
            }

        # Use the first NCBI result for this prototype.
        ncbi_uid = ids[0]

        summary = get_ncbi_gene_summary(
            ncbi_uid
        )

        return {
            "NCBI_Gene_ID": summary.get(
                "uid",
                ncbi_uid
            ),

            "Gene_Name": summary.get(
                "name",
                "Not available"
            ),

            "Gene_Description": summary.get(
                "description",
                "Not available"
            ),

            "Organism": summary.get(
                "organism",
                {}).get(
                    "scientificname",
                    "Not available"
                ),

            "Chromosome": summary.get(
                "chromosome",
                "Not available"
            ),

            "NCBI_Status": summary.get(
                "status",
                "Not available"
            )
        }

    except Exception as error:

        return {
            "NCBI_Gene_ID": "Error",
            "Gene_Name": "Error",
            "Gene_Description": str(error),
            "Organism": "Error",
            "Chromosome": "Error",
            "NCBI_Status": "Error"
        }


# =========================================================
# NCBI ANNOTATION
# =========================================================

st.header("8. NCBI Gene Annotation")


if significant_genes.empty:

    st.info(
        "NCBI annotation will appear when significant genes "
        "are available."
    )

else:

    st.write(
        "The significant genes are now sent to NCBI Gene "
        "for annotation."
    )

    if st.button(
        "🔎 Annotate Significant Genes with NCBI"
    ):

        annotation_results = []

        progress_bar = st.progress(0)

        status_text = st.empty()

        total_genes = len(
            significant_genes
        )

        for index, gene_id in enumerate(
            significant_genes["Gene_ID"]
        ):

            status_text.write(
                f"Annotating {gene_id}..."
            )

            annotation = annotate_gene_ncbi(
                str(gene_id)
            )

            annotation["Gene_ID"] = gene_id

            annotation_results.append(
                annotation
            )

            progress_bar.progress(
                (index + 1) / total_genes
            )

            # Small delay between requests
            time.sleep(0.35)

        status_text.success(
            "NCBI annotation completed."
        )

        ncbi_results = pd.DataFrame(
            annotation_results
        )

        # Merge annotation with DE results
        annotation_table = (
            significant_genes[
                [
                    "Gene_ID",
                    "Log2_Fold_Change",
                    "P_Value",
                    "Adjusted_P_Value",
                    "Regulation"
                ]
            ]
            .merge(
                ncbi_results,
                on="Gene_ID",
                how="left"
            )
        )

        st.subheader(
            "NCBI Annotation Results"
        )

        st.dataframe(
            annotation_table,
            use_container_width=True
        )

        st.session_state[
            "ncbi_annotation"
        ] = annotation_table


# =========================================================
# INTERPRETATION
# =========================================================

st.header("9. Interpretation")


st.markdown(
    """
### Current pipeline stage

Your transcriptomics workflow now contains:

**Expression data**

↓

**Differential expression**

↓

**Significant genes**

↓

**NCBI Gene annotation**

The NCBI annotation provides biological information
associated with the significant genes, such as:

- NCBI Gene ID
- Gene name
- Gene description
- Organism
- Chromosome
- NCBI record status

The next stages will add **UniProt protein annotation**,
then **InterPro protein-domain information**, and finally
**KEGG pathway mapping**.
"""
)


# =========================================================
# DOWNLOAD
# =========================================================

st.header("10. Download Results")


csv_data = results.to_csv(
    index=False
)


st.download_button(
    label="⬇️ Download Differential Expression Results",
    data=csv_data,
    file_name="differential_expression_results.csv",
    mime="text/csv"
)
