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


data["Control_Mean"] = data[
    control_columns
].mean(axis=1)


data["Treatment_Mean"] = data[
    treatment_columns
].mean(axis=1)


data["Fold_Change"] = (
    data["Treatment_Mean"]
    / data["Control_Mean"]
)


data["Fold_Change"] = data[
    "Fold_Change"
].replace(
    [np.inf, -np.inf],
    np.nan
)


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
# NCBI API
# =========================================================

NCBI_BASE_URL = (
    "https://eutils.ncbi.nlm.nih.gov/"
    "entrez/eutils/"
)


def search_ncbi_gene(gene_id):

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

            time.sleep(0.35)

        status_text.success(
            "NCBI annotation completed."
        )

        ncbi_results = pd.DataFrame(
            annotation_results
        )

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
# UNIPROT API
# =========================================================

UNIPROT_SEARCH_URL = (
    "https://rest.uniprot.org/uniprotkb/search"
)


def search_uniprot_gene(
    gene_id,
    organism_id="3702"
):

    """
    Search UniProtKB using the gene identifier
    and Arabidopsis thaliana taxonomy.

    Returns matching UniProt protein records.
    """

    query = (
        f'gene_exact:{gene_id} '
        f'AND organism_id:{organism_id}'
    )

    params = {
        "query": query,
        "format": "json",
        "size": 10,
        "fields": (
            "accession,"
            "id,"
            "gene_names,"
            "protein_name,"
            "organism_name,"
            "length,"
            "reviewed,"
            "cc_function"
        )
    }

    response = requests.get(
        UNIPROT_SEARCH_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def extract_protein_name(entry):

    try:

        protein_description = entry.get(
            "proteinDescription",
            {}
        )

        recommended = (
            protein_description
            .get(
                "recommendedName",
                {}
            )
            .get(
                "fullName",
                {}
            )
            .get(
                "value"
            )
        )

        if recommended:
            return recommended

        submitted = (
            protein_description
            .get(
                "submittedName",
                []
            )
        )

        if submitted:

            return (
                submitted[0]
                .get("fullName", {})
                .get("value", "Not available")
            )

    except Exception:
        pass

    return "Not available"


def extract_function(entry):

    try:

        comments = entry.get(
            "comments",
            []
        )

        for comment in comments:

            if comment.get(
                "commentType"
            ) == "FUNCTION":

                texts = comment.get(
                    "texts",
                    []
                )

                if texts:

                    return texts[0].get(
                        "value",
                        "Not available"
                    )

    except Exception:
        pass

    return "Not available"


def get_uniprot_annotation(gene_id):

    """
    Retrieve the best available UniProt result
    for a gene in Arabidopsis thaliana.
    """

    try:

        result = search_uniprot_gene(
            gene_id
        )

        entries = result.get(
            "results",
            []
        )

        if not entries:

            return {
                "UniProt_Accession": "Not found",
                "UniProt_ID": "Not found",
                "UniProt_Protein": "Not found",
                "UniProt_Organism": "Not found",
                "Protein_Length": "Not available",
                "UniProt_Reviewed": "Not available",
                "UniProt_Function": "Not available",
                "UniProt_Status": "Not found"
            }

        # -------------------------------------------------
        # Prefer reviewed / Swiss-Prot entries if present
        # -------------------------------------------------

        reviewed_entries = [
            entry
            for entry in entries
            if entry.get(
                "entryType"
            ) == "UniProtKB reviewed (Swiss-Prot)"
        ]

        if reviewed_entries:

            entry = reviewed_entries[0]

        else:

            entry = entries[0]

        # -------------------------------------------------
        # Extract accession
        # -------------------------------------------------

        accession = entry.get(
            "primaryAccession",
            "Not available"
        )

        # -------------------------------------------------
        # Entry ID
        # -------------------------------------------------

        entry_id = entry.get(
            "uniProtkbId",
            "Not available"
        )

        # -------------------------------------------------
        # Organism
        # -------------------------------------------------

        organism = (
            entry.get(
                "organism",
                {}
            )
            .get(
                "scientificName",
                "Not available"
            )
        )

        # -------------------------------------------------
        # Protein length
        # -------------------------------------------------

        protein_length = (
            entry.get(
                "sequence",
                {}
            )
            .get(
                "length",
                "Not available"
            )
        )

        # -------------------------------------------------
        # Review status
        # -------------------------------------------------

        entry_type = entry.get(
            "entryType",
            "Not available"
        )

        if (
            "reviewed"
            in entry_type.lower()
        ):

            reviewed_status = "Reviewed"

        else:

            reviewed_status = "Unreviewed"

        # -------------------------------------------------
        # Protein name
        # -------------------------------------------------

        protein_name = extract_protein_name(
            entry
        )

        # -------------------------------------------------
        # Function
        # -------------------------------------------------

        function = extract_function(
            entry
        )

        return {
            "UniProt_Accession": accession,
            "UniProt_ID": entry_id,
            "UniProt_Protein": protein_name,
            "UniProt_Organism": organism,
            "Protein_Length": protein_length,
            "UniProt_Reviewed": reviewed_status,
            "UniProt_Function": function,
            "UniProt_Status": "Found"
        }

    except Exception as error:

        return {
            "UniProt_Accession": "Error",
            "UniProt_ID": "Error",
            "UniProt_Protein": "Error",
            "UniProt_Organism": "Error",
            "Protein_Length": "Error",
            "UniProt_Reviewed": "Error",
            "UniProt_Function": str(error),
            "UniProt_Status": "Error"
        }


# =========================================================
# UNIPROT ANNOTATION
# =========================================================

st.header("9. UniProt Protein Annotation")


if significant_genes.empty:

    st.info(
        "UniProt annotation will appear when significant "
        "genes are available."
    )

else:

    st.write(
        "Searches UniProtKB for protein records corresponding "
        "to the significant Arabidopsis genes."
    )

    if st.button(
        "🧬 Annotate Significant Genes with UniProt"
    ):

        uniprot_results = []

        progress_bar = st.progress(0)

        status_text = st.empty()

        total_genes = len(
            significant_genes
        )

        for index, gene_id in enumerate(
            significant_genes["Gene_ID"]
        ):

            status_text.write(
                f"Searching UniProt for {gene_id}..."
            )

            annotation = get_uniprot_annotation(
                str(gene_id)
            )

            annotation["Gene_ID"] = gene_id

            uniprot_results.append(
                annotation
            )

            progress_bar.progress(
                (index + 1) / total_genes
            )

            time.sleep(0.2)

        status_text.success(
            "UniProt annotation completed."
        )

        uniprot_table = pd.DataFrame(
            uniprot_results
        )

        # -------------------------------------------------
        # Merge with differential-expression results
        # -------------------------------------------------

        uniprot_display = (
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
                uniprot_table,
                on="Gene_ID",
                how="left"
            )
        )

        st.subheader(
            "UniProt Annotation Results"
        )

        st.dataframe(
            uniprot_display,
            use_container_width=True
        )

        # Save for next stages
        st.session_state[
            "uniprot_annotation"
        ] = uniprot_display


# =========================================================
# INTERPRETATION
# =========================================================

st.header("10. Current Pipeline Status")


st.markdown(
    """
### Pipeline completed so far

**1. Expression data**

↓

**2. Differential expression**

- Control mean
- Treatment mean
- Fold change
- log₂ fold change
- p-value
- FDR-adjusted p-value

↓

**3. Significant gene identification**

- Upregulated
- Downregulated
- Not significant

↓

**4. NCBI annotation**

- NCBI Gene ID
- Gene name
- Description
- Organism

↓

**5. UniProt annotation**

- UniProt accession
- UniProt entry ID
- Protein name
- Organism
- Protein length
- Reviewed/unreviewed status
- Function when available

### Next stage

The next module will use the protein information to retrieve:

**InterPro domains and protein families**

After that, we will connect the annotated genes/proteins to:

**KEGG pathways**
"""
)


# =========================================================
# DOWNLOAD DIFFERENTIAL EXPRESSION RESULTS
# =========================================================

st.header("11. Download Results")


csv_data = results.to_csv(
    index=False
)


st.download_button(
    label="⬇️ Download Differential Expression Results",
    data=csv_data,
    file_name="differential_expression_results.csv",
    mime="text/csv"
)
