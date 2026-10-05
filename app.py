import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from scipy.stats import ttest_ind


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Transcriptomics KEGG Pipeline",
    page_icon="🧬",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🧬 Transcriptomics & KEGG Pathway Analyzer")

st.write(
    "A pipeline for differential expression analysis, "
    "gene annotation, and KEGG pathway mapping."
)


# ============================================================
# LOAD DATA
# ============================================================

st.header("1. Transcriptomics Data")

try:
    data = pd.read_csv("dummy_data.csv")

    st.success("Transcriptomics data loaded successfully.")

    st.dataframe(
        data,
        use_container_width=True
    )

except Exception as e:

    st.error(
        f"Unable to load dummy_data.csv: {e}"
    )

    st.stop()


# ============================================================
# DEFINE EXPERIMENTAL GROUPS
# ============================================================

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


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

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


# ============================================================
# DIFFERENTIAL EXPRESSION
# ============================================================

st.header("2. Differential Expression Analysis")


# ------------------------------------------------------------
# Calculate group means
# ------------------------------------------------------------

data["Control_Mean"] = data[
    control_columns
].mean(axis=1)

data["Treatment_Mean"] = data[
    treatment_columns
].mean(axis=1)


# ------------------------------------------------------------
# Fold change
# ------------------------------------------------------------

# Avoid division by zero
data["Fold_Change"] = (
    data["Treatment_Mean"]
    / data["Control_Mean"].replace(0, np.nan)
)


# ------------------------------------------------------------
# Log2 fold change
# ------------------------------------------------------------

data["Log2_Fold_Change"] = np.log2(
    data["Fold_Change"]
)


# ============================================================
# WELCH'S T-TEST
# ============================================================

p_values = []

for _, row in data.iterrows():

    control_values = row[
        control_columns
    ].astype(float).values

    treatment_values = row[
        treatment_columns
    ].astype(float).values

    try:

        statistic, p_value = ttest_ind(
            treatment_values,
            control_values,
            equal_var=False,
            nan_policy="omit"
        )

    except Exception:

        p_value = np.nan

    p_values.append(p_value)


data["P_Value"] = p_values


# ============================================================
# BENJAMINI-HOCHBERG FDR
# ============================================================

def benjamini_hochberg(p_values):

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    adjusted = np.full(
        len(p_values),
        np.nan
    )

    valid = ~np.isnan(p_values)

    if valid.sum() == 0:
        return adjusted

    valid_p = p_values[valid]

    order = np.argsort(valid_p)

    ranked = valid_p[order]

    n = len(ranked)

    adjusted_ranked = (
        ranked
        * n
        / np.arange(1, n + 1)
    )

    adjusted_ranked = np.minimum.accumulate(
        adjusted_ranked[::-1]
    )[::-1]

    adjusted_ranked = np.minimum(
        adjusted_ranked,
        1.0
    )

    result = np.empty(n)

    result[order] = adjusted_ranked

    adjusted[valid] = result

    return adjusted


data["Adjusted_P_Value"] = (
    benjamini_hochberg(
        data["P_Value"].values
    )
)


# ============================================================
# SIGNIFICANCE THRESHOLDS
# ============================================================

st.subheader("Significance Thresholds")

col1, col2 = st.columns(2)

with col1:

    log2fc_threshold = st.number_input(
        "Absolute log2 Fold Change threshold",
        min_value=0.0,
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


# ============================================================
# CLASSIFY GENES
# ============================================================

def classify_gene(row):

    log2fc = row["Log2_Fold_Change"]
    adjusted_p = row["Adjusted_P_Value"]

    if pd.isna(log2fc) or pd.isna(adjusted_p):

        return "Not significant"

    if (
        log2fc >= log2fc_threshold
        and adjusted_p <= adjusted_p_threshold
    ):

        return "Upregulated"

    elif (
        log2fc <= -log2fc_threshold
        and adjusted_p <= adjusted_p_threshold
    ):

        return "Downregulated"

    else:

        return "Not significant"


data["Regulation"] = data.apply(
    classify_gene,
    axis=1
)


# ============================================================
# DISPLAY DIFFERENTIAL EXPRESSION RESULTS
# ============================================================

st.subheader(
    "Differential Expression Results"
)

display_columns = [
    "Gene_ID",
    "Control_Mean",
    "Treatment_Mean",
    "Fold_Change",
    "Log2_Fold_Change",
    "P_Value",
    "Adjusted_P_Value",
    "Regulation"
]

st.dataframe(
    data[display_columns],
    use_container_width=True
)


# ============================================================
# SUMMARY
# ============================================================

st.header("3. Differential Expression Summary")

upregulated_count = (
    data["Regulation"] == "Upregulated"
).sum()

downregulated_count = (
    data["Regulation"] == "Downregulated"
).sum()

not_significant_count = (
    data["Regulation"] == "Not significant"
).sum()


col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Upregulated",
        int(upregulated_count)
    )

with col2:

    st.metric(
        "Downregulated",
        int(downregulated_count)
    )

with col3:

    st.metric(
        "Not Significant",
        int(not_significant_count)
    )


# ============================================================
# SIGNIFICANT GENES
# ============================================================

st.header("4. Significant Genes")

significant_data = data[
    data["Regulation"].isin(
        [
            "Upregulated",
            "Downregulated"
        ]
    )
].copy()


if significant_data.empty:

    st.warning(
        "No significant genes found using the current thresholds."
    )

else:

    st.dataframe(
        significant_data[
            [
                "Gene_ID",
                "Log2_Fold_Change",
                "P_Value",
                "Adjusted_P_Value",
                "Regulation"
            ]
        ],
        use_container_width=True
    )


# ============================================================
# DOWNLOAD DIFFERENTIAL EXPRESSION RESULTS
# ============================================================

csv_data = data.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="⬇️ Download Differential Expression Results",
    data=csv_data,
    file_name="differential_expression_results.csv",
    mime="text/csv"
)


# ============================================================
# NCBI ANNOTATION
# ============================================================

st.header("6. NCBI Gene Annotation")


NCBI_BASE_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
)


def search_ncbi_gene(gene_id):

    search_url = (
        NCBI_BASE_URL
        + "esearch.fcgi"
    )

    params = {
        "db": "gene",
        "term": (
            f'"{gene_id}"[All Fields] '
            f'AND "Arabidopsis thaliana"[Organism]'
        ),
        "retmode": "json"
    }

    response = requests.get(
        search_url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data_json = response.json()

    id_list = (
        data_json
        .get("esearchresult", {})
        .get("idlist", [])
    )

    if not id_list:

        return None

    return id_list[0]


def get_ncbi_gene_summary(gene_id):

    ncbi_id = search_ncbi_gene(
        gene_id
    )

    if ncbi_id is None:

        return {
            "NCBI_Gene_ID": "Not found",
            "Gene_Name": "Not found",
            "Gene_Description": "Not found",
            "Organism": "Not found",
            "Chromosome": "Not found",
            "NCBI_Status": "No match"
        }

    summary_url = (
        NCBI_BASE_URL
        + "esummary.fcgi"
    )

    params = {
        "db": "gene",
        "id": ncbi_id,
        "retmode": "json"
    }

    response = requests.get(
        summary_url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    summary_data = response.json()

    result = summary_data.get(
        "result",
        {}
    )

    record = result.get(
        str(ncbi_id),
        {}
    )

    return {
        "NCBI_Gene_ID": ncbi_id,

        "Gene_Name": record.get(
            "name",
            "Not available"
        ),

        "Gene_Description": record.get(
            "description",
            "Not available"
        ),

        "Organism": record.get(
            "organism",
            {}).get(
                "scientificname",
                "Not available"
            ),

        "Chromosome": record.get(
            "chromosome",
            "Not available"
        ),

        "NCBI_Status": "Annotated"
    }


# ============================================================
# NCBI ANNOTATION BUTTON
# ============================================================

if not significant_data.empty:

    if st.button(
        "🔎 Annotate Significant Genes with NCBI"
    ):

        ncbi_results = []

        progress_bar = st.progress(0)

        total = len(
            significant_data
        )

        for index, (_, row) in enumerate(
            significant_data.iterrows()
        ):

            gene_id = row[
                "Gene_ID"
            ]

            try:

                annotation = (
                    get_ncbi_gene_summary(
                        gene_id
                    )
                )

            except Exception as e:

                annotation = {
                    "NCBI_Gene_ID": "API error",
                    "Gene_Name": "API error",
                    "Gene_Description": str(e),
                    "Organism": "API error",
                    "Chromosome": "API error",
                    "NCBI_Status": "Error"
                }

            annotation[
                "Gene_ID"
            ] = gene_id

            ncbi_results.append(
                annotation
            )

            progress_bar.progress(
                (index + 1) / total
            )

            time.sleep(0.2)

        ncbi_df = pd.DataFrame(
            ncbi_results
        )

        st.session_state[
            "ncbi_annotation"
        ] = ncbi_df

        st.success(
            "NCBI annotation completed."
        )


# ============================================================
# DISPLAY NCBI RESULTS
# ============================================================

if "ncbi_annotation" in st.session_state:

    st.subheader(
        "NCBI Annotation Results"
    )

    st.dataframe(
        st.session_state[
            "ncbi_annotation"
        ],
        use_container_width=True
    )


# ============================================================
# UNIPROT ANNOTATION
# ============================================================

st.header("8. UniProt Protein Annotation")


UNIPROT_BASE_URL = (
    "https://rest.uniprot.org/uniprotkb/search"
)


def get_uniprot_annotations(
    gene_id
):

    params = {

        "query": (
            f"gene_exact:{gene_id} "
            f"AND organism_id:3702"
        ),

        "format": "json",

        "fields": (
            "accession,"
            "id,"
            "gene_names,"
            "protein_name,"
            "organism_name,"
            "length,"
            "reviewed,"
            "cc_function"
        ),

        "size": 20
    }

    response = requests.get(
        UNIPROT_BASE_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    results = response.json().get(
        "results",
        []
    )

    if not results:

        return {
            "UniProt_Accession": "Not found",
            "UniProt_ID": "Not found",
            "UniProt_Protein": "Not found",
            "UniProt_Organism": "Not found",
            "Protein_Length": "Not found",
            "UniProt_Reviewed": "Not found",
            "UniProt_Function": "Not found",
            "UniProt_Status": "No match"
        }

    # --------------------------------------------------------
    # Prefer reviewed / Swiss-Prot entry
    # --------------------------------------------------------

    reviewed_results = [
        result
        for result in results
        if result.get("entryType") == "UniProtKB reviewed (Swiss-Prot)"
    ]

    if reviewed_results:

        record = reviewed_results[0]

    else:

        record = results[0]


    # --------------------------------------------------------
    # Basic fields
    # --------------------------------------------------------

    accession = record.get(
        "primaryAccession",
        "Not available"
    )

    entry_id = record.get(
        "uniProtkbId",
        "Not available"
    )


    # --------------------------------------------------------
    # Protein name
    # --------------------------------------------------------

    protein_description = record.get(
        "proteinDescription",
        {}
    )

    recommended_name = (
        protein_description
        .get("recommendedName", {})
    )

    protein_name = recommended_name.get(
        "fullName",
        {}
    ).get(
        "value",
        "Not available"
    )


    # --------------------------------------------------------
    # Organism
    # --------------------------------------------------------

    organism = record.get(
        "organism",
        {}
    )

    organism_name = organism.get(
        "scientificName",
        "Not available"
    )


    # --------------------------------------------------------
    # Protein length
    # --------------------------------------------------------

    protein_length = record.get(
        "sequence",
        {}
    ).get(
        "length",
        "Not available"
    )


    # --------------------------------------------------------
    # Reviewed status
    # --------------------------------------------------------

    reviewed = (
        record.get("entryType")
        == "UniProtKB reviewed (Swiss-Prot)"
    )


    # --------------------------------------------------------
    # Function annotation
    # --------------------------------------------------------

    comments = record.get(
        "comments",
        []
    )

    function_text = "Not available"

    for comment in comments:

        if comment.get("commentType") == "FUNCTION":

            texts = comment.get(
                "texts",
                []
            )

            if texts:

                function_text = texts[0].get(
                    "value",
                    "Not available"
                )

                break


    return {

        "UniProt_Accession":
            accession,

        "UniProt_ID":
            entry_id,

        "UniProt_Protein":
            protein_name,

        "UniProt_Organism":
            organism_name,

        "Protein_Length":
            protein_length,

        "UniProt_Reviewed":
            reviewed,

        "UniProt_Function":
            function_text,

        "UniProt_Status":
            "Annotated"
    }


# ============================================================
# UNIPROT ANNOTATION BUTTON
# ============================================================

if "ncbi_annotation" in st.session_state:

    if not significant_data.empty:

        if st.button(
            "🧬 Annotate Significant Genes with UniProt"
        ):

            uniprot_results = []

            progress_bar = st.progress(0)

            total = len(
                significant_data
            )

            for index, (_, row) in enumerate(
                significant_data.iterrows()
            ):

                gene_id = row[
                    "Gene_ID"
                ]

                try:

                    annotation = (
                        get_uniprot_annotations(
                            gene_id
                        )
                    )

                except Exception as e:

                    annotation = {

                        "UniProt_Accession":
                            "API error",

                        "UniProt_ID":
                            "API error",

                        "UniProt_Protein":
                            str(e),

                        "UniProt_Organism":
                            "API error",

                        "Protein_Length":
                            "API error",

                        "UniProt_Reviewed":
                            "API error",

                        "UniProt_Function":
                            "API error",

                        "UniProt_Status":
                            "Error"
                    }

                annotation[
                    "Gene_ID"
                ] = gene_id

                uniprot_results.append(
                    annotation
                )

                progress_bar.progress(
                    (index + 1) / total
                )

                time.sleep(0.2)

            uniprot_df = pd.DataFrame(
                uniprot_results
            )

            st.session_state[
                "uniprot_annotation"
            ] = uniprot_df

            st.success(
                "UniProt annotation completed."
            )


# ============================================================
# DISPLAY UNIPROT RESULTS
# ============================================================

if "uniprot_annotation" in st.session_state:

    st.subheader(
        "UniProt Annotation Results"
    )

    st.dataframe(
        st.session_state[
            "uniprot_annotation"
        ],
        use_container_width=True
    )


# ============================================================
# INTERPRO ANNOTATION
# ============================================================

st.header(
    "10. InterPro Protein Domain Annotation"
)


INTERPRO_BASE_URL = (
    "https://www.ebi.ac.uk/interpro/api"
)


def get_interpro_annotations(
    uniprot_accession
):
    """
    Retrieve InterPro entries associated
    with a UniProt protein.
    """

    url = (
        f"{INTERPRO_BASE_URL}"
        f"/entry/interpro/protein/uniprot/"
        f"{uniprot_accession}/"
    )

    params = {
        "page_size": 200
    }

    headers = {
        "Accept": "application/json"
    }

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=60
    )

    # --------------------------------------------------------
    # No data
    # --------------------------------------------------------

    if response.status_code == 204:

        return []


    # --------------------------------------------------------
    # Protein not found
    # --------------------------------------------------------

    if response.status_code == 404:

        return []


    response.raise_for_status()

    data_json = response.json()

    all_results = data_json.get(
        "results",
        []
    )


    # --------------------------------------------------------
    # Handle pagination
    # --------------------------------------------------------

    next_url = data_json.get(
        "next"
    )

    while next_url:

        next_response = requests.get(
            next_url,
            headers=headers,
            timeout=60
        )

        if next_response.status_code != 200:
            break

        next_data = (
            next_response.json()
        )

        all_results.extend(
            next_data.get(
                "results",
                []
            )
        )

        next_url = next_data.get(
            "next"
        )

        time.sleep(0.2)


    return all_results


def extract_interpro_match(
    match
):
    """
    Extract InterPro metadata and
    protein match regions.
    """

    metadata = match.get(
        "metadata",
        {}
    )


    # --------------------------------------------------------
    # InterPro accession
    # --------------------------------------------------------

    accession = metadata.get(
        "accession",
        "Not available"
    )


    # --------------------------------------------------------
    # InterPro name
    # --------------------------------------------------------

    name = metadata.get(
        "name",
        "Not available"
    )

    # Some API responses may return
    # name as a dictionary.

    if isinstance(name, dict):

        name = name.get(
            "name",
            "Not available"
        )


    # --------------------------------------------------------
    # Entry type
    # --------------------------------------------------------

    entry_type = metadata.get(
        "type",
        "Not available"
    )


    # --------------------------------------------------------
    # Source database
    # --------------------------------------------------------

    source_database = metadata.get(
        "source_database"
    )

    if source_database:

        source_text = str(
            source_database
        )

    else:

        member_databases = metadata.get(
            "member_databases"
        )

        if isinstance(
            member_databases,
            dict
        ) and member_databases:

            source_text = ", ".join(
                member_databases.keys()
            )

        else:

            source_text = "InterPro"


    # --------------------------------------------------------
    # Protein locations
    # --------------------------------------------------------

    regions = []


    # IMPORTANT:
    # The protein locations are nested
    # under match["proteins"].

    proteins = match.get(
        "proteins",
        []
    )


    for protein in proteins:

        locations = protein.get(
            "entry_protein_locations",
            []
        )

        for location in locations:

            fragments = location.get(
                "fragments",
                []
            )

            for fragment in fragments:

                start = fragment.get(
                    "start"
                )

                end = fragment.get(
                    "end"
                )

                if (
                    start is not None
                    and end is not None
                ):

                    regions.append(
                        f"{start}-{end}"
                    )


    # Remove duplicate regions
    regions = list(
        dict.fromkeys(regions)
    )


    if regions:

        region_text = ", ".join(
            regions
        )

    else:

        region_text = "Not provided"


    return {

        "InterPro_Accession":
            accession,

        "InterPro_Name":
            name,

        "InterPro_Type":
            entry_type,

        "InterPro_Source":
            source_text,

        "InterPro_Regions":
            region_text
    }


def annotate_uniprot_with_interpro(
    uniprot_accession
):
    """
    Retrieve all InterPro annotations
    for one UniProt protein.
    """

    matches = (
        get_interpro_annotations(
            uniprot_accession
        )
    )

    annotations = []


    for match in matches:

        annotation = (
            extract_interpro_match(
                match
            )
        )

        annotations.append(
            annotation
        )


    return annotations


# ============================================================
# INTERPRO BUTTON
# ============================================================

if "uniprot_annotation" not in st.session_state:

    st.info(
        "Run UniProt annotation first before using InterPro."
    )

else:

    if st.button(
        "🔬 Annotate Proteins with InterPro"
    ):

        uniprot_df = st.session_state[
            "uniprot_annotation"
        ].copy()


        interpro_results = []

        progress_bar = st.progress(0)

        total = len(
            uniprot_df
        )


        for index, row in enumerate(
            uniprot_df.itertuples(
                index=False
            )
        ):

            row_dict = row._asdict()


            # ------------------------------------------------
            # Gene ID
            # ------------------------------------------------

            gene_id = row_dict.get(
                "Gene_ID",
                "Unknown"
            )


            # ------------------------------------------------
            # UniProt accession
            # ------------------------------------------------

            accession = row_dict.get(
                "UniProt_Accession",
                None
            )


            # ------------------------------------------------
            # Missing accession
            # ------------------------------------------------

            if (
                pd.isna(accession)
                or str(accession).strip() == ""
                or str(accession).strip()
                in [
                    "Not found",
                    "Not available",
                    "N/A",
                    "API error"
                ]
            ):

                interpro_results.append({

                    "Gene_ID":
                        gene_id,

                    "UniProt_Accession":
                        accession,

                    "InterPro_Accession":
                        "Not available",

                    "InterPro_Name":
                        "Not available",

                    "InterPro_Type":
                        "Not available",

                    "InterPro_Source":
                        "Not available",

                    "InterPro_Regions":
                        "Not available"
                })

            else:

                accession = str(
                    accession
                ).strip()


                try:

                    annotations = (
                        annotate_uniprot_with_interpro(
                            accession
                        )
                    )


                    # ----------------------------------------
                    # No InterPro matches
                    # ----------------------------------------

                    if not annotations:

                        interpro_results.append({

                            "Gene_ID":
                                gene_id,

                            "UniProt_Accession":
                                accession,

                            "InterPro_Accession":
                                "No InterPro match",

                            "InterPro_Name":
                                "No InterPro match",

                            "InterPro_Type":
                                "Not available",

                            "InterPro_Source":
                                "Not available",

                            "InterPro_Regions":
                                "Not provided"
                        })


                    # ----------------------------------------
                    # InterPro matches found
                    # ----------------------------------------

                    else:

                        for annotation in annotations:

                            interpro_results.append({

                                "Gene_ID":
                                    gene_id,

                                "UniProt_Accession":
                                    accession,

                                "InterPro_Accession":
                                    annotation[
                                        "InterPro_Accession"
                                    ],

                                "InterPro_Name":
                                    annotation[
                                        "InterPro_Name"
                                    ],

                                "InterPro_Type":
                                    annotation[
                                        "InterPro_Type"
                                    ],

                                "InterPro_Source":
                                    annotation[
                                        "InterPro_Source"
                                    ],

                                "InterPro_Regions":
                                    annotation[
                                        "InterPro_Regions"
                                    ]
                            })


                except Exception as e:

                    interpro_results.append({

                        "Gene_ID":
                            gene_id,

                        "UniProt_Accession":
                            accession,

                        "InterPro_Accession":
                            "API error",

                        "InterPro_Name":
                            str(e),

                        "InterPro_Type":
                            "Error",

                        "InterPro_Source":
                            "InterPro API",

                        "InterPro_Regions":
                            "Not available"
                    })


            # ------------------------------------------------
            # Small delay between requests
            # ------------------------------------------------

            time.sleep(0.5)


            progress_bar.progress(
                (index + 1) / total
            )


        interpro_df = pd.DataFrame(
            interpro_results
        )


        st.session_state[
            "interpro_annotation"
        ] = interpro_df


        st.success(
            "InterPro annotation completed."
        )


# ============================================================
# DISPLAY INTERPRO RESULTS
# ============================================================

if "interpro_annotation" in st.session_state:

    st.subheader(
        "InterPro Annotation Results"
    )

    st.dataframe(
        st.session_state[
            "interpro_annotation"
        ],
        use_container_width=True
    )


    # --------------------------------------------------------
    # InterPro summary
    # --------------------------------------------------------

    st.subheader(
        "InterPro Summary"
    )

    interpro_df = st.session_state[
        "interpro_annotation"
    ]


    total_proteins = (
        interpro_df[
            "UniProt_Accession"
        ]
        .nunique()
    )


    matched_rows = interpro_df[
        interpro_df[
            "InterPro_Accession"
        ]
        .notna()
        &
        (
            interpro_df[
                "InterPro_Accession"
            ]
            != "No InterPro match"
        )
        &
        (
            interpro_df[
                "InterPro_Accession"
            ]
            != "API error"
        )
    ]


    matched_entries = (
        matched_rows[
            "InterPro_Accession"
        ]
        .nunique()
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Proteins processed",
            int(total_proteins)
        )


    with col2:

        st.metric(
            "InterPro entries found",
            int(matched_entries)
        )


    # --------------------------------------------------------
    # Download InterPro results
    # --------------------------------------------------------

    interpro_csv = (
        interpro_df
        .to_csv(index=False)
        .encode("utf-8")
    )


    st.download_button(
        label="⬇️ Download InterPro Results",
        data=interpro_csv,
        file_name="interpro_annotation_results.csv",
        mime="text/csv"
    )


# ============================================================
# CURRENT PIPELINE STATUS
# ============================================================

st.header("11. Current Pipeline Status")


st.markdown(
    """
### Current workflow

**Transcriptomics data**
↓  
**Differential expression analysis**
↓  
**Significant genes**
↓  
**NCBI gene annotation**
↓  
**UniProt protein annotation**
↓  
**InterPro protein/domain annotation**
↓  
**KEGG pathway mapping — next stage**
"""
)


st.info(
    "The next major stage is KEGG pathway mapping. "
    "InterPro results should be validated before moving "
    "to pathway-level interpretation."
)
# ============================================================
# DISPLAY KEGG RESULTS
# ============================================================

if "kegg_annotation" in st.session_state:

    st.subheader(
        "KEGG Pathway Mapping Results"
    )

    st.dataframe(
        st.session_state[
            "kegg_annotation"
        ],
        use_container_width=True
    )


    # --------------------------------------------------------
    # KEGG summary
    # --------------------------------------------------------

    st.subheader(
        "KEGG Summary"
    )

    kegg_df = st.session_state[
        "kegg_annotation"
    ]


    mapped_genes = kegg_df[
        kegg_df[
            "KEGG_Status"
        ] == "Mapped"
    ]["Gene_ID"].nunique()


    genes_without_mapping = kegg_df[
        kegg_df[
            "KEGG_Status"
        ] == "No mapping"
    ]["Gene_ID"].nunique()


    pathway_count = kegg_df[
        kegg_df[
            "KEGG_Status"
        ] == "Mapped"
    ]["KEGG_Pathway_ID"].nunique()


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Genes mapped to KEGG",
            int(mapped_genes)
        )


    with col2:

        st.metric(
            "Genes without KEGG mapping",
            int(genes_without_mapping)
        )


    with col3:

        st.metric(
            "Unique KEGG pathways",
            int(pathway_count)
        )


    # --------------------------------------------------------
    # Download KEGG results
    # --------------------------------------------------------

    kegg_csv = (
        kegg_df
        .to_csv(index=False)
        .encode("utf-8")
    )


    st.download_button(
        label="⬇️ Download KEGG Results",
        data=kegg_csv,
        file_name="kegg_pathway_mapping.csv",
        mime="text/csv"
    )
