import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import urllib.parse
import streamlit.components.v1 as components

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Transcriptomics KEGG Pipeline",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Transcriptomics & KEGG Pathway Analyzer")

st.write(
    "A pipeline for transcriptomics time-course analysis, "
    "gene annotation, protein domain annotation, and "
    "KEGG pathway visualization."
)

# ============================================================
# LOAD KOMAGATAELLA PHAFFII TPM DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(
        "komagataella_dummy_TPM_5timepoints.csv"
    )

data = load_data()

# Check that the expected columns are present.
required_columns = [
    "Gene_ID",
    "TPM_18hr",
    "TPM_60hr",
    "TPM_130hr",
    "TPM_164hr",
    "TPM_240hr"
]

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    st.error(
        "The Komagataella TPM file is missing these columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

# ============================================================
# INPUT DATA
# ============================================================

st.subheader("📊 Komagataella phaffii Time-Course TPM Data")

st.write(
    f"Loaded **{len(data)} genes** across "
    "**5 time points**: 18, 60, 130, 164 and 240 hours."
)

st.dataframe(
    data,
    use_container_width=True
)

# ============================================================
# TIME-COURSE ANALYSIS
# ============================================================

st.header("1️⃣ Time-Course Expression Analysis")

timepoint_columns = {
    "18 hr": "TPM_18hr",
    "60 hr": "TPM_60hr",
    "130 hr": "TPM_130hr",
    "164 hr": "TPM_164hr",
    "240 hr": "TPM_240hr"
}

selected_baseline = st.selectbox(
    "Baseline time point",
    options=list(timepoint_columns.keys()),
    index=0
)

selected_comparison = st.selectbox(
    "Comparison time point",
    options=list(timepoint_columns.keys()),
    index=4
)

baseline_column = timepoint_columns[selected_baseline]
comparison_column = timepoint_columns[selected_comparison]

if baseline_column == comparison_column:
    st.warning(
        "Please select two different time points."
    )
    st.stop()

# ------------------------------------------------------------
# Fold change
# ------------------------------------------------------------

# Small pseudocount prevents division by zero if a future
# dataset contains a TPM value of exactly zero.
pseudocount = 0.01

data["Fold_Change"] = (
    (data[comparison_column] + pseudocount)
    /
    (data[baseline_column] + pseudocount)
)

data["Log2_Fold_Change"] = np.log2(
    data["Fold_Change"]
)

# ------------------------------------------------------------
# Exploratory classification
# ------------------------------------------------------------

st.info(
    "This TPM file contains one value per gene at each time point, "
    "not biological replicates. Therefore this Stage 1 prototype "
    "does **not** calculate a statistical p-value or FDR. "
    "The regulation labels below are based only on the selected "
    "log₂ fold-change threshold."
)

log2fc_threshold = st.slider(
    "Absolute log₂ fold-change threshold",
    min_value=0.5,
    max_value=3.0,
    value=1.0,
    step=0.1
)

def classify_gene(row):
    log2fc = row["Log2_Fold_Change"]

    if log2fc >= log2fc_threshold:
        return "Upregulated"

    if log2fc <= -log2fc_threshold:
        return "Downregulated"

    return "Not significant"

data["Regulation"] = data.apply(
    classify_gene,
    axis=1
)

# Keep these columns so the existing annotation sections below
# can continue to use the same Gene_ID / Log2_Fold_Change /
# Regulation structure.
data["P_Value"] = np.nan
data["Adjusted_P_Value"] = np.nan

# ============================================================
# EXPRESSION RESULTS
# ============================================================

st.subheader("Expression Comparison Results")

de_columns = [
    "Gene_ID",
    baseline_column,
    comparison_column,
    "Fold_Change",
    "Log2_Fold_Change",
    "Regulation"
]

st.dataframe(
    data[de_columns],
    use_container_width=True
)

# ============================================================
# SUMMARY
# ============================================================

st.subheader("📈 Time-Course Comparison Summary")

up_count = (
    data["Regulation"] == "Upregulated"
).sum()

down_count = (
    data["Regulation"] == "Downregulated"
).sum()

not_sig_count = (
    data["Regulation"] == "Not significant"
).sum()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Upregulated",
        up_count
    )

with col2:
    st.metric(
        "Downregulated",
        down_count
    )

with col3:
    st.metric(
        "Below threshold",
        not_sig_count
    )

# ============================================================
# CANDIDATE GENES
# ============================================================

candidate_genes = data[
    data["Regulation"] != "Not significant"
].copy()

# Keep the old variable name because the annotation sections
# below already use significant_genes.
significant_genes = candidate_genes.copy()

st.subheader(
    "🎯 Genes Showing Expression Change"
)

if len(candidate_genes) > 0:

    st.dataframe(
        candidate_genes[
            [
                "Gene_ID",
                baseline_column,
                comparison_column,
                "Log2_Fold_Change",
                "Regulation"
            ]
        ],
        use_container_width=True
    )

else:

    st.info(
        "No genes meet the current log₂ fold-change threshold."
    )

# ============================================================
# QUICK CHECK
# ============================================================

st.subheader("🧬 Gene ID Check")

pas_count = (
    data["Gene_ID"]
    .astype(str)
    .str.startswith("PAS_")
).sum()

st.success(
    f"Detected **{pas_count} PAS genes** out of "
    f"**{len(data)} total genes**."
)

if pas_count != len(data):
    st.warning(
        "Some Gene_ID values do not start with PAS_. "
        "Check the input file before annotation."
    )



# ============================================================
# NCBI CONFIGURATION
# ============================================================

st.header(
    "2️⃣ NCBI Gene Annotation"
)


NCBI_BASE_URL = (
    "https://eutils.ncbi.nlm.nih.gov/"
    "entrez/eutils/"
)

NCBI_TOOL_NAME = (
    "TranscriptomicsKEGGPipeline"
)


# ------------------------------------------------------------
# Optional NCBI API key
#
# If you later add these to Streamlit secrets:
#
# NCBI_API_KEY = "your_api_key"
# NCBI_EMAIL = "your_email@example.com"
#
# the application will automatically use them.
# ------------------------------------------------------------

try:

    NCBI_API_KEY = st.secrets.get(
        "NCBI_API_KEY",
        ""
    )

    NCBI_EMAIL = st.secrets.get(
        "NCBI_EMAIL",
        ""
    )

except Exception:

    NCBI_API_KEY = ""

    NCBI_EMAIL = ""


# ============================================================
# NCBI REQUEST HELPER
# ============================================================

def ncbi_request(
    endpoint,
    params,
    max_retries=5
):

    """
    Send an NCBI E-utilities request.

    Features:
    - automatic retry for HTTP 429
    - exponential backoff
    - NCBI tool identifier
    - optional email
    - optional API key
    """


    request_params = params.copy()


    # NCBI recommends including tool.

    request_params[
        "tool"
    ] = NCBI_TOOL_NAME


    # Include email if configured.

    if NCBI_EMAIL:

        request_params[
            "email"
        ] = NCBI_EMAIL


    # Include API key if configured.

    if NCBI_API_KEY:

        request_params[
            "api_key"
        ] = NCBI_API_KEY


    for attempt in range(
        max_retries
    ):

        try:

            response = requests.get(

                NCBI_BASE_URL + endpoint,

                params=request_params,

                timeout=60
            )


            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            if response.status_code == 200:

                return response


            # ------------------------------------------------
            # RATE LIMIT
            # ------------------------------------------------

            if response.status_code == 429:

                wait_time = (
                    2 ** attempt
                )

                time.sleep(
                    wait_time
                )

                continue


            # ------------------------------------------------
            # OTHER HTTP ERROR
            # ------------------------------------------------

            response.raise_for_status()


        except requests.exceptions.RequestException:

            if attempt == (
                max_retries - 1
            ):

                raise


            wait_time = (
                2 ** attempt
            )

            time.sleep(
                wait_time
            )


    raise RuntimeError(
        "NCBI request failed after "
        f"{max_retries} attempts."
    )


# ============================================================
# NCBI SEARCH FOR ONE GENE
# ============================================================

def search_ncbi_gene(
    gene_id
):

    """
    Search NCBI Gene for one gene identifier.

    We keep one search per gene because this allows us
    to preserve the mapping:

        input Gene_ID → NCBI Gene UID

    The requests are deliberately spaced to stay below
    the unauthenticated NCBI request rate.
    """


    search_params = {

        "db":
            "gene",

        "term":
            (
                f'"{gene_id}"[All Fields]'
            ),

        "retmode":
            "json",

        "retmax":
            5
    }


    response = ncbi_request(

        "esearch.fcgi",

        search_params
    )


    search_data = (
        response.json()
    )


    ids = (

        search_data

        .get(
            "esearchresult",
            {}
        )

        .get(
            "idlist",
            []
        )
    )


    return ids


# ============================================================
# BATCH NCBI SUMMARY
# ============================================================

def get_ncbi_summaries(
    ncbi_gene_ids
):

    """
    Retrieve NCBI Gene summaries in ONE request.

    ESummary accepts a comma-separated list of UIDs,
    which dramatically reduces the number of requests.
    """


    if not ncbi_gene_ids:

        return {}


    # Remove duplicate IDs while preserving order.

    unique_ids = list(
        dict.fromkeys(
            ncbi_gene_ids
        )
    )


    summaries = {}


    # ESummary supports a list of IDs.
    #
    # We use batches of 100 to remain comfortably
    # within practical request sizes.

    batch_size = 100


    for start in range(
        0,
        len(unique_ids),
        batch_size
    ):

        batch = unique_ids[
            start:
            start + batch_size
        ]


        summary_params = {

            "db":
                "gene",

            "id":
                ",".join(batch),

            "retmode":
                "json"
        }


        response = ncbi_request(

            "esummary.fcgi",

            summary_params
        )


        summary_data = (
            response.json()
        )


        result = summary_data.get(
            "result",
            {}
        )


        for ncbi_id in batch:

            if ncbi_id in result:

                summaries[
                    ncbi_id
                ] = result[
                    ncbi_id
                ]


        # Small pause between batches.

        time.sleep(
            0.5
        )


    return summaries


# ============================================================
# EXTRACT NCBI ANNOTATION
# ============================================================

def extract_ncbi_annotation(
    gene_id,
    ncbi_ids,
    summaries
):

    """
    Convert NCBI Gene data into one annotation row.

    If multiple NCBI matches exist, we do not silently
    pretend that there is only one match.
    """


    if not ncbi_ids:

        return {

            "Gene_ID":
                gene_id,

            "NCBI_Gene_ID":
                "Not found",

            "Gene_Name":
                "Not found",

            "Gene_Description":
                "Not found",

            "Organism":
                "Not found",

            "Chromosome":
                "Not found",

            "NCBI_Status":
                "No NCBI match"
        }


    # --------------------------------------------------------
    # If multiple matches occur, use the first matching
    # record but explicitly record that multiple matches
    # were returned.
    # --------------------------------------------------------

    ncbi_gene_id = ncbi_ids[0]


    if ncbi_gene_id not in summaries:

        return {

            "Gene_ID":
                gene_id,

            "NCBI_Gene_ID":
                ncbi_gene_id,

            "Gene_Name":
                "Not available",

            "Gene_Description":
                "Not available",

            "Organism":
                "Not available",

            "Chromosome":
                "Not available",

            "NCBI_Status":
                "Summary unavailable"
        }


    result = summaries[
        ncbi_gene_id
    ]


    organism = result.get(
        "organism",
        {}
    )


    if not isinstance(
        organism,
        dict
    ):

        organism = {}


    organism_name = organism.get(
        "scientificname",
        "Not available"
    )


    if len(ncbi_ids) > 1:

        status = (
            f"Matched; "
            f"{len(ncbi_ids)} NCBI matches returned"
        )

    else:

        status = "Matched"


    return {

        "Gene_ID":
            gene_id,

        "NCBI_Gene_ID":
            ncbi_gene_id,

        "Gene_Name":
            result.get(
                "name",
                "Not available"
            ),

        "Gene_Description":
            result.get(
                "description",
                "Not available"
            ),

        "Organism":
            organism_name,

        "Chromosome":
            result.get(
                "chromosome",
                "Not available"
            ),

        "NCBI_Status":
            status
    }


# ============================================================
# RUN NCBI ANNOTATION
# ============================================================

if st.button(
    "🔎 Annotate Significant Genes with NCBI"
):

    if len(significant_genes) == 0:

        st.warning(
            "There are no significant genes to annotate."
        )

    else:

        gene_ids = (
            significant_genes[
                "Gene_ID"
            ]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )


        st.info(
            f"Searching NCBI for {len(gene_ids)} "
            "significant gene(s). "
            "Requests are deliberately spaced to "
            "avoid NCBI rate limiting."
        )


        # ----------------------------------------------------
        # STEP 1
        # Search each gene
        # ----------------------------------------------------

        gene_to_ncbi_ids = {}


        progress = st.progress(0)

        status_text = st.empty()


        total_genes = len(
            gene_ids
        )


        for i, gene_id in enumerate(
            gene_ids
        ):

            status_text.write(
                f"🔎 Searching NCBI: "
                f"{gene_id}"
            )


            try:

                ncbi_ids = (
                    search_ncbi_gene(
                        gene_id
                    )
                )


                gene_to_ncbi_ids[
                    gene_id
                ] = ncbi_ids


            except Exception as e:

                gene_to_ncbi_ids[
                    gene_id
                ] = []


                st.warning(
                    f"NCBI search failed for "
                    f"{gene_id}: {e}"
                )


            progress.progress(
                (i + 1)
                /
                total_genes
            )


            # ------------------------------------------------
            # IMPORTANT:
            #
            # This pause applies BETWEEN NCBI requests.
            #
            # 0.5 sec = approximately 2 requests/sec.
            # ------------------------------------------------

            time.sleep(
                0.5
            )


        # ----------------------------------------------------
        # STEP 2
        # Collect all NCBI Gene UIDs
        # ----------------------------------------------------

        all_ncbi_ids = []


        for ids in (
            gene_to_ncbi_ids.values()
        ):

            all_ncbi_ids.extend(
                ids
            )


        all_ncbi_ids = list(
            dict.fromkeys(
                all_ncbi_ids
            )
        )


        # ----------------------------------------------------
        # STEP 3
        # ONE batched ESummary request
        # ----------------------------------------------------

        st.write(
            f"📦 Retrieving summaries for "
            f"{len(all_ncbi_ids)} NCBI Gene record(s)..."
        )


        try:

            ncbi_summaries = (
                get_ncbi_summaries(
                    all_ncbi_ids
                )
            )

        except Exception as e:

            ncbi_summaries = {}

            st.error(
                "NCBI summary retrieval failed: "
                f"{e}"
            )


        # ----------------------------------------------------
        # STEP 4
        # Build final annotation table
        # ----------------------------------------------------

        ncbi_results = []


        for gene_id in gene_ids:

            annotation = (
                extract_ncbi_annotation(

                    gene_id,

                    gene_to_ncbi_ids.get(
                        gene_id,
                        []
                    ),

                    ncbi_summaries
                )
            )


            ncbi_results.append(
                annotation
            )


        ncbi_df = pd.DataFrame(
            ncbi_results
        )


        # ----------------------------------------------------
        # Save in Streamlit session
        # ----------------------------------------------------

        st.session_state[
            "ncbi_annotation"
        ] = ncbi_df


        status_text.success(
            "✅ NCBI annotation completed."
        )


# ============================================================
# DISPLAY NCBI RESULTS
# ============================================================

if "ncbi_annotation" in st.session_state:

    st.subheader(
        "NCBI Annotation Results"
    )


    ncbi_display = st.session_state[
        "ncbi_annotation"
    ]


    st.dataframe(
        ncbi_display,
        use_container_width=True
    )


    # --------------------------------------------------------
    # NCBI status summary
    # --------------------------------------------------------

    matched_count = (
        ncbi_display[
            "NCBI_Status"
        ]
        .astype(str)
        .str.startswith(
            "Matched"
        )
        .sum()
    )


    no_match_count = (
        ncbi_display[
            "NCBI_Status"
        ]
        ==
        "No NCBI match"
    ).sum()


    st.write(
        f"**NCBI summary:** "
        f"{matched_count} matched, "
        f"{no_match_count} not found."
    )


# ============================================================
# UNIPROT ANNOTATION
# ============================================================

st.header(
    "3️⃣ UniProt Protein Annotation"
)

UNIPROT_URL = (
    "https://rest.uniprot.org/"
    "uniprotkb/search"
)

# Komagataella phaffii GS115 / ATCC 20864
# UniProt/NCBI taxonomy ID: 644223
UNIPROT_TAXONOMY_ID = "644223"


@st.cache_data(ttl=3600)
def get_uniprot_annotation(gene_id):
    """
    Find a UniProtKB entry for a Komagataella phaffii
    ordered locus name such as PAS_chr4_0821.

    Matching strategy:
    1. Search the PAS locus as a gene field within the
       Komagataella phaffii GS115 taxonomy.
    2. Search the exact locus text within the same organism.
    3. Prefer a result whose UniProt gene metadata contains
       the input PAS locus exactly.
    4. Prefer reviewed Swiss-Prot if more than one exact
       result is available.
    """

    gene_id = str(gene_id).strip()

    if not gene_id:
        return {
            "UniProt_Accession": "Not found",
            "UniProt_ID": "Not found",
            "UniProt_Protein": "Not found",
            "UniProt_Organism": "Not found",
            "Protein_Length": "Not found",
            "UniProt_Reviewed": "Not found",
            "UniProt_Function": "Not available",
            "UniProt_Annotation_Score": "Not available",
            "UniProt_Status": "Empty Gene_ID"
        }

    fields = (
        "accession,"
        "id,"
        "gene_names,"
        "protein_name,"
        "organism_name,"
        "length,"
        "reviewed,"
        "cc_function,"
        "annotation_score"
    )

    # --------------------------------------------------------
    # Try several UniProt search forms.
    #
    # The important change is that all searches are restricted
    # to Komagataella phaffii GS115, not Arabidopsis.
    # --------------------------------------------------------

    queries = [
        (
            f"gene:{gene_id} "
            f"AND organism_id:{UNIPROT_TAXONOMY_ID}"
        ),
        (
            f"gene_exact:{gene_id} "
            f"AND organism_id:{UNIPROT_TAXONOMY_ID}"
        ),
        (
            f'"{gene_id}" '
            f"AND organism_id:{UNIPROT_TAXONOMY_ID}"
        )
    ]

    all_results = []

    for query in queries:

        params = {
            "query": query,
            "format": "json",
            "fields": fields,
            "size": 20
        }

        response = requests.get(
            UNIPROT_URL,
            params=params,
            timeout=30
        )

        if response.status_code != 200:
            continue

        data_json = response.json()

        results = data_json.get(
            "results",
            []
        )

        if results:
            all_results.extend(results)

    # --------------------------------------------------------
    # Remove duplicate UniProt accessions.
    # --------------------------------------------------------

    unique_results = []
    seen_accessions = set()

    for result in all_results:

        accession = result.get(
            "primaryAccession"
        )

        if not accession:
            continue

        if accession in seen_accessions:
            continue

        seen_accessions.add(accession)
        unique_results.append(result)

    if not unique_results:

        return {
            "UniProt_Accession": "Not found",
            "UniProt_ID": "Not found",
            "UniProt_Protein": "Not found",
            "UniProt_Organism": "Not found",
            "Protein_Length": "Not found",
            "UniProt_Reviewed": "Not found",
            "UniProt_Function": "Not available",
            "UniProt_Annotation_Score": "Not available",
            "UniProt_Status": (
                "No UniProt match for "
                f"{gene_id} in Komagataella phaffii GS115"
            )
        }

    # --------------------------------------------------------
    # Score each result for an exact PAS locus match.
    # --------------------------------------------------------

    def result_has_exact_locus(result):

        genes = result.get(
            "genes",
            []
        )

        for gene in genes:

            # Main gene name
            gene_name = gene.get(
                "geneName",
                {}
            )

            if (
                isinstance(gene_name, dict)
                and
                gene_name.get("value") == gene_id
            ):
                return True

            # Ordered locus names
            ordered_names = gene.get(
                "orderedLocusNames",
                []
            )

            for item in ordered_names:

                if (
                    isinstance(item, dict)
                    and
                    item.get("value") == gene_id
                ):
                    return True

            # ORF names
            orf_names = gene.get(
                "orfNames",
                []
            )

            for item in orf_names:

                if (
                    isinstance(item, dict)
                    and
                    item.get("value") == gene_id
                ):
                    return True

            # Synonyms
            synonyms = gene.get(
                "synonyms",
                []
            )

            for item in synonyms:

                if (
                    isinstance(item, dict)
                    and
                    item.get("value") == gene_id
                ):
                    return True

        return False

    exact_results = [
        result
        for result in unique_results
        if result_has_exact_locus(result)
    ]

    if exact_results:
        candidate_results = exact_results
    else:
        candidate_results = unique_results

    # Prefer reviewed Swiss-Prot when available.
    reviewed_results = [
        result
        for result in candidate_results
        if result.get("entryType")
        == "UniProtKB reviewed (Swiss-Prot)"
    ]

    if reviewed_results:
        result = reviewed_results[0]
    else:
        result = candidate_results[0]

    accession = result.get(
        "primaryAccession",
        "Not available"
    )

    uniprot_id = result.get(
        "uniProtkbId",
        "Not available"
    )

    protein_description = result.get(
        "proteinDescription",
        {}
    )

    recommended_name = protein_description.get(
        "recommendedName",
        {}
    )

    protein_name = (
        recommended_name
        .get(
            "fullName",
            {}
        )
        .get(
            "value",
            "Not available"
        )
    )

    # Some unreviewed records use a submitted name.
    if protein_name == "Not available":

        submitted_names = (
            protein_description
            .get(
                "submissionNames",
                []
            )
        )

        if submitted_names:

            protein_name = (
                submitted_names[0]
                .get(
                    "fullName",
                    {}
                )
                .get(
                    "value",
                    "Not available"
                )
            )

    organism = (
        result.get(
            "organism",
            {}
        )
        .get(
            "scientificName",
            "Not available"
        )
    )

    protein_length = (
        result.get(
            "sequence",
            {}
        )
        .get(
            "length",
            "Not available"
        )
    )

    reviewed = (
        result.get(
            "entryType",
            ""
        )
        ==
        "UniProtKB reviewed (Swiss-Prot)"
    )

    comments = result.get(
        "comments",
        []
    )

    function_text = "Not available"

    for comment in comments:

        if (
            comment.get("commentType")
            ==
            "FUNCTION"
        ):

            texts = comment.get(
                "texts",
                []
            )

            if texts:

                function_text = (
                    texts[0]
                    .get(
                        "value",
                        "Not available"
                    )
                )

            break

    annotation_score = result.get(
        "annotationScore",
        "Not available"
    )

    if exact_results:

        if reviewed:
            status = (
                "Matched exact PAS locus "
                "(reviewed Swiss-Prot)"
            )
        else:
            status = (
                "Matched exact PAS locus "
                "(UniProtKB)"
            )

    else:

        status = (
            "Matched within Komagataella phaffii; "
            "exact PAS locus not confirmed in returned "
            "gene metadata"
        )

    return {
        "UniProt_Accession": accession,
        "UniProt_ID": uniprot_id,
        "UniProt_Protein": protein_name,
        "UniProt_Organism": organism,
        "Protein_Length": protein_length,
        "UniProt_Reviewed": (
            "Yes"
            if reviewed
            else
            "No"
        ),
        "UniProt_Function": function_text,
        "UniProt_Annotation_Score": annotation_score,
        "UniProt_Status": status
    }


# ============================================================
# RUN UNIPROT
# ============================================================

if st.button(
    "🧬 Annotate Significant Genes with UniProt"
):

    if len(significant_genes) == 0:

        st.warning(
            "There are no genes above the current "
            "log₂ fold-change threshold."
        )

    else:

        uniprot_results = []

        progress = st.progress(0)

        status_text = st.empty()

        total = len(
            significant_genes
        )

        for i, gene_id in enumerate(
            significant_genes["Gene_ID"]
        ):

            status_text.write(
                f"🧬 Searching UniProt: {gene_id}"
            )

            try:

                annotation = (
                    get_uniprot_annotation(
                        gene_id
                    )
                )

            except Exception as e:

                annotation = {

                    "UniProt_Accession":
                        "Error",

                    "UniProt_ID":
                        "Error",

                    "UniProt_Protein":
                        "Error",

                    "UniProt_Organism":
                        "Error",

                    "Protein_Length":
                        "Error",

                    "UniProt_Reviewed":
                        "Error",

                    "UniProt_Function":
                        str(e),

                    "UniProt_Annotation_Score":
                        "Error",

                    "UniProt_Status":
                        "Request failed"
                }

            annotation[
                "Gene_ID"
            ] = gene_id

            uniprot_results.append(
                annotation
            )

            progress.progress(
                (i + 1) / total
            )

            # Be polite to the UniProt REST service.
            time.sleep(0.3)

        status_text.success(
            f"UniProt search completed for {total} gene(s)."
        )

        uniprot_df = pd.DataFrame(
            uniprot_results
        )

        st.session_state[
            "uniprot_annotation"
        ] = uniprot_df


# ============================================================
# DISPLAY UNIPROT
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

    matched_count = (
        ~st.session_state[
            "uniprot_annotation"
        ][
            "UniProt_Status"
        ].astype(str).str.contains(
            "No UniProt match|Request failed",
            na=False
        )
    ).sum()

    total_results = len(
        st.session_state[
            "uniprot_annotation"
        ]
    )

    st.write(
        f"**UniProt summary:** "
        f"{matched_count} matched out of "
        f"{total_results} queried."
    )



# ============================================================
# INTERPRO ANNOTATION
# ============================================================

st.header(
    "4️⃣ InterPro Protein Domain Annotation"
)


INTERPRO_BASE_URL = (
    "https://www.ebi.ac.uk/interpro/api"
)


def get_interpro_annotations(
    uniprot_accession
):

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


    if response.status_code in [
        204,
        404
    ]:

        return []


    response.raise_for_status()


    data_json = response.json()


    all_results = data_json.get(
        "results",
        []
    )


    next_url = data_json.get(
        "next"
    )


    while next_url:

        next_response = requests.get(
            next_url,
            headers=headers,
            timeout=60
        )


        if (
            next_response.status_code
            != 200
        ):

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


        time.sleep(
            0.4
        )


    return all_results


def extract_interpro_match(
    match
):

    metadata = match.get(
        "metadata",
        {}
    )


    accession = metadata.get(
        "accession",
        "Not available"
    )


    name = metadata.get(
        "name",
        "Not available"
    )


    if isinstance(
        name,
        dict
    ):

        name = name.get(
            "name",
            "Not available"
        )


    entry_type = metadata.get(
        "type",
        "Not available"
    )


    source_database = metadata.get(
        "source_database"
    )


    if source_database:

        source_text = str(
            source_database
        )

    else:

        member_databases = (
            metadata.get(
                "member_databases"
            )
        )


        if (
            isinstance(
                member_databases,
                dict
            )
            and member_databases
        ):

            source_text = (
                ", ".join(
                    member_databases.keys()
                )
            )

        else:

            source_text = "InterPro"


    regions = []


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
                    and
                    end is not None
                ):

                    regions.append(
                        f"{start}-{end}"
                    )


    regions = list(
        dict.fromkeys(
            regions
        )
    )


    region_text = (

        ", ".join(regions)

        if regions

        else

        "Not provided"
    )


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


# ============================================================
# RUN INTERPRO
# ============================================================

if st.button(
    "🔬 Annotate Significant Proteins with InterPro"
):

    if (
        "uniprot_annotation"
        not in st.session_state
    ):

        st.warning(
            "Run UniProt annotation first."
        )

    else:

        uniprot_df = (
            st.session_state[
                "uniprot_annotation"
            ]
        )


        interpro_results = []


        valid_rows = uniprot_df[
            ~uniprot_df[
                "UniProt_Accession"
            ].isin(
                [
                    "Not found",
                    "Error"
                ]
            )
        ]


        progress = st.progress(0)

        total = len(
            valid_rows
        )


        for i, (
            _,
            row
        ) in enumerate(
            valid_rows.iterrows()
        ):

            gene_id = row[
                "Gene_ID"
            ]

            accession = row[
                "UniProt_Accession"
            ]


            try:

                matches = (
                    get_interpro_annotations(
                        accession
                    )
                )


                if not matches:

                    interpro_results.append({

                        "Gene_ID":
                            gene_id,

                        "UniProt_Accession":
                            accession,

                        "InterPro_Accession":
                            "Not found",

                        "InterPro_Name":
                            "No InterPro match",

                        "InterPro_Type":
                            "Not available",

                        "InterPro_Source":
                            "Not available",

                        "InterPro_Regions":
                            "Not provided"
                    })


                else:

                    for match in matches:

                        parsed = (
                            extract_interpro_match(
                                match
                            )
                        )


                        parsed[
                            "Gene_ID"
                        ] = gene_id


                        parsed[
                            "UniProt_Accession"
                        ] = accession


                        interpro_results.append(
                            parsed
                        )


            except Exception as e:

                interpro_results.append({

                    "Gene_ID":
                        gene_id,

                    "UniProt_Accession":
                        accession,

                    "InterPro_Accession":
                        "Error",

                    "InterPro_Name":
                        "Error",

                    "InterPro_Type":
                        "Error",

                    "InterPro_Source":
                        "Error",

                    "InterPro_Regions":
                        str(e)
                })


            if total > 0:

                progress.progress(
                    (i + 1)
                    /
                    total
                )


            time.sleep(
                0.4
            )


        interpro_df = pd.DataFrame(
            interpro_results
        )


        st.session_state[
            "interpro_annotation"
        ] = interpro_df


# ============================================================
# DISPLAY INTERPRO
# ============================================================

if "interpro_annotation" in st.session_state:

    st.subheader(
        "InterPro Results"
    )

    st.dataframe(
        st.session_state[
            "interpro_annotation"
        ],
        use_container_width=True
    )


# ============================================================
# KEGG
# ============================================================

st.header(
    "5️⃣ KEGG Gene & Pathway Mapping"
)

KEGG_BASE_URL = "https://rest.kegg.jp"

# Komagataella phaffii GS115
# KEGG organism code: ppa
# Taxonomy ID: 644223
KEGG_ORGANISM = "ppa"


@st.cache_data(ttl=3600)
def kegg_pas_gene_to_entry(gene_id):
    """
    Directly verify a PAS locus against the ppa KEGG organism.

    This is more reliable for this dataset than going:
        PAS locus -> UniProt -> KEGG

    because KEGG itself uses PAS locus IDs for ppa genes.
    """

    gene_id = str(gene_id).strip()

    if not gene_id:
        return []

    # First try the exact organism-specific KEGG entry.
    url = (
        f"{KEGG_BASE_URL}/get/"
        f"{KEGG_ORGANISM}:{gene_id}"
    )

    response = requests.get(
        url,
        timeout=30
    )

    if response.status_code == 200:

        return [{
            "KEGG_Gene_ID":
                f"{KEGG_ORGANISM}:{gene_id}",
            "Gene_ID":
                gene_id
        }]

    # Fallback: search the ppa gene database.
    search_url = (
        f"{KEGG_BASE_URL}/find/"
        f"{KEGG_ORGANISM}/"
        f"{gene_id}"
    )

    response = requests.get(
        search_url,
        timeout=30
    )

    if response.status_code != 200:
        return []

    mappings = []

    for line in response.text.strip().splitlines():

        parts = line.split("\t")

        if len(parts) < 1:
            continue

        kegg_gene = parts[0]

        if kegg_gene.lower() == (
            f"{KEGG_ORGANISM}:{gene_id}".lower()
        ):

            mappings.append({
                "KEGG_Gene_ID":
                    kegg_gene,
                "Gene_ID":
                    gene_id
            })

    return mappings


@st.cache_data(ttl=3600)
def kegg_uniprot_to_gene(uniprot_accession):
    """
    Secondary mapping route:
        UniProt accession -> ppa KEGG gene.

    This is useful when the PAS locus is not directly
    represented by the KEGG lookup.
    """

    if not uniprot_accession:
        return []

    accession = str(
        uniprot_accession
    ).strip()

    if (
        accession in [
            "",
            "Not found",
            "Not available",
            "Error"
        ]
    ):
        return []

    url = (
        f"{KEGG_BASE_URL}/conv/"
        f"{KEGG_ORGANISM}/"
        f"uniprot:{accession}"
    )

    response = requests.get(
        url,
        timeout=30
    )

    if response.status_code != 200:
        return []

    mappings = []

    for line in response.text.strip().splitlines():

        parts = line.split("\t")

        if len(parts) != 2:
            continue

        mappings.append({
            "UniProt_ID":
                parts[0],
            "KEGG_Gene_ID":
                parts[1]
        })

    return mappings


@st.cache_data(ttl=3600)
def get_kegg_pathways(kegg_gene_id):
    """
    Find all pathways associated with a ppa KEGG gene.
    """

    if not kegg_gene_id:
        return []

    url = (
        f"{KEGG_BASE_URL}/link/"
        f"pathway/"
        f"{kegg_gene_id}"
    )

    response = requests.get(
        url,
        timeout=30
    )

    if response.status_code != 200:
        return []

    pathways = []

    for line in response.text.strip().splitlines():

        parts = line.split("\t")

        if len(parts) != 2:
            continue

        pathways.append({
            "KEGG_Gene_ID":
                parts[0],
            "KEGG_Pathway_ID":
                parts[1].replace(
                    "path:",
                    ""
                )
        })

    return pathways


@st.cache_data(ttl=3600)
def get_kegg_pathway_names():
    """
    Download the complete ppa pathway-name table once
    and cache it.
    """

    url = (
        f"{KEGG_BASE_URL}/list/"
        f"pathway/"
        f"{KEGG_ORGANISM}"
    )

    response = requests.get(
        url,
        timeout=30
    )

    response.raise_for_status()

    pathway_names = {}

    for line in response.text.strip().splitlines():

        parts = line.split("\t")

        if len(parts) < 2:
            continue

        pathway_id = (
            parts[0]
            .replace(
                "path:",
                ""
            )
            .strip()
        )

        pathway_name = parts[1].strip()

        pathway_names[
            pathway_id
        ] = pathway_name

    return pathway_names


def get_kegg_gene_entry_text(kegg_gene_id):

    if not kegg_gene_id:
        return ""

    url = (
        f"{KEGG_BASE_URL}/get/"
        f"{kegg_gene_id}"
    )

    response = requests.get(
        url,
        timeout=30
    )

    if response.status_code != 200:
        return ""

    return response.text


# ============================================================
# RUN KEGG MAPPING
# ============================================================

if st.button(
    "🧬 Map Significant PAS Genes to KEGG"
):

    if len(significant_genes) == 0:

        st.warning(
            "There are no genes above the current "
            "log₂ fold-change threshold."
        )

    else:

        kegg_results = []

        pathway_names = (
            get_kegg_pathway_names()
        )

        progress = st.progress(0)

        status_text = st.empty()

        total = len(
            significant_genes
        )

        # Use UniProt results if they already exist.
        uniprot_df = st.session_state.get(
            "uniprot_annotation"
        )

        if uniprot_df is not None:

            uniprot_lookup = (
                uniprot_df
                .drop_duplicates(
                    "Gene_ID"
                )
                .set_index(
                    "Gene_ID"
                )
                .to_dict(
                    "index"
                )
            )

        else:

            uniprot_lookup = {}

        for i, gene_id in enumerate(
            significant_genes["Gene_ID"]
        ):

            gene_id = str(
                gene_id
            ).strip()

            status_text.write(
                f"🔎 Mapping KEGG: {gene_id}"
            )

            # ------------------------------------------------
            # Route 1: direct PAS -> KEGG
            # ------------------------------------------------

            direct_mappings = (
                kegg_pas_gene_to_entry(
                    gene_id
                )
            )

            # ------------------------------------------------
            # Route 2: UniProt -> KEGG
            # ------------------------------------------------

            uniprot_accession = (
                uniprot_lookup
                .get(
                    gene_id,
                    {}
                )
                .get(
                    "UniProt_Accession",
                    ""
                )
            )

            uniprot_mappings = []

            if not direct_mappings:

                uniprot_mappings = (
                    kegg_uniprot_to_gene(
                        uniprot_accession
                    )
                )

            # ------------------------------------------------
            # Combine mappings without duplicates.
            # ------------------------------------------------

            combined_mappings = []

            seen_kegg_ids = set()

            for mapping in (
                direct_mappings
                +
                uniprot_mappings
            ):

                kegg_gene_id = (
                    mapping
                    .get(
                        "KEGG_Gene_ID"
                    )
                )

                if not kegg_gene_id:
                    continue

                if kegg_gene_id in seen_kegg_ids:
                    continue

                seen_kegg_ids.add(
                    kegg_gene_id
                )

                combined_mappings.append(
                    kegg_gene_id
                )

            # ------------------------------------------------
            # No KEGG gene.
            # ------------------------------------------------

            if not combined_mappings:

                kegg_results.append({

                    "Gene_ID":
                        gene_id,

                    "UniProt_Accession":
                        uniprot_accession,

                    "KEGG_Gene_ID":
                        "Not found",

                    "KEGG_Pathway_ID":
                        "Not found",

                    "KEGG_Pathway_Name":
                        "No KEGG pathway match",

                    "KEGG_Status":
                        "No KEGG gene match"
                })

            else:

                # ------------------------------------------------
                # Get pathways for each KEGG gene.
                # ------------------------------------------------

                gene_had_pathway = False

                for kegg_gene_id in (
                    combined_mappings
                ):

                    pathways = (
                        get_kegg_pathways(
                            kegg_gene_id
                        )
                    )

                    if not pathways:

                        kegg_results.append({

                            "Gene_ID":
                                gene_id,

                            "UniProt_Accession":
                                uniprot_accession,

                            "KEGG_Gene_ID":
                                kegg_gene_id,

                            "KEGG_Pathway_ID":
                                "Not found",

                            "KEGG_Pathway_Name":
                                "No pathway assigned",

                            "KEGG_Status":
                                "KEGG gene found; "
                                "no pathway"
                        })

                        continue

                    gene_had_pathway = True

                    for pathway in pathways:

                        pathway_id = (
                            pathway[
                                "KEGG_Pathway_ID"
                            ]
                        )

                        pathway_name = (
                            pathway_names
                            .get(
                                pathway_id,
                                "Name not available"
                            )
                        )

                        kegg_results.append({

                            "Gene_ID":
                                gene_id,

                            "UniProt_Accession":
                                uniprot_accession,

                            "KEGG_Gene_ID":
                                kegg_gene_id,

                            "KEGG_Pathway_ID":
                                pathway_id,

                            "KEGG_Pathway_Name":
                                pathway_name,

                            "KEGG_Status":
                                "Matched"
                        })

                if not gene_had_pathway:

                    # The individual no-pathway rows already
                    # contain the useful information.

                    pass

            progress.progress(
                (i + 1) / total
            )

            # KEGG asks users to limit request frequency.
            time.sleep(0.35)

        kegg_df = pd.DataFrame(
            kegg_results
        )

        # Attach expression information.
        expression_columns = [
            "Gene_ID",
            "Log2_Fold_Change",
            "Regulation"
        ]

        kegg_df = kegg_df.merge(
            significant_genes[
                expression_columns
            ],
            on="Gene_ID",
            how="left"
        )

        st.session_state[
            "kegg_annotation"
        ] = kegg_df

        status_text.success(
            f"KEGG mapping completed for "
            f"{total} gene(s)."
        )


# ============================================================
# DISPLAY KEGG
# ============================================================

if "kegg_annotation" in st.session_state:

    st.subheader(
        "KEGG Gene & Pathway Results"
    )

    kegg_display = st.session_state[
        "kegg_annotation"
    ]

    st.dataframe(
        kegg_display,
        use_container_width=True
    )

    matched_genes = (
        kegg_display[
            kegg_display[
                "KEGG_Status"
            ] == "Matched"
        ]["Gene_ID"]
        .nunique()
    )

    total_genes = (
        kegg_display[
            "Gene_ID"
        ].nunique()
    )

    st.write(
        f"**KEGG summary:** "
        f"{matched_genes} of "
        f"{total_genes} queried genes "
        f"have at least one KEGG pathway match."
    )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    csv_data = (
        kegg_display
        .to_csv(
            index=False
        )
        .encode("utf-8")
    )

    st.download_button(
        label="⬇️ Download KEGG Results CSV",
        data=csv_data,
        file_name=(
            "komagataella_kegg_results.csv"
        ),
        mime="text/csv"
    )


# ============================================================
# KEGG PATHWAY SUMMARY
# ============================================================

if "kegg_annotation" in st.session_state:

    kegg_display = st.session_state[
        "kegg_annotation"
    ]

    pathway_rows = (
        kegg_display[
            kegg_display[
                "KEGG_Status"
            ] == "Matched"
        ]
    )

    if len(pathway_rows) > 0:

        st.subheader(
            "🧬 Pathway ↔ Gene Summary"
        )

        pathway_summary = (
            pathway_rows
            .groupby(
                [
                    "KEGG_Pathway_ID",
                    "KEGG_Pathway_Name"
                ],
                dropna=False
            )
            .agg(
                Gene_Count=(
                    "Gene_ID",
                    "nunique"
                ),
                Genes=(
                    "Gene_ID",
                    lambda values:
                    ", ".join(
                        sorted(
                            set(values)
                        )
                    )
                )
            )
            .reset_index()
            .sort_values(
                "Gene_Count",
                ascending=False
            )
        )

        st.dataframe(
            pathway_summary,
            use_container_width=True
        )


# ============================================================
# KEGG PATHWAY VISUALIZATION
# ============================================================

st.header(
    "6️⃣ Visual KEGG Pathway Analysis"
)

if "kegg_annotation" not in st.session_state:

    st.info(
        "Run the KEGG mapping first."
    )

else:

    kegg_display = st.session_state[
        "kegg_annotation"
    ]

    matched = kegg_display[
        kegg_display[
            "KEGG_Status"
        ] == "Matched"
    ].copy()

    if len(matched) == 0:

        st.info(
            "No matched KEGG pathways are available "
            "for visualization."
        )

    else:

        pathway_options = (
            matched[
                [
                    "KEGG_Pathway_ID",
                    "KEGG_Pathway_Name"
                ]
            ]
            .drop_duplicates()
            .sort_values(
                "KEGG_Pathway_ID"
            )
        )

        pathway_labels = {}

        for _, row in (
            pathway_options.iterrows()
        ):

            pathway_labels[
                row["KEGG_Pathway_ID"]
            ] = (
                f'{row["KEGG_Pathway_ID"]} — '
                f'{row["KEGG_Pathway_Name"]}'
            )

        selected_pathway = st.selectbox(
            "Select a KEGG pathway",
            options=list(
                pathway_labels.keys()
            ),
            format_func=lambda x:
                pathway_labels[x]
        )

        pathway_genes = matched[
            matched[
                "KEGG_Pathway_ID"
            ] == selected_pathway
        ].copy()

        st.write(
            f"**Selected pathway:** "
            f"{pathway_labels[selected_pathway]}"
        )

        st.dataframe(
            pathway_genes[
                [
                    "Gene_ID",
                    "Log2_Fold_Change",
                    "Regulation",
                    "KEGG_Gene_ID"
                ]
            ],
            use_container_width=True
        )

        # ----------------------------------------------------
        # KEGG Mapper-style color dataset.
        #
        # Format:
        #   ppa:GENE_ID bgcolor,fgcolor
        #
        # Red = upregulated
        # Blue = downregulated
        # ----------------------------------------------------

        color_lines = []

        for _, row in (
            pathway_genes.iterrows()
        ):

            kegg_gene_id = row[
                "KEGG_Gene_ID"
            ]

            regulation = row[
                "Regulation"
            ]

            if not isinstance(
                kegg_gene_id,
                str
            ):
                continue

            if regulation == "Upregulated":

                color_lines.append(
                    f"{kegg_gene_id} "
                    "#ffcccc,#cc0000"
                )

            elif regulation == "Downregulated":

                color_lines.append(
                    f"{kegg_gene_id} "
                    "#ccccff,#0000cc"
                )

        color_dataset = "\n".join(
            color_lines
        )

        encoded_dataset = (
            urllib.parse.quote(
                color_dataset,
                safe=""
            )
        )

        # ----------------------------------------------------
        # Use the selected organism-specific pathway map.
        # ----------------------------------------------------

        pathway_url = (
            "https://www.kegg.jp/"
            "kegg-bin/show_pathway"
            f"?map={selected_pathway}"
        )

        if encoded_dataset:

            pathway_url += (
                f"&multi_query="
                f"{encoded_dataset}"
            )

        st.markdown(
            "### 🔗 Open colored pathway in KEGG"
        )

        st.markdown(
            f"[Open **{pathway_labels[selected_pathway]}** "
            f"in KEGG]({pathway_url})"
        )

        st.caption(
            "KEGG is opened in a separate page so the official "
            "interactive pathway map remains available."
        )

        st.code(
            color_dataset,
            language="text"
        )

        st.caption(
            "Color convention: red = upregulated, "
            "blue = downregulated."
        )

# ============================================================
# END
# ============================================================
