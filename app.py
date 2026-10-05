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

st.title("🧬 Transcriptomics & KEGG Pathway Analyzer")

st.write(
    "A pipeline for differential expression analysis, "
    "gene annotation, and KEGG pathway mapping."
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("dummy_data.csv")


data = load_data()

st.subheader("📊 Input Transcriptomics Data")

st.dataframe(
    data,
    use_container_width=True
)


# ============================================================
# DIFFERENTIAL EXPRESSION ANALYSIS
# ============================================================

st.header("1️⃣ Differential Expression Analysis")


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


# ------------------------------------------------------------
# Calculate mean expression
# ------------------------------------------------------------

data["Control_Mean"] = data[control_columns].mean(axis=1)

data["Treatment_Mean"] = data[treatment_columns].mean(axis=1)


# ------------------------------------------------------------
# Fold change
# ------------------------------------------------------------

data["Fold_Change"] = (
    data["Treatment_Mean"] /
    data["Control_Mean"]
)


# ------------------------------------------------------------
# Log2 fold change
# ------------------------------------------------------------

data["Log2_Fold_Change"] = np.log2(
    data["Fold_Change"]
)


# ------------------------------------------------------------
# Welch's t-test
# ------------------------------------------------------------

p_values = []

for _, row in data.iterrows():

    control_values = row[control_columns].astype(float)

    treatment_values = row[treatment_columns].astype(float)

    statistic, p_value = ttest_ind(
        control_values,
        treatment_values,
        equal_var=False
    )

    p_values.append(p_value)


data["P_Value"] = p_values


# ============================================================
# BENJAMINI-HOCHBERG FDR
# ============================================================

def benjamini_hochberg(pvalues):

    pvalues = np.array(pvalues)

    n = len(pvalues)

    order = np.argsort(pvalues)

    ranked_pvalues = pvalues[order]

    adjusted = np.empty(n)

    previous = 1.0

    for i in range(n - 1, -1, -1):

        rank = i + 1

        value = (
            ranked_pvalues[i] *
            n /
            rank
        )

        value = min(value, previous)

        adjusted[i] = value

        previous = value

    result = np.empty(n)

    result[order] = adjusted

    return result


data["Adjusted_P_Value"] = benjamini_hochberg(
    data["P_Value"]
)


# ============================================================
# SIGNIFICANCE THRESHOLDS
# ============================================================

log2fc_threshold = st.slider(
    "Absolute log₂ fold-change threshold",
    min_value=0.5,
    max_value=3.0,
    value=1.0,
    step=0.1
)

adjusted_p_threshold = st.number_input(
    "Adjusted p-value threshold",
    min_value=0.001,
    max_value=0.20,
    value=0.05,
    step=0.01
)


# ============================================================
# CLASSIFY GENES
# ============================================================

def classify_gene(row):

    log2fc = row["Log2_Fold_Change"]

    adjusted_p = row["Adjusted_P_Value"]

    if (
        adjusted_p <= adjusted_p_threshold
        and
        log2fc >= log2fc_threshold
    ):
        return "Upregulated"

    elif (
        adjusted_p <= adjusted_p_threshold
        and
        log2fc <= -log2fc_threshold
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

st.subheader("Differential Expression Results")

de_columns = [
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
    data[de_columns],
    use_container_width=True
)


# ============================================================
# SUMMARY
# ============================================================

st.subheader("📈 Differential Expression Summary")

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
        "Not Significant",
        not_sig_count
    )


# ============================================================
# SIGNIFICANT GENES
# ============================================================

significant_genes = data[
    data["Regulation"] != "Not significant"
].copy()


st.subheader("🎯 Significant Genes")

if len(significant_genes) > 0:

    st.dataframe(
        significant_genes[
            [
                "Gene_ID",
                "Log2_Fold_Change",
                "Adjusted_P_Value",
                "Regulation"
            ]
        ],
        use_container_width=True
    )

else:

    st.info(
        "No genes meet the current significance thresholds."
    )


# ============================================================
# NCBI ANNOTATION
# ============================================================

st.header("2️⃣ NCBI Gene Annotation")


NCBI_BASE_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
)


def get_ncbi_annotation(gene_id):

    search_url = (
        NCBI_BASE_URL +
        "esearch.fcgi"
    )

    search_params = {

        "db": "gene",

        "term": (
            f'"{gene_id}"[Gene Name] '
            f'OR "{gene_id}"[All Fields]'
        ),

        "retmode": "json",

        "retmax": 5
    }

    response = requests.get(
        search_url,
        params=search_params,
        timeout=30
    )

    response.raise_for_status()

    search_data = response.json()

    ids = search_data.get(
        "esearchresult",
        {}
    ).get(
        "idlist",
        []
    )

    if not ids:

        return {
            "NCBI_Gene_ID": "Not found",
            "Gene_Name": "Not found",
            "Gene_Description": "Not found",
            "Organism": "Not found",
            "Chromosome": "Not found",
            "NCBI_Status": "No NCBI match"
        }

    ncbi_gene_id = ids[0]


    summary_url = (
        NCBI_BASE_URL +
        "esummary.fcgi"
    )

    summary_params = {

        "db": "gene",

        "id": ncbi_gene_id,

        "retmode": "json"
    }

    summary_response = requests.get(
        summary_url,
        params=summary_params,
        timeout=30
    )

    summary_response.raise_for_status()

    summary_data = summary_response.json()

    result = summary_data.get(
        "result",
        {}
    ).get(
        ncbi_gene_id,
        {}
    )


    return {

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
            result.get(
                "organism",
                {}).get(
                    "scientificname",
                    "Not available"
                ),

        "Chromosome":
            result.get(
                "chromosome",
                "Not available"
            ),

        "NCBI_Status":
            "Matched"
    }


if st.button(
    "🔎 Annotate Significant Genes with NCBI"
):

    if len(significant_genes) == 0:

        st.warning(
            "There are no significant genes to annotate."
        )

    else:

        ncbi_results = []

        progress = st.progress(0)

        total = len(significant_genes)

        for i, gene_id in enumerate(
            significant_genes["Gene_ID"]
        ):

            try:

                annotation = get_ncbi_annotation(
                    gene_id
                )

            except Exception as e:

                annotation = {

                    "NCBI_Gene_ID":
                        "Error",

                    "Gene_Name":
                        "Error",

                    "Gene_Description":
                        str(e),

                    "Organism":
                        "Error",

                    "Chromosome":
                        "Error",

                    "NCBI_Status":
                        "Request failed"
                }

            annotation["Gene_ID"] = gene_id

            ncbi_results.append(
                annotation
            )

            progress.progress(
                (i + 1) / total
            )

            time.sleep(0.2)


        ncbi_df = pd.DataFrame(
            ncbi_results
        )


        st.session_state[
            "ncbi_annotation"
        ] = ncbi_df


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

st.header("3️⃣ UniProt Protein Annotation")


UNIPROT_URL = (
    "https://rest.uniprot.org/uniprotkb/search"
)


def get_uniprot_annotation(gene_id):

    params = {

        "query":
            f"gene_exact:{gene_id} "
            f"AND organism_id:3702",

        "format":
            "json",

        "fields":
            (
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
        UNIPROT_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data_json = response.json()

    results = data_json.get(
        "results",
        []
    )


    if not results:

        return {

            "UniProt_Accession":
                "Not found",

            "UniProt_ID":
                "Not found",

            "UniProt_Protein":
                "Not found",

            "UniProt_Organism":
                "Not found",

            "Protein_Length":
                "Not found",

            "UniProt_Reviewed":
                "Not found",

            "UniProt_Function":
                "Not available",

            "UniProt_Status":
                "No UniProt match"
        }


    # Prefer reviewed Swiss-Prot entry
    reviewed_results = [
        r for r in results
        if r.get("entryType") == "UniProtKB reviewed (Swiss-Prot)"
    ]


    if reviewed_results:

        result = reviewed_results[0]

    else:

        result = results[0]


    accession = result.get(
        "primaryAccession",
        "Not available"
    )


    uniprot_id = result.get(
        "uniProtkbId",
        "Not available"
    )


    protein_description = (
        result.get(
            "proteinDescription",
            {}
        )
    )


    recommended_name = (
        protein_description
        .get(
            "recommendedName",
            {}
        )
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
        == "UniProtKB reviewed (Swiss-Prot)"
    )


    comments = result.get(
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
            uniprot_id,

        "UniProt_Protein":
            protein_name,

        "UniProt_Organism":
            organism,

        "Protein_Length":
            protein_length,

        "UniProt_Reviewed":
            "Yes" if reviewed else "No",

        "UniProt_Function":
            function_text,

        "UniProt_Status":
            "Matched"
    }


if st.button(
    "🧬 Annotate Significant Genes with UniProt"
):

    if len(significant_genes) == 0:

        st.warning(
            "There are no significant genes to annotate."
        )

    else:

        uniprot_results = []

        progress = st.progress(0)

        total = len(significant_genes)


        for i, gene_id in enumerate(
            significant_genes["Gene_ID"]
        ):

            try:

                annotation = get_uniprot_annotation(
                    gene_id
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

                    "UniProt_Status":
                        "Request failed"
                }


            annotation["Gene_ID"] = gene_id

            uniprot_results.append(
                annotation
            )


            progress.progress(
                (i + 1) / total
            )

            time.sleep(0.2)


        uniprot_df = pd.DataFrame(
            uniprot_results
        )


        st.session_state[
            "uniprot_annotation"
        ] = uniprot_df


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

st.header("4️⃣ InterPro Protein Domain Annotation")


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


    if response.status_code == 204:

        return []


    if response.status_code == 404:

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


        if next_response.status_code != 200:

            break


        next_data = next_response.json()


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


    if isinstance(name, dict):

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

        member_databases = metadata.get(
            "member_databases"
        )


        if (
            isinstance(
                member_databases,
                dict
            )
            and member_databases
        ):

            source_text = ", ".join(
                member_databases.keys()
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

        uniprot_df = st.session_state[
            "uniprot_annotation"
        ]


        interpro_results = []

        progress = st.progress(0)


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


        total = len(valid_rows)


        if total == 0:

            st.warning(
                "No valid UniProt accessions were found."
            )

        else:

            for i, (_, row) in enumerate(
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


                progress.progress(
                    (i + 1) / total
                )


                time.sleep(0.2)


            interpro_df = pd.DataFrame(
                interpro_results
            )


            st.session_state[
                "interpro_annotation"
            ] = interpro_df


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
# KEGG MAPPING
# ============================================================

st.header("5️⃣ KEGG Pathway Mapping")


KEGG_BASE_URL = (
    "https://rest.kegg.jp"
)


KEGG_ORGANISM = "ath"


# ------------------------------------------------------------
# UniProt → KEGG gene
# ------------------------------------------------------------

def kegg_uniprot_to_gene(
    uniprot_accession
):

    url = (
        f"{KEGG_BASE_URL}/conv/"
        f"{KEGG_ORGANISM}/"
        f"uniprot:{uniprot_accession}"
    )


    response = requests.get(
        url,
        timeout=30
    )


    if response.status_code == 404:

        return []


    response.raise_for_status()


    text = response.text.strip()


    if not text:

        return []


    mappings = []


    for line in text.splitlines():

        parts = line.split(
            "\t"
        )


        if len(parts) != 2:

            continue


        source_id = parts[0]

        kegg_id = parts[1]


        mappings.append({

            "UniProt_ID":
                source_id,

            "KEGG_Gene_ID":
                kegg_id
        })


    return mappings


# ------------------------------------------------------------
# KEGG gene → pathways
# ------------------------------------------------------------

def get_kegg_pathways(
    kegg_gene_id
):

    url = (
        f"{KEGG_BASE_URL}/link/"
        f"pathway/"
        f"{kegg_gene_id}"
    )


    response = requests.get(
        url,
        timeout=30
    )


    if response.status_code == 404:

        return []


    response.raise_for_status()


    text = response.text.strip()


    if not text:

        return []


    pathways = []


    for line in text.splitlines():

        parts = line.split(
            "\t"
        )


        if len(parts) != 2:

            continue


        gene_id = parts[0]

        pathway_id = parts[1]


        pathways.append({

            "KEGG_Gene_ID":
                gene_id,

            "KEGG_Pathway_ID":
                pathway_id
        })


    return pathways


# ------------------------------------------------------------
# Pathway IDs → Pathway names
# ------------------------------------------------------------

def get_kegg_pathway_names(
    pathway_ids
):

    if not pathway_ids:

        return {}


    unique_ids = list(
        dict.fromkeys(
            pathway_ids
        )
    )


    pathway_names = {}


    for i in range(
        0,
        len(unique_ids),
        10
    ):

        batch = unique_ids[
            i:i + 10
        ]


        url = (
            f"{KEGG_BASE_URL}/list/"
            +
            "+".join(batch)
        )


        response = requests.get(
            url,
            timeout=30
        )


        if response.status_code == 404:

            continue


        response.raise_for_status()


        text = response.text.strip()


        if not text:

            continue


        for line in text.splitlines():

            parts = line.split(
                "\t"
            )


            if len(parts) >= 2:

                pathway_id = parts[0]

                pathway_name = parts[1]


                pathway_names[
                    pathway_id
                ] = pathway_name


        time.sleep(0.4)


    return pathway_names


# ============================================================
# COMPLETE KEGG ANNOTATION FUNCTION
# ============================================================

def annotate_uniprot_with_kegg(
    uniprot_df
):

    kegg_results = []


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


    for _, row in valid_rows.iterrows():

        gene_id = row[
            "Gene_ID"
        ]


        accession = row[
            "UniProt_Accession"
        ]


        try:

            # ------------------------------------------------
            # STEP 1
            # UniProt → KEGG gene
            # ------------------------------------------------

            kegg_gene_mappings = (
                kegg_uniprot_to_gene(
                    accession
                )
            )


            if not kegg_gene_mappings:

                kegg_results.append({

                    "Gene_ID":
                        gene_id,

                    "UniProt_Accession":
                        accession,

                    "KEGG_Gene_ID":
                        "Not found",

                    "KEGG_Pathway_ID":
                        "Not found",

                    "KEGG_Pathway_Name":
                        "No KEGG pathway match",

                    "KEGG_Status":
                        "No KEGG mapping"
                })

                continue


            # ------------------------------------------------
            # STEP 2
            # Get pathways
            # ------------------------------------------------

            all_pathways = []


            for mapping in kegg_gene_mappings:

                kegg_gene_id = mapping[
                    "KEGG_Gene_ID"
                ]


                pathway_mappings = (
                    get_kegg_pathways(
                        kegg_gene_id
                    )
                )


                for pathway in pathway_mappings:

                    all_pathways.append({

                        "KEGG_Gene_ID":
                            kegg_gene_id,

                        "KEGG_Pathway_ID":
                            pathway[
                                "KEGG_Pathway_ID"
                            ]
                    })


                time.sleep(0.2)


            if not all_pathways:

                for mapping in kegg_gene_mappings:

                    kegg_results.append({

                        "Gene_ID":
                            gene_id,

                        "UniProt_Accession":
                            accession,

                        "KEGG_Gene_ID":
                            mapping[
                                "KEGG_Gene_ID"
                            ],

                        "KEGG_Pathway_ID":
                            "Not found",

                        "KEGG_Pathway_Name":
                            "No KEGG pathway match",

                        "KEGG_Status":
                            "KEGG gene found"
                    })


                continue


            # ------------------------------------------------
            # STEP 3
            # Remove duplicate pathways
            # ------------------------------------------------

            unique_pathways = []


            seen = set()


            for item in all_pathways:

                key = (
                    item["KEGG_Gene_ID"],
                    item["KEGG_Pathway_ID"]
                )


                if key not in seen:

                    seen.add(key)

                    unique_pathways.append(
                        item
                    )


            # ------------------------------------------------
            # STEP 4
            # Get pathway names
            # ------------------------------------------------

            pathway_ids = [

                item[
                    "KEGG_Pathway_ID"
                ]

                for item in unique_pathways
            ]


            pathway_names = (
                get_kegg_pathway_names(
                    pathway_ids
                )
            )


            # ------------------------------------------------
            # STEP 5
            # Create final KEGG rows
            # ------------------------------------------------

            for item in unique_pathways:

                pathway_id = item[
                    "KEGG_Pathway_ID"
                ]


                pathway_name = pathway_names.get(
                    pathway_id,
                    "Name not available"
                )


                kegg_results.append({

                    "Gene_ID":
                        gene_id,

                    "UniProt_Accession":
                        accession,

                    "KEGG_Gene_ID":
                        item[
                            "KEGG_Gene_ID"
                        ],

                    "KEGG_Pathway_ID":
                        pathway_id,

                    "KEGG_Pathway_Name":
                        pathway_name,

                    "KEGG_Status":
                        "Matched"
                })


        except Exception as e:

            kegg_results.append({

                "Gene_ID":
                    gene_id,

                "UniProt_Accession":
                    accession,

                "KEGG_Gene_ID":
                    "Error",

                "KEGG_Pathway_ID":
                    "Error",

                "KEGG_Pathway_Name":
                    str(e),

                "KEGG_Status":
                    "Request failed"
            })


    return pd.DataFrame(
        kegg_results
    )


# ============================================================
# KEGG BUTTON
# ============================================================

if st.button(
    "🧬 Map UniProt Proteins to KEGG Pathways"
):

    if (
        "uniprot_annotation"
        not in st.session_state
    ):

        st.warning(
            "Run UniProt annotation first."
        )

    else:

        uniprot_df = st.session_state[
            "uniprot_annotation"
        ]


        if len(uniprot_df) == 0:

            st.warning(
                "No UniProt annotation results available."
            )

        else:

            progress_placeholder = st.empty()


            progress_placeholder.info(
                "🔄 Mapping UniProt proteins to KEGG..."
            )


            try:

                kegg_df = (
                    annotate_uniprot_with_kegg(
                        uniprot_df
                    )
                )


                st.session_state[
                    "kegg_annotation"
                ] = kegg_df


                progress_placeholder.success(
                    "✅ KEGG mapping completed."
                )


            except Exception as e:

                progress_placeholder.error(
                    f"KEGG mapping failed: {e}"
                )


# ============================================================
# DISPLAY KEGG RESULTS
# ============================================================

if "kegg_annotation" in st.session_state:

    st.subheader(
        "🛣️ KEGG Pathway Results"
    )


    kegg_df = st.session_state[
        "kegg_annotation"
    ]


    st.dataframe(
        kegg_df,
        use_container_width=True
    )


# ============================================================
# FINAL INTEGRATED TABLE
# ============================================================

st.header(
    "6️⃣ Integrated Transcriptomics Annotation"
)


if (
    "ncbi_annotation" in st.session_state
    and
    "uniprot_annotation" in st.session_state
    and
    "interpro_annotation" in st.session_state
    and
    "kegg_annotation" in st.session_state
):


    # --------------------------------------------------------
    # Start with differential expression results
    # --------------------------------------------------------

    final_df = significant_genes.copy()


    # --------------------------------------------------------
    # NCBI
    # --------------------------------------------------------

    ncbi_df = st.session_state[
        "ncbi_annotation"
    ].copy()


    final_df = final_df.merge(
        ncbi_df,
        on="Gene_ID",
        how="left"
    )


    # --------------------------------------------------------
    # UniProt
    # --------------------------------------------------------

    uniprot_df = st.session_state[
        "uniprot_annotation"
    ].copy()


    final_df = final_df.merge(
        uniprot_df,
        on="Gene_ID",
        how="left"
    )


    # --------------------------------------------------------
    # InterPro
    # --------------------------------------------------------

    interpro_df = st.session_state[
        "interpro_annotation"
    ].copy()


    interpro_summary = (
        interpro_df
        .groupby("Gene_ID")
        .agg({

            "InterPro_Accession":
                lambda x:
                "; ".join(
                    x.astype(str)
                    .unique()
                ),

            "InterPro_Name":
                lambda x:
                "; ".join(
                    x.astype(str)
                    .unique()
                ),

            "InterPro_Type":
                lambda x:
                "; ".join(
                    x.astype(str)
                    .unique()
                ),

            "InterPro_Source":
                lambda x:
                "; ".join(
                    x.astype(str)
                    .unique()
                ),

            "InterPro_Regions":
                lambda x:
                "; ".join(
                    x.astype(str)
                    .unique()
                )
        })
        .reset_index()
    )


    final_df = final_df.merge(
        interpro_summary,
        on="Gene_ID",
        how="left"
    )


    # --------------------------------------------------------
    # KEGG
    # --------------------------------------------------------

    kegg_df = st.session_state[
        "kegg_annotation"
    ].copy()


    kegg_summary = (
        kegg_df
        .groupby("Gene_ID")
        .agg({

            "KEGG_Gene_ID":
                lambda x:
                "; ".join(
                    x.astype(str)
                    .unique()
                ),

            "KEGG_Pathway_ID":
                lambda x:
                "; ".join(
                    x.astype(str)
                    .unique()
                ),

            "KEGG_Pathway_Name":
                lambda x:
                "; ".join(
                    x.astype(str)
                    .unique()
                ),

            "KEGG_Status":
                lambda x:
                "; ".join(
                    x.astype(str)
                    .unique()
                )
        })
        .reset_index()
    )


    final_df = final_df.merge(
        kegg_summary,
        on="Gene_ID",
        how="left"
    )


    # --------------------------------------------------------
    # Display final table
    # --------------------------------------------------------

    st.success(
        "🎉 Complete annotation pipeline available!"
    )


    st.dataframe(
        final_df,
        use_container_width=True
    )


    # --------------------------------------------------------
    # Download final table
    # --------------------------------------------------------

    csv_data = final_df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="⬇️ Download Integrated Results CSV",
        data=csv_data,
        file_name="transcriptomics_integrated_results.csv",
        mime="text/csv"
    )


else:

    st.info(
        "Run NCBI, UniProt, InterPro, and KEGG annotation "
        "to generate the integrated results table."
    )


# ============================================================
# PIPELINE STATUS
# ============================================================

st.header("🧬 Pipeline Status")


status_data = {

    "Step": [

        "Transcriptomics Data",

        "Differential Expression",

        "NCBI Annotation",

        "UniProt Annotation",

        "InterPro Annotation",

        "KEGG Mapping",

        "Integrated Results"
    ],

    "Status": [

        "✅ Complete",

        "✅ Complete",

        (
            "✅ Complete"
            if "ncbi_annotation"
            in st.session_state
            else
            "⏳ Pending"
        ),

        (
            "✅ Complete"
            if "uniprot_annotation"
            in st.session_state
            else
            "⏳ Pending"
        ),

        (
            "✅ Complete"
            if "interpro_annotation"
            in st.session_state
            else
            "⏳ Pending"
        ),

        (
            "✅ Complete"
            if "kegg_annotation"
            in st.session_state
            else
            "⏳ Pending"
        ),

        (
            "✅ Complete"
            if (
                "ncbi_annotation"
                in st.session_state
                and
                "uniprot_annotation"
                in st.session_state
                and
                "interpro_annotation"
                in st.session_state
                and
                "kegg_annotation"
                in st.session_state
            )
            else
            "⏳ Pending"
        )
    ]
}


status_df = pd.DataFrame(
    status_data
)


st.dataframe(
    status_df,
    use_container_width=True,
    hide_index=True
)
