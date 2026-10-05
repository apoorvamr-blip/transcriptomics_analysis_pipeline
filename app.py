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
