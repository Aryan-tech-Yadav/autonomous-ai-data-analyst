from pathlib import Path

import pandas as pd
import streamlit as st

from src.agent.pipeline import AnalysisPipeline


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Autonomous AI Data Analyst",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("📊 Autonomous AI Data Analyst")

st.write(
    "Upload a CSV or Excel business file, ask a question, "
    "and let the AI plan and execute the analysis."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    provider = st.selectbox(
        "LLM Provider",
        options=[
            "nvidia",
        ],
        index=0,
    )

    st.divider()

    st.caption(
        "Nemotron is currently configured as the "
        "primary analysis model."
    )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your business data",
    type=[
        "csv",
        "xlsx",
    ],
)


# ============================================================
# DATA LOADING
# ============================================================

if uploaded_file is not None:

    try:

        if uploaded_file.name.lower().endswith(
            ".csv"
        ):

            df = pd.read_csv(
                uploaded_file
            )

        else:

            df = pd.read_excel(
                uploaded_file
            )

    except Exception as exc:

        st.error(
            f"Could not read the uploaded file: {exc}"
        )

        st.stop()

    st.success(
        f"Loaded `{uploaded_file.name}` "
        f"— {df.shape[0]} rows × {df.shape[1]} columns"
    )

    # ========================================================
    # DATA PREVIEW
    # ========================================================

    with st.expander(
        "👀 Preview dataset",
        expanded=False,
    ):

        st.dataframe(
            df.head(20),
            use_container_width=True,
        )

    # ========================================================
    # USER QUESTION
    # ========================================================

    st.subheader(
        "💬 Ask your data"
    )

    user_query = st.text_area(
        "What would you like to know?",
        placeholder=(
            "Example: Show me a bar chart of "
            "total revenue by region."
        ),
        height=100,
    )

    analyze_button = st.button(
        "🚀 Analyze",
        type="primary",
        use_container_width=True,
    )

    # ========================================================
    # ANALYSIS
    # ========================================================

    if analyze_button:

        if not user_query.strip():

            st.warning(
                "Please enter a question first."
            )

            st.stop()

        with st.spinner(
            "🤖 AI is analyzing your data..."
        ):

            try:

                pipeline = AnalysisPipeline(
                    provider=provider
                )

                result = pipeline.run(
                    df=df,
                    user_query=user_query,
                )

            except Exception as exc:

                st.error(
                    f"Analysis failed: {exc}"
                )

                st.stop()

        # ====================================================
        # PIPELINE STATUS
        # ====================================================

        status = result.get(
            "status"
        )

        if status == "success":

            st.success(
                "Analysis completed successfully."
            )

        elif status == "partial":

            st.warning(
                "Analysis completed with some errors."
            )

        else:

            st.error(
                "Analysis failed."
            )

        # ====================================================
        # FINAL ANSWER
        # ====================================================

        final_response = result.get(
            "final_response"
        )

        if final_response:

            st.subheader(
                "🧠 AI Analysis"
            )

            st.markdown(
                final_response
            )

        # ====================================================
        # CHART DETECTION
        # ====================================================

        execution = result.get(
            "execution",
            {},
        )

        execution_results = execution.get(
            "results",
            [],
        )

        chart_paths = []

        for execution_result in execution_results:

            if execution_result.get(
                "status"
            ) != "success":

                continue

            operation = execution_result.get(
                "operation"
            )

            if operation not in {
                "generate_bar_chart",
                "generate_line_chart",
            }:

                continue

            operation_result = (
                execution_result.get(
                    "result",
                    {},
                )
            )

            output_path = (
                operation_result.get(
                    "output_path"
                )
            )

            if output_path:

                chart_paths.append(
                    output_path
                )

        # ====================================================
        # DISPLAY CHARTS
        # ====================================================

        if chart_paths:

            st.subheader(
                "📊 Visualization"
            )

            for chart_path in chart_paths:

                path = Path(
                    chart_path
                )

                if path.exists():

                    st.image(
                        str(path),
                        use_container_width=True,
                    )

                else:

                    st.warning(
                        f"Chart file was generated "
                        f"but could not be found: "
                        f"`{chart_path}`"
                    )

        # ====================================================
        # DEBUG / DETAILS
        # ====================================================

        with st.expander(
            "🔍 Analysis details",
            expanded=False,
        ):

            st.json(
                result
            )


else:

    st.info(
        "Upload a CSV or Excel file to get started."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Autonomous AI Data Analyst • "
    "Nemotron-powered analysis pipeline"
)
