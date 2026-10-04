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

st.markdown(
    """
Upload a CSV or Excel business file, ask a question,
and let the autonomous AI analyze your data.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    provider = st.selectbox(
        "LLM Provider",
        options=["nvidia"],
        index=0,
    )

    st.divider()

    st.markdown("### 🤖 AI Engine")

    st.info(
        "Nemotron is currently configured as the "
        "primary analysis model."
    )

    st.divider()

    st.caption(
        "Autonomous AI Data Analyst"
    )

    st.caption(
        "Upload → Plan → Execute → Analyze → Explain"
    )


# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader("📁 Upload Business Data")

uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=["csv", "xlsx"],
)


# ============================================================
# EMPTY STATE
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload a CSV or Excel file to start your analysis."
    )

    st.markdown(
        """
### What this system can do

- 📊 Analyze business datasets
- 🔢 Calculate metrics
- 🧮 Perform aggregations
- 🏆 Find highest/lowest values
- 📈 Generate charts
- 🤖 Automatically plan analysis steps
- 🧠 Generate a business-friendly explanation
"""
    )

    st.stop()


# ============================================================
# DATA LOADING
# ============================================================

try:

    if uploaded_file.name.lower().endswith(".csv"):

        df = pd.read_csv(uploaded_file)

    else:

        df = pd.read_excel(uploaded_file)

except Exception as exc:

    st.error(
        f"Could not read the uploaded file: {exc}"
    )

    st.stop()


# ============================================================
# DATASET HEADER
# ============================================================

st.success(
    f"Loaded `{uploaded_file.name}` successfully."
)


# ============================================================
# DATASET METRICS
# ============================================================

row_count = df.shape[0]
column_count = df.shape[1]
missing_count = int(df.isna().sum().sum())
duplicate_count = int(df.duplicated().sum())

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

with metric_1:

    st.metric(
        "Rows",
        f"{row_count:,}",
    )

with metric_2:

    st.metric(
        "Columns",
        f"{column_count:,}",
    )

with metric_3:

    st.metric(
        "Missing Values",
        f"{missing_count:,}",
    )

with metric_4:

    st.metric(
        "Duplicate Rows",
        f"{duplicate_count:,}",
    )


# ============================================================
# DATA PREVIEW
# ============================================================

with st.expander(
    "👀 Preview Dataset",
    expanded=False,
):

    st.dataframe(
        df.head(20),
        use_container_width=True,
    )


# ============================================================
# COLUMN INFORMATION
# ============================================================

with st.expander(
    "🧱 Dataset Columns",
    expanded=False,
):

    column_info = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [
                str(dtype)
                for dtype in df.dtypes
            ],
            "Missing": [
                int(df[column].isna().sum())
                for column in df.columns
            ],
            "Unique": [
                int(df[column].nunique(dropna=True))
                for column in df.columns
            ],
        }
    )

    st.dataframe(
        column_info,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# USER QUESTION
# ============================================================

st.subheader("💬 Ask Your Data")

user_query = st.text_area(
    "What would you like to know?",
    placeholder=(
        "Example: Calculate total revenue by region, "
        "find the highest-revenue region, and create a bar chart."
    ),
    height=120,
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze_button = st.button(
    "🚀 Analyze Data",
    type="primary",
    use_container_width=True,
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze_button:

    if not user_query.strip():

        st.warning(
            "Please enter a question first."
        )

        st.stop()

    progress_container = st.empty()

    progress_container.info(
        "🤖 AI is planning and executing your analysis..."
    )

    try:

        pipeline = AnalysisPipeline(
            provider=provider
        )

        result = pipeline.run(
            df=df,
            user_query=user_query,
        )

    except Exception as exc:

        progress_container.empty()

        st.error(
            f"Analysis failed: {exc}"
        )

        st.stop()

    progress_container.empty()


    # ========================================================
    # PIPELINE STATUS
    # ========================================================

    status = result.get(
        "status"
    )

    if status == "success":

        st.success(
            "✅ Analysis completed successfully."
        )

    elif status == "partial":

        st.warning(
            "⚠️ Analysis completed with some errors."
        )

    else:

        st.error(
            "❌ Analysis failed."
        )


    # ========================================================
    # FINAL AI RESPONSE
    # ========================================================

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

    else:

        st.warning(
            "No final AI response was generated."
        )


    # ========================================================
    # EXECUTION RESULTS
    # ========================================================

    execution = result.get(
        "execution",
        {},
    )

    execution_results = execution.get(
        "results",
        [],
    )


    # ========================================================
    # EXECUTION SUMMARY
    # ========================================================

    successful_steps = sum(
        1
        for item in execution_results
        if item.get("status") == "success"
    )

    failed_steps = sum(
        1
        for item in execution_results
        if item.get("status") != "success"
    )

    if execution_results:

        st.subheader(
            "⚙️ Execution Summary"
        )

        summary_1, summary_2, summary_3 = st.columns(3)

        with summary_1:

            st.metric(
                "Operations",
                len(execution_results),
            )

        with summary_2:

            st.metric(
                "Successful",
                successful_steps,
            )

        with summary_3:

            st.metric(
                "Failed",
                failed_steps,
            )


    # ========================================================
    # CHART DETECTION
    # ========================================================

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


    # ========================================================
    # DISPLAY CHARTS
    # ========================================================

    if chart_paths:

        st.subheader(
            "📊 Visualization"
        )

        for index, chart_path in enumerate(
            chart_paths,
            start=1,
        ):

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
                    f"Chart {index} was generated but "
                    f"the file could not be found: "
                    f"`{chart_path}`"
                )


    # ========================================================
    # EXECUTION OPERATIONS
    # ========================================================

    if execution_results:

        with st.expander(
            "🔧 Executed Operations",
            expanded=False,
        ):

            for index, item in enumerate(
                execution_results,
                start=1,
            ):

                operation = item.get(
                    "operation",
                    "unknown",
                )

                operation_status = item.get(
                    "status",
                    "unknown",
                )

                if operation_status == "success":

                    st.success(
                        f"{index}. `{operation}` → success"
                    )

                else:

                    st.error(
                        f"{index}. `{operation}` → "
                        f"{operation_status}"
                    )


    # ========================================================
    # ANALYSIS DETAILS
    # ========================================================

    with st.expander(
        "🔍 Analysis Details",
        expanded=False,
    ):

        st.json(
            result
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Autonomous AI Data Analyst • "
    "Nemotron-powered autonomous analysis"
)
