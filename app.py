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
# SESSION STATE
# ============================================================

if "dataset" not in st.session_state:
    st.session_state.dataset = None

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None

if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "analysis_count" not in st.session_state:
    st.session_state.analysis_count = 0


# ============================================================
# DATA LOADING
# ============================================================

def load_uploaded_file(uploaded_file):
    """Load CSV or Excel into a DataFrame."""

    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)

    return pd.read_excel(uploaded_file)


# ============================================================
# RESULT HELPERS
# ============================================================

def extract_chart_paths(result):
    """Extract generated chart paths from execution results."""

    execution = result.get("execution") or {}
    execution_results = execution.get("results", [])

    chart_paths = []

    for item in execution_results:

        if item.get("status") != "success":
            continue

        operation = item.get("operation")

        if operation not in {
            "generate_bar_chart",
            "generate_line_chart",
        }:
            continue

        operation_result = item.get("result") or {}

        output_path = operation_result.get("output_path")

        if output_path:
            chart_paths.append(output_path)

    return chart_paths


def extract_insights(result):
    """
    Extract verified business insights from the pipeline.

    Pipeline structure:
        result["insights"]["insights"]
    """

    insights_wrapper = result.get("insights") or {}

    if isinstance(insights_wrapper, dict):
        insights = insights_wrapper.get("insights", [])

        if isinstance(insights, list):
            return insights

    return []


# ============================================================
# DATASET OVERVIEW
# ============================================================

def render_dataset_overview(df):
    """Render verified dataset quality and size metrics."""

    if df is None or df.empty:
        return

    rows, columns = df.shape

    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    total_cells = rows * columns

    if total_cells > 0:
        missing_percentage = (
            missing_cells / total_cells
        ) * 100
    else:
        missing_percentage = 0.0

    st.markdown("### 📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Rows",
            f"{rows:,}",
        )

    with col2:
        st.metric(
            "Columns",
            f"{columns:,}",
        )

    with col3:
        st.metric(
            "Missing Cells",
            f"{missing_cells:,}",
            f"{missing_percentage:.1f}%",
            delta_color="inverse",
        )

    with col4:
        st.metric(
            "Duplicate Rows",
            f"{duplicate_rows:,}",
        )

# ============================================================
# STRUCTURED ANALYSIS RESULTS
# ============================================================

def render_structured_results(result):
    """Render verified tabular results from executed operations."""

    execution = result.get("execution") or {}
    execution_results = execution.get("results", [])

    tables = []

    for item in execution_results:

        if item.get("status") != "success":
            continue

        operation = item.get("operation", "")
        operation_result = item.get("result") or {}

        if not isinstance(operation_result, dict):
            continue

        # --------------------------------------------------------
        # GROUPBY / AGGREGATION
        # --------------------------------------------------------

        if operation in {
            "groupby_aggregation",
            "groupby_aggregate",
            "group_by",
        }:

            results = operation_result.get("results")

            if isinstance(results, list) and results:
                rows = []

                for row in results:
                    if not isinstance(row, dict):
                        continue

                    rows.append(row)

                if rows:
                    tables.append(
                        (
                            "Regional / Group Analysis",
                            rows,
                        )
                    )

        # --------------------------------------------------------
        # RANKING
        # --------------------------------------------------------

        elif operation == "rank_by_value":

            results = operation_result.get("results")

            if isinstance(results, list) and results:
                rows = []

                for row in results:
                    if not isinstance(row, dict):
                        continue

                    rows.append(row)

                if rows:
                    tables.append(
                        (
                            "Ranking Analysis",
                            rows,
                        )
                    )

        # --------------------------------------------------------
        # CATEGORICAL ANALYSIS
        # --------------------------------------------------------

        elif operation == "categorical_analysis":

            counts = operation_result.get("counts")

            if isinstance(counts, dict) and counts:

                rows = [
                    {
                        "category": category,
                        "count": count,
                    }
                    for category, count in counts.items()
                ]

                tables.append(
                    (
                        "Category Distribution",
                        rows,
                    )
                )

        # --------------------------------------------------------
        # STATISTICS
        # --------------------------------------------------------

        elif operation == "statistics":

            statistics = operation_result.get("statistics")

            if isinstance(statistics, dict) and statistics:

                rows = [
                    {
                        "metric": key,
                        "value": value,
                    }
                    for key, value in statistics.items()
                ]

                tables.append(
                    (
                        "Statistical Summary",
                        rows,
                    )
                )

    if not tables:
        return

    st.markdown("### 📋 Analysis Results")

    for title, rows in tables:

        st.markdown(f"#### {title}")

        display_df = pd.DataFrame(rows)

        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
        )

# ============================================================
# BUSINESS INSIGHTS UI
# ============================================================

def render_business_insights(result):
    """Render deterministic business insights from the pipeline."""

    insights_data = result.get("insights") or {}

    if not isinstance(insights_data, dict):
        return

    insights = insights_data.get("insights", [])

    if not isinstance(insights, list) or not insights:
        return

    st.markdown("### 💡 Business Insights")

    for insight in insights:

        if not isinstance(insight, dict):
            continue

        message = insight.get("message")

        if not message:
            continue

        insight_type = str(
            insight.get("type", "")
        ).lower()

        if "highest" in insight_type or "maximum" in insight_type:
            st.success(f"📈 {message}")

        elif "lowest" in insight_type:
            st.warning(f"📉 {message}")

        elif "decrease" in message.lower():
            st.warning(f"📊 {message}")

        elif "increase" in message.lower():
            st.success(f"📊 {message}")

        else:
            st.info(f"🔎 {message}")


# ============================================================
# STRUCTURED ANALYSIS RESULTS
# ============================================================

def render_structured_results(result):
    """Render verified tabular results from executed operations."""

    execution = result.get("execution") or {}
    execution_results = execution.get("results", [])

    group_rows = []
    ranking_rows = []
    other_tables = []

    for item in execution_results:

        if item.get("status") != "success":
            continue

        operation = item.get("operation", "")
        operation_result = item.get("result") or {}

        if not isinstance(operation_result, dict):
            continue

        # --------------------------------------------------------
        # GROUPBY / AGGREGATION
        # --------------------------------------------------------

        if operation in {
            "groupby_aggregation",
            "groupby_aggregate",
            "group_by",
        }:

            results = operation_result.get("results")

            if isinstance(results, list) and results:

                for row in results:

                    if isinstance(row, dict):
                        group_rows.append(row)

        # --------------------------------------------------------
        # RANKING
        # --------------------------------------------------------

        elif operation == "rank_by_value":

            results = operation_result.get("results")

            if isinstance(results, list) and results:

                ascending = bool(
                    operation_result.get(
                        "ascending",
                        False,
                    )
                )

                direction = (
                    "Lowest"
                    if ascending
                    else "Highest"
                )

                for row in results:

                    if not isinstance(row, dict):
                        continue

                    ranking_rows.append(
                        {
                            "Direction": direction,
                            "Rank": row.get("rank"),
                            "Region": row.get(
                                "group",
                                row.get(
                                    "Region",
                                    "—",
                                ),
                            ),
                            "Value": row.get(
                                "value",
                                row.get(
                                    "Revenue",
                                    "—",
                                ),
                            ),
                        }
                    )

        # --------------------------------------------------------
        # CATEGORICAL ANALYSIS
        # --------------------------------------------------------

        elif operation == "categorical_analysis":

            counts = operation_result.get("counts")

            if isinstance(counts, dict) and counts:

                rows = [
                    {
                        "Category": category,
                        "Count": count,
                    }
                    for category, count in counts.items()
                ]

                other_tables.append(
                    (
                        "Category Distribution",
                        rows,
                    )
                )

        # --------------------------------------------------------
        # STATISTICS
        # --------------------------------------------------------

        elif operation == "statistics":

            statistics = operation_result.get(
                "statistics"
            )

            if isinstance(statistics, dict) and statistics:

                rows = [
                    {
                        "Metric": key,
                        "Value": value,
                    }
                    for key, value in statistics.items()
                ]

                other_tables.append(
                    (
                        "Statistical Summary",
                        rows,
                    )
                )

    # ------------------------------------------------------------
    # NOTHING TO DISPLAY
    # ------------------------------------------------------------

    if (
        not group_rows
        and not ranking_rows
        and not other_tables
    ):
        return

    st.markdown("### 📋 Analysis Results")

    # ------------------------------------------------------------
    # REGIONAL / GROUP ANALYSIS
    # ------------------------------------------------------------

    if group_rows:

        display_df = pd.DataFrame(
            group_rows
        ).drop_duplicates()

        numeric_columns = [
            column
            for column in display_df.columns
            if pd.api.types.is_numeric_dtype(
                display_df[column]
            )
        ]

        if numeric_columns:

            value_column = numeric_columns[-1]

            display_df = display_df.sort_values(
                by=value_column,
                ascending=False,
                kind="stable",
            )

        st.markdown(
            "#### 🌎 Regional / Group Analysis"
        )

        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
        )

    # ------------------------------------------------------------
    # PERFORMANCE HIGHLIGHTS
    # ------------------------------------------------------------

    if ranking_rows:

        ranking_df = pd.DataFrame(
            ranking_rows
        ).drop_duplicates(
            subset=[
                "Direction",
                "Region",
                "Value",
            ]
        )

        st.markdown(
            "#### 🏆 Performance Highlights"
        )

        st.dataframe(
            ranking_df,
            width="stretch",
            hide_index=True,
        )

    # ------------------------------------------------------------
    # OTHER STRUCTURED RESULTS
    # ------------------------------------------------------------

    for title, rows in other_tables:

        st.markdown(
            f"#### {title}"
        )

        display_df = pd.DataFrame(rows)

        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
        )
# ============================================================
# EXECUTION SUMMARY
# ============================================================

def render_execution_summary(result):
    """Render execution metrics."""

    execution = result.get("execution") or {}
    execution_results = execution.get("results", [])

    if not execution_results:
        return

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

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Operations",
            len(execution_results),
        )

    with col2:
        st.metric(
            "Successful",
            successful_steps,
        )

    with col3:
        st.metric(
            "Failed",
            failed_steps,
        )


# ============================================================
# CHARTS
# ============================================================

def render_charts(chart_paths):
    """Render generated charts."""

    if not chart_paths:
        return

    st.markdown("### 📊 Visualizations")

    for index, chart_path in enumerate(
        chart_paths,
        start=1,
    ):

        path = Path(chart_path)

        if path.exists():

            st.image(
                str(path),
                caption=f"Generated chart {index}",
                width="stretch",
            )

        else:

            st.warning(
                f"Chart {index} was generated but "
                f"the file could not be found: "
                f"`{chart_path}`"
            )


# ============================================================
# EXECUTION OPERATIONS
# ============================================================

def render_execution_operations(result):
    """Show executed operations."""

    execution = result.get("execution") or {}
    execution_results = execution.get("results", [])

    if not execution_results:
        return

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

            status = item.get(
                "status",
                "unknown",
            )

            if status == "success":

                st.success(
                    f"{index}. `{operation}` → success"
                )

            else:

                st.error(
                    f"{index}. `{operation}` → {status}"
                )


# ============================================================
# CONVERSATION RENDERING
# ============================================================

def render_conversation_item(item):
    """Render one previous analysis."""

    query = item.get("query", "")
    response = item.get("response", "")
    result = item.get("result") or {}
    chart_paths = item.get("chart_paths", [])

    status = result.get("status")

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        if status == "success":
            st.success(
                "Analysis completed successfully."
            )

        elif status in {
            "partial",
            "partial_success",
        }:
            st.warning(
                "Analysis completed with some errors."
            )

        else:
            st.error(
                "Analysis failed."
            )

        if response:
            st.markdown(response)

        render_structured_results(result)

        render_business_insights(result)

        render_charts(chart_paths)

        render_execution_summary(result)

        render_execution_operations(result)

        with st.expander(
            "🔍 Analysis Details",
            expanded=False,
        ):
            st.json(result)


# ============================================================
# HEADER
# ============================================================

st.title(
    "📊 Autonomous AI Data Analyst"
)

st.markdown(
    """
Upload a CSV or Excel business file and let an autonomous AI
system **plan, execute, analyze, visualize, and explain**
your data.
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
        "NVIDIA Nemotron is currently configured as "
        "the primary analysis model."
    )

    st.divider()

    if st.session_state.dataset is not None:

        st.markdown("### 📁 Current Dataset")

        st.caption(
            st.session_state.dataset_name
        )

        rows, columns = (
            st.session_state.dataset.shape
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Rows",
                f"{rows:,}",
            )

        with col2:
            st.metric(
                "Columns",
                f"{columns:,}",
            )

        st.divider()

        st.markdown("### 💬 Session")

        st.metric(
            "Analyses",
            st.session_state.analysis_count,
        )

        if st.button(
            "🗑️ Clear Conversation",
            width="stretch",
        ):

            st.session_state.conversation = []
            st.session_state.analysis_count = 0

            st.rerun()

    st.divider()

    st.caption(
        "Upload → Plan → Execute → Recover → Insight → Explain"
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
# DATASET MANAGEMENT
# ============================================================

if uploaded_file is not None:

    file_changed = (
        st.session_state.dataset_name
        != uploaded_file.name
    )

    if file_changed:

        try:

            new_df = load_uploaded_file(
                uploaded_file
            )

        except Exception as exc:

            st.error(
                f"Could not read the uploaded file: {exc}"
            )

            st.stop()

        st.session_state.dataset = new_df
        st.session_state.dataset_name = (
            uploaded_file.name
        )

        st.session_state.conversation = []
        st.session_state.analysis_count = 0

        st.success(
            f"Loaded `{uploaded_file.name}` successfully."
        )

else:

    if st.session_state.dataset is None:

        st.info(
            "Upload a CSV or Excel file to start your analysis."
        )

        st.markdown(
            """
### What this system can do

- 📊 Analyze business datasets
- 🔢 Calculate business metrics
- 🧮 Perform aggregations
- 🏆 Find highest/lowest performers
- 📈 Generate visualizations
- 🤖 Automatically plan analysis steps
- 🔄 Recover from execution failures
- 💡 Detect verified business insights
- 🧠 Generate business-friendly explanations
- 💬 Continue asking questions about the same dataset
"""
        )

        st.stop()


# ============================================================
# CURRENT DATASET
# ============================================================

df = st.session_state.dataset

st.success(
    f"Dataset ready: `{st.session_state.dataset_name}`"
)

# ============================================================
# DATASET OVERVIEW
# ============================================================

render_dataset_overview(df)


# ============================================================
# DATASET PREVIEW
# ============================================================

with st.expander(
    "👀 Preview Dataset",
    expanded=False,
):

    st.dataframe(
        df.head(20),
        width="stretch",
    )


# ============================================================
# ANALYSIS HISTORY
# ============================================================

if st.session_state.conversation:

    st.markdown("## 💬 Analysis History")

    for item in st.session_state.conversation:
        render_conversation_item(item)


# ============================================================
# CURRENT QUERY
# ============================================================

st.markdown("## 🔎 Ask Your Data Analyst")

user_query = st.chat_input(
    "Ask a business question about your dataset..."
)


# ============================================================
# ANALYSIS EXECUTION
# ============================================================

if user_query:

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):

        with st.spinner(
            "🤖 Autonomous analyst is working..."
        ):

            try:

                pipeline = AnalysisPipeline(
                    provider=provider
                )

                result = pipeline.run(
                    df=df,
                    user_query=user_query,
                    conversation_history=[
                        {
                            "query": item.get(
                                "query",
                                "",
                            ),
                            "response": item.get(
                                "response",
                                "",
                            ),
                        }
                        for item in st.session_state.conversation
                    ],
                )

            except Exception as exc:

                st.error(
                    f"Unexpected application error: {exc}"
                )

                st.stop()

        status = result.get("status")

        if status == "success":

            st.success(
                "Analysis completed successfully."
            )

        elif status in {
            "partial",
            "partial_success",
        }:

            st.warning(
                "Analysis completed with some errors."
            )

        else:

            st.error(
                "Analysis failed."
            )

        final_response = (
            result.get("final_response")
            or "No final response was generated."
        )

        st.markdown(final_response)

        # ----------------------------------------------------
        # STRUCTURED ANALYSIS RESULTS
        # ----------------------------------------------------

        render_structured_results(result)

        # ----------------------------------------------------
        # VERIFIED BUSINESS INSIGHTS
        # ----------------------------------------------------

        render_business_insights(result)

        # ----------------------------------------------------
        # CHARTS
        # ----------------------------------------------------

        chart_paths = extract_chart_paths(
            result
        )

        render_charts(
            chart_paths
        )

        # ----------------------------------------------------
        # EXECUTION SUMMARY
        # ----------------------------------------------------

        render_execution_summary(
            result
        )

        # ----------------------------------------------------
        # EXECUTION OPERATIONS
        # ----------------------------------------------------

        render_execution_operations(
            result
        )

        # ----------------------------------------------------
        # DETAILS
        # ----------------------------------------------------

        with st.expander(
            "🔍 Analysis Details",
            expanded=False,
        ):

            st.json(result)

    # --------------------------------------------------------
    # SAVE CONVERSATION
    # --------------------------------------------------------

    st.session_state.conversation.append(
        {
            "query": user_query,
            "response": final_response,
            "result": result,
            "chart_paths": chart_paths,
        }
    )

    st.session_state.analysis_count += 1

    st.rerun()
