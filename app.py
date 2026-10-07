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
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown(
    """
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    .hero {
        padding: 1.5rem 1.75rem;
        border: 1px solid rgba(128,128,128,.20);
        border-radius: 18px;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        margin-bottom: .35rem;
        font-size: 2.25rem;
    }

    .hero p {
        margin-bottom: 0;
        opacity: .75;
        font-size: 1.05rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 1.25rem;
        margin-bottom: .75rem;
    }

    .status-card {
        padding: 1rem 1.1rem;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,.18);
        min-height: 90px;
    }

    .status-label {
        font-size: .78rem;
        opacity: .65;
        text-transform: uppercase;
        letter-spacing: .04em;
    }

    .status-value {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: .25rem;
    }

    .upload-card {
        padding: 1.25rem;
        border-radius: 16px;
        border: 1px dashed rgba(128,128,128,.35);
        margin-bottom: 1rem;
    }

    .small-muted {
        opacity: .65;
        font-size: .85rem;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 14px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "dataset": None,
    "dataset_name": None,
    "conversation": [],
    "analysis_count": 0,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# DATA LOADING
# ============================================================

def load_uploaded_file(uploaded_file):
    """Load CSV or Excel data."""

    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if filename.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file type.")


# ============================================================
# RESULT HELPERS
# ============================================================

def get_execution_results(result):
    execution = result.get("execution") or {}

    results = execution.get("results", [])

    return results if isinstance(results, list) else []


def extract_chart_paths(result):
    """Extract generated chart image paths."""

    chart_paths = []

    for item in get_execution_results(result):

        if item.get("status") != "success":
            continue

        operation = item.get("operation")

        if operation not in {
            "generate_bar_chart",
            "generate_line_chart",
        }:
            continue

        operation_result = item.get("result") or {}

        if not isinstance(operation_result, dict):
            continue

        output_path = operation_result.get("output_path")

        if output_path:
            chart_paths.append(output_path)

    return chart_paths


def extract_insights(result):
    """Extract pipeline-generated insights."""

    insights_wrapper = result.get("insights") or {}

    if not isinstance(insights_wrapper, dict):
        return []

    insights = insights_wrapper.get("insights", [])

    return insights if isinstance(insights, list) else []


# ============================================================
# DATASET HEALTH
# ============================================================

def render_dataset_health(df):
    """Render dataset health metrics."""

    if df is None or df.empty:
        return

    rows, columns = df.shape

    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    total_cells = rows * columns

    missing_pct = (
        missing_cells / total_cells * 100
        if total_cells
        else 0
    )

    st.markdown(
        '<div class="section-title">📊 Dataset Health</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Rows", f"{rows:,}")

    with c2:
        st.metric("Columns", f"{columns:,}")

    with c3:
        st.metric(
            "Missing Cells",
            f"{missing_cells:,}",
            f"{missing_pct:.1f}%",
            delta_color="inverse",
        )

    with c4:
        st.metric(
            "Duplicate Rows",
            f"{duplicate_rows:,}",
        )


# ============================================================
# DATASET PREVIEW
# ============================================================

def render_dataset_preview(df):
    st.markdown(
        '<div class="section-title">👀 Dataset Preview</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Preview first 20 rows", expanded=False):
        st.dataframe(
            df.head(20),
            width="stretch",
            hide_index=True,
        )


# ============================================================
# STRUCTURED RESULTS
# ============================================================

def render_structured_results(result):
    """Render structured outputs from successful operations."""

    group_rows = []
    ranking_rows = []
    other_tables = []

    for item in get_execution_results(result):

        if item.get("status") != "success":
            continue

        operation = item.get("operation", "")
        operation_result = item.get("result") or {}

        if not isinstance(operation_result, dict):
            continue

        # GROUPBY
        if operation in {
            "groupby_aggregation",
            "groupby_aggregate",
            "group_by",
        }:

            results = operation_result.get("results")

            if isinstance(results, list):

                for row in results:
                    if isinstance(row, dict):
                        group_rows.append(row)

        # RANKING
        elif operation == "rank_by_value":

            results = operation_result.get("results")

            if isinstance(results, list):

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
                            "Group": row.get(
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

        # CATEGORICAL
        elif operation == "categorical_analysis":

            counts = operation_result.get("counts")

            if isinstance(counts, dict) and counts:

                other_tables.append(
                    (
                        "Category Distribution",
                        [
                            {
                                "Category": category,
                                "Count": count,
                            }
                            for category, count
                            in counts.items()
                        ],
                    )
                )

        # STATISTICS
        elif operation in {
            "statistics",
            "calculate_statistics",
        }:

            statistics = operation_result.get(
                "statistics"
            )

            if isinstance(statistics, dict) and statistics:

                other_tables.append(
                    (
                        "Statistical Summary",
                        [
                            {
                                "Metric": key,
                                "Value": value,
                            }
                            for key, value
                            in statistics.items()
                        ],
                    )
                )

    if not (
        group_rows
        or ranking_rows
        or other_tables
    ):
        return

    st.markdown(
        '<div class="section-title">📋 Analysis Results</div>',
        unsafe_allow_html=True,
    )

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

        st.markdown("#### 🌎 Group Analysis")

        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
        )

    if ranking_rows:

        ranking_df = pd.DataFrame(
            ranking_rows
        ).drop_duplicates()

        st.markdown("#### 🏆 Performance Highlights")

        st.dataframe(
            ranking_df,
            width="stretch",
            hide_index=True,
        )

    for title, rows in other_tables:

        st.markdown(f"#### {title}")

        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
        )


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

def render_business_insights(result):
    """Render verified business insights."""

    insights = extract_insights(result)

    if not insights:
        return

    st.markdown(
        '<div class="section-title">💡 Business Insights</div>',
        unsafe_allow_html=True,
    )

    for insight in insights:

        if not isinstance(insight, dict):
            continue

        message = insight.get("message")

        if not message:
            continue

        insight_type = str(
            insight.get("type", "")
        ).lower()

        if (
            "highest" in insight_type
            or "maximum" in insight_type
            or "increase" in message.lower()
        ):
            st.success(f"📈 {message}")

        elif (
            "lowest" in insight_type
            or "decrease" in message.lower()
        ):
            st.warning(f"📉 {message}")

        else:
            st.info(f"🔎 {message}")


# ============================================================
# CHARTS
# ============================================================

def render_charts(chart_paths):
    """Render generated chart images."""

    valid_paths = []

    for chart_path in chart_paths:

        if Path(chart_path).exists():
            valid_paths.append(chart_path)

    if not valid_paths:
        return

    st.markdown(
        '<div class="section-title">📈 Visualizations</div>',
        unsafe_allow_html=True,
    )

    columns = st.columns(
        min(2, len(valid_paths))
    )

    for index, chart_path in enumerate(valid_paths):

        with columns[index % len(columns)]:

            st.image(
                chart_path,
                caption=f"Generated visualization {index + 1}",
                width="stretch",
            )


# ============================================================
# EXECUTION SUMMARY
# ============================================================

def render_execution_summary(result):
    """Render execution statistics."""

    execution_results = get_execution_results(result)

    if not execution_results:
        return

    successful = sum(
        1
        for item in execution_results
        if item.get("status") == "success"
    )

    failed = len(execution_results) - successful

    autonomous = result.get("autonomous") or {}

    rounds = autonomous.get("rounds")

    st.markdown(
        '<div class="section-title">⚙️ Execution Summary</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Operations",
            len(execution_results),
        )

    with c2:
        st.metric(
            "Successful",
            successful,
        )

    with c3:
        st.metric(
            "Failed",
            failed,
        )

    with c4:
        if rounds is not None:
            st.metric(
                "Autonomous Rounds",
                rounds,
            )
        else:
            st.metric(
                "Autonomous",
                "Completed",
            )


# ============================================================
# EXECUTED OPERATIONS
# ============================================================

def render_execution_operations(result):

    execution_results = get_execution_results(result)

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
# ANALYSIS DETAILS
# ============================================================

def render_analysis_details(result):

    with st.expander(
        "🔍 Technical Analysis Details",
        expanded=False,
    ):
        st.json(result)


# ============================================================
# ANALYSIS RESULT
# ============================================================

def render_analysis_result(
    result,
    final_response,
    chart_paths,
):
    """Render a completed analysis."""

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

    if final_response:
        st.markdown(final_response)

    render_business_insights(result)

    render_structured_results(result)

    render_charts(chart_paths)

    render_execution_summary(result)

    render_execution_operations(result)

    render_analysis_details(result)


# ============================================================
# CONVERSATION HISTORY
# ============================================================

def render_conversation_item(item):

    query = item.get("query", "")
    response = item.get("response", "")
    result = item.get("result") or {}
    chart_paths = item.get("chart_paths", [])

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        render_analysis_result(
            result=result,
            final_response=response,
            chart_paths=chart_paths,
        )


def render_conversation_history():

    if not st.session_state.conversation:
        return

    st.markdown(
        '<div class="section-title">💬 Analysis History</div>',
        unsafe_allow_html=True,
    )

    for item in st.session_state.conversation:
        render_conversation_item(item)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ Analyst Settings")

    provider = st.selectbox(
        "LLM Provider",
        options=["nvidia"],
        index=0,
    )

    st.caption(
        "NVIDIA Nemotron is the configured analysis provider."
    )

    st.divider()

    if st.session_state.dataset is not None:

        st.markdown("### 📁 Current Dataset")

        st.write(
            st.session_state.dataset_name
        )

        rows, columns = (
            st.session_state.dataset.shape
        )

        c1, c2 = st.columns(2)

        with c1:
            st.metric("Rows", f"{rows:,}")

        with c2:
            st.metric("Columns", f"{columns:,}")

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
        "Upload → Understand → Plan → Execute → Recover → Explain"
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">
    <h1>📊 Autonomous AI Data Analyst</h1>
    <p>
        Upload business data, ask questions in natural language,
        and let the analyst plan, execute, visualize and explain
        the investigation automatically.
    </p>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">📁 Your Dataset</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel business file",
    type=["csv", "xlsx"],
    help="Supported formats: CSV and XLSX.",
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

            if new_df.empty:
                st.error(
                    "The uploaded dataset is empty."
                )
                st.stop()

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

elif st.session_state.dataset is None:

    st.info(
        "Upload a CSV or Excel file above to start."
    )

    st.markdown(
        """
### 🤖 What your analyst can do

- 📊 Understand business datasets
- 🔢 Calculate metrics
- 🧮 Aggregate and compare data
- 🏆 Find highest and lowest performers
- 📈 Detect trends
- 📉 Generate charts
- 🔄 Recover from execution problems
- 💡 Detect verified business insights
- 🧠 Continue multi-step investigations
- 💬 Answer follow-up questions about the same dataset
"""
    )

    st.stop()


# ============================================================
# CURRENT DATASET
# ============================================================

df = st.session_state.dataset

st.caption(
    f"Active dataset: `{st.session_state.dataset_name}`"
)

render_dataset_health(df)

render_dataset_preview(df)


# ============================================================
# HISTORY
# ============================================================

render_conversation_history()


# ============================================================
# ANALYST CHAT
# ============================================================

st.markdown(
    '<div class="section-title">🤖 Ask Your Data Analyst</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Ask a business question in natural language. "
    "The autonomous engine decides which analysis steps are required."
)

user_query = st.chat_input(
    "Example: Which region generated the highest revenue?"
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
                        for item
                        in st.session_state.conversation
                    ],
                )

            except Exception as exc:

                st.error(
                    f"Unexpected application error: {exc}"
                )

                st.stop()

        final_response = (
            result.get("final_response")
            or "No final response was generated."
        )

        chart_paths = extract_chart_paths(
            result
        )

        render_analysis_result(
            result=result,
            final_response=final_response,
            chart_paths=chart_paths,
        )

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
