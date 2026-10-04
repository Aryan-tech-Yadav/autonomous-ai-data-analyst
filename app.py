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
# HELPER FUNCTIONS
# ============================================================

def load_uploaded_file(uploaded_file):
    """
    Load CSV or Excel file into a DataFrame.
    """

    if uploaded_file.name.lower().endswith(".csv"):

        return pd.read_csv(uploaded_file)

    return pd.read_excel(uploaded_file)


def extract_chart_paths(result):
    """
    Extract generated chart paths from pipeline execution.
    """

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

    return chart_paths


def render_execution_summary(result):
    """
    Render compact execution summary.
    """

    execution = result.get(
        "execution",
        {},
    )

    execution_results = execution.get(
        "results",
        [],
    )

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


def render_charts(chart_paths):
    """
    Render generated charts.
    """

    if not chart_paths:
        return

    st.markdown("#### 📊 Visualization")

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


def render_execution_operations(result):
    """
    Render executed operation names.
    """

    execution = result.get(
        "execution",
        {},
    )

    execution_results = execution.get(
        "results",
        [],
    )

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


def render_conversation_item(item, index):
    """
    Render one previous analysis in the conversation.
    """

    query = item.get(
        "query",
        "",
    )

    response = item.get(
        "response",
        "",
    )

    result = item.get(
        "result",
        {},
    )

    chart_paths = item.get(
        "chart_paths",
        [],
    )

    status = result.get(
        "status"
    )

    with st.chat_message("user"):

        st.markdown(
            query
        )

    with st.chat_message("assistant"):

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

        if response:

            st.markdown(
                response
            )

        render_charts(
            chart_paths
        )

        render_execution_summary(
            result
        )

        render_execution_operations(
            result
        )

        with st.expander(
            "🔍 Analysis Details",
            expanded=False,
        ):

            st.json(
                result
            )


# ============================================================
# HEADER
# ============================================================

st.title(
    "📊 Autonomous AI Data Analyst"
)

st.markdown(
    """
Upload a CSV or Excel business file and have an autonomous AI
plan, execute, analyze, visualize, and explain your data.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Configuration"
    )

    provider = st.selectbox(
        "LLM Provider",
        options=[
            "nvidia",
        ],
        index=0,
    )

    st.divider()

    st.markdown(
        "### 🤖 AI Engine"
    )

    st.info(
        "Nemotron is currently configured as the "
        "primary analysis model."
    )

    st.divider()

    if st.session_state.dataset is not None:

        st.markdown(
            "### 📁 Current Dataset"
        )

        st.caption(
            st.session_state.dataset_name
        )

        st.metric(
            "Rows",
            f"{st.session_state.dataset.shape[0]:,}",
        )

        st.metric(
            "Columns",
            f"{st.session_state.dataset.shape[1]:,}",
        )

        st.divider()

        st.markdown(
            "### 💬 Session"
        )

        st.metric(
            "Analyses",
            st.session_state.analysis_count,
        )

        if st.button(
            "🗑️ Clear Conversation",
            use_container_width=True,
        ):

            st.session_state.conversation = []
            st.session_state.analysis_count = 0

            st.rerun()

    st.divider()

    st.caption(
        "Upload → Plan → Execute → Analyze → Explain"
    )


# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader(
    "📁 Upload Business Data"
)

uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=[
        "csv",
        "xlsx",
    ],
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
- 🔢 Calculate metrics
- 🧮 Perform aggregations
- 🏆 Find highest/lowest values
- 📈 Generate charts
- 🤖 Automatically plan analysis steps
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
# DATASET METRICS
# ============================================================

row_count = df.shape[0]
column_count = df.shape[1]
missing_count = int(
    df.isna().sum().sum()
)
duplicate_count = int(
    df.duplicated().sum()
)

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
                int(
                    df[column].isna().sum()
                )
                for column in df.columns
            ],
            "Unique": [
                int(
                    df[column].nunique(
                        dropna=True
                    )
                )
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
# CONVERSATION HISTORY
# ============================================================

if st.session_state.conversation:

    st.subheader(
        "💬 Analysis Conversation"
    )

    for index, item in enumerate(
        st.session_state.conversation,
        start=1,
    ):

        render_conversation_item(
            item,
            index,
        )


# ============================================================
# NEW QUESTION
# ============================================================

st.subheader(
    "💬 Ask Your Data"
)

user_query = st.chat_input(
    "Ask a question about your dataset..."
)


# ============================================================
# ANALYSIS
# ============================================================

if user_query:

    user_query = user_query.strip()

    if not user_query:

        st.warning(
            "Please enter a question first."
        )

        st.stop()

    with st.chat_message("user"):

        st.markdown(
            user_query
        )

    with st.chat_message("assistant"):

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


        # ====================================================
        # PIPELINE STATUS
        # ====================================================

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


        # ====================================================
        # FINAL RESPONSE
        # ====================================================

        final_response = result.get(
            "final_response"
        )

        if final_response:

            st.markdown(
                final_response
            )

        else:

            st.warning(
                "No final AI response was generated."
            )


        # ====================================================
        # CHARTS
        # ====================================================

        chart_paths = extract_chart_paths(
            result
        )

        render_charts(
            chart_paths
        )


        # ====================================================
        # EXECUTION SUMMARY
        # ====================================================

        render_execution_summary(
            result
        )


        # ====================================================
        # EXECUTION OPERATIONS
        # ====================================================

        render_execution_operations(
            result
        )


        # ====================================================
        # ANALYSIS DETAILS
        # ====================================================

        with st.expander(
            "🔍 Analysis Details",
            expanded=False,
        ):

            st.json(
                result
            )


    # ========================================================
    # SAVE CONVERSATION
    # ========================================================

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


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Autonomous AI Data Analyst • "
    "Nemotron-powered autonomous analysis"
)
