"""
app.py — Simple Streamlit UI for the Intelligent Form Agent

Run with:
    uv run streamlit run app.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

# ── Path setup so `src/` modules are importable ──────────────────────────────
SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import llm_client

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Form Agent",
    page_icon="📋",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────────────────────
# Agent singleton — one per browser session to avoid SQLite threading issues
# ─────────────────────────────────────────────────────────────────────────────

def agent():
    if "_form_agent" not in st.session_state:
        from agent import FormAgent
        st.session_state["_form_agent"] = FormAgent()
    return st.session_state["_form_agent"]


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📋 Form Agent")

    # Ollama status
    ollama_ok = llm_client.is_available()
    if ollama_ok:
        st.success(f"🟢 Ollama online\n\nModel: `{llm_client.DEFAULT_MODEL}`")
    else:
        st.warning("🟡 Ollama offline\n\nLLM features degraded. Run `ollama serve`.")

    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "⬆️ Upload Forms", "🔍 Form Explorer", "💬 Ask the Agent"],
    )

    st.divider()

    # Quick stats
    st.subheader("Quick Stats")
    try:
        ag = agent()
        all_ids = ag.list_form_ids()
        mem_ids = ag.list_form_ids("membership")
        hosp_ids = ag.list_form_ids("hospital")
        st.metric("Total Forms", len(all_ids))
        c1, c2 = st.columns(2)
        c1.metric("Membership", len(mem_ids))
        c2.metric("Hospital", len(hosp_ids))
    except Exception:
        st.caption("No forms loaded yet.")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Dashboard
# ─────────────────────────────────────────────────────────────────────────────

if page == "🏠 Dashboard":
    st.title("📋 Intelligent Form Agent")
    st.caption("Local RAG pipeline · Schema-guided extraction · Grounded Q&A with citations")
    st.divider()

    try:
        ag = agent()
        all_ids = ag.list_form_ids()
        mem_ids = ag.list_form_ids("membership")
        hosp_ids = ag.list_form_ids("hospital")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Forms", len(all_ids))
        col2.metric("Membership", len(mem_ids))
        col3.metric("Hospital", len(hosp_ids))
        col4.metric("LLM Online", "✅" if ollama_ok else "❌")

        st.divider()

        if all_ids:
            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("📝 Membership Status")
                if mem_ids:
                    import structured_store as ss
                    conn = ag.conn
                    approved = ss.count_forms(conn, "membership", {"status": "Approved"})
                    pending  = ss.count_forms(conn, "membership", {"status": "Pending"})
                    rejected = ss.count_forms(conn, "membership", {"status": "Rejected"})
                    st.write(f"✅ Approved: **{approved}**")
                    st.write(f"🕐 Pending: **{pending}**")
                    st.write(f"❌ Rejected: **{rejected}**")
                else:
                    st.caption("No membership forms loaded.")

            with col_right:
                st.subheader("🗂 Loaded Forms")
                import structured_store as ss
                conn = ag.conn
                for fid in all_ids[:15]:
                    ftype = ss.get_form_type(conn, fid)
                    badge = "🏥" if ftype == "hospital" else "📝"
                    st.write(f"{badge} `{fid}`")
                if len(all_ids) > 15:
                    st.caption(f"… and {len(all_ids) - 15} more")
        else:
            st.info("No forms loaded yet. Go to **⬆️ Upload Forms** to get started.")

    except Exception as e:
        st.error(f"Could not initialize agent: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Upload Forms
# ─────────────────────────────────────────────────────────────────────────────

elif page == "⬆️ Upload Forms":
    st.title("⬆️ Upload Forms")
    st.caption("Ingest PDF or DOCX forms — extraction, validation, and indexing happen automatically.")
    st.divider()

    tab_single, tab_dir = st.tabs(["📄 Single File", "📁 Load Directory"])

    with tab_single:
        uploaded = st.file_uploader("Drop a PDF or DOCX form", type=["pdf", "docx"])
        custom_id = st.text_input(
            "Custom form ID (optional)",
            placeholder="e.g. membership_007",
        )

        if uploaded and st.button("🚀 Ingest Form"):
            import tempfile, os
            suffix = Path(uploaded.name).suffix
            # Derive a clean default form_id from the original filename,
            # NOT the temp path — prevents junk IDs like 'tmpnzew1_zk'.
            default_id = custom_id.strip() or Path(uploaded.name).stem
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            with st.spinner("Extracting, validating, and indexing…"):
                try:
                    ag = agent()
                    fid, chunk_count = ag.load_form(tmp_path, form_id=default_id)
                    st.success(f"✅ Ingested as `{fid}` · {chunk_count} chunk(s) embedded")
                    st.balloons()
                    import structured_store as ss
                    form = ss.get_form(ag.conn, fid)
                    if form:
                        st.subheader("Extracted Fields")
                        display = {k: v for k, v in form.items() if k not in ("form_id", "form_type")}
                        st.json(display)
                except Exception as e:
                    st.error(f"❌ Ingestion failed: {e}")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    with tab_dir:
        dir_path = st.text_input("Directory path containing forms", value="data/sample_forms")
        if st.button("📂 Load Directory"):
            with st.spinner(f"Loading all forms from `{dir_path}`…"):
                try:
                    ag = agent()
                    report = ag.load_directory(dir_path)
                    if report.loaded:
                        st.success(f"✅ Loaded {len(report.loaded)} form(s): {', '.join(report.loaded)}")
                    if report.failed:
                        for src, err in report.failed:
                            st.error(f"❌ `{src}`: {err}")
                    if not report.loaded and not report.failed:
                        st.warning("No supported files found in that directory.")
                except Exception as e:
                    st.error(f"❌ {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Form Explorer
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🔍 Form Explorer":
    st.title("🔍 Form Explorer")
    st.caption("Browse structured fields and ask questions about a specific form.")
    st.divider()

    try:
        ag = agent()
        all_ids = ag.list_form_ids()
    except Exception as e:
        st.error(f"Agent error: {e}")
        st.stop()

    if not all_ids:
        st.info("No forms loaded. Go to **⬆️ Upload Forms** first.")
        st.stop()

    selected_id = st.selectbox("Select a form", options=all_ids)

    import structured_store as ss
    form = ss.get_form(ag.conn, selected_id)

    if form is None:
        st.error(f"Could not retrieve data for `{selected_id}`.")
        st.stop()

    form_type = form.get("form_type", "unknown")
    type_label = "🏥 Hospital" if form_type == "hospital" else "📝 Membership"
    st.write(f"**Form:** `{selected_id}` | **Type:** {type_label}")

    tab_fields, tab_qa, tab_summary = st.tabs(["📋 Fields", "💬 Q&A", "📝 Summary"])

    with tab_fields:
        display = {k: v for k, v in form.items() if k not in ("form_id", "form_type")}
        st.json(display)

    with tab_qa:
        if form_type == "membership":
            suggestions = [
                "What is the application status?",
                "What are the remarks for this applicant?",
                "What is the applicant's occupation?",
                "Tell me about this applicant's credit history.",
            ]
        else:
            suggestions = [
                "What is the diagnosis?",
                "What are the doctor's notes?",
                "When was the patient admitted and discharged?",
                "What department treated this patient?",
            ]

        st.write("**Suggested questions:**")
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            if cols[i % 2].button(s, key=f"sug_{s}"):
                st.session_state["form_question"] = s

        question = st.text_input(
            "Your question",
            value=st.session_state.get("form_question", ""),
            placeholder="e.g. What is the status of this application?",
        )

        if st.button("🔍 Ask", key="btn_form_ask") and question.strip():
            with st.spinner("Routing & answering…"):
                try:
                    result = ag.answer_question(question.strip(), form_id=selected_id)
                    st.info(result.answer)
                    if result.cited_form_ids:
                        st.caption(f"📎 Sources: {', '.join(result.cited_form_ids)}")
                    if not result.grounded:
                        st.warning("⚠️ Answer could not be fully grounded in the form's content.")
                except Exception as e:
                    st.error(f"❌ {e}")

    with tab_summary:
        st.write(f"Generate a concise LLM summary of **`{selected_id}`**.")
        if not ollama_ok:
            st.warning("⚠️ Ollama is offline — summary requires the LLM to be running.")
        else:
            if st.button("📝 Generate Summary"):
                with st.spinner("Generating summary…"):
                    try:
                        result = ag.summarize_form(selected_id)
                        st.info(result.answer)
                        if result.cited_form_ids:
                            st.caption(f"📎 Sources: {', '.join(result.cited_form_ids)}")
                    except Exception as e:
                        st.error(f"❌ {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Ask the Agent (cross-form)
# ─────────────────────────────────────────────────────────────────────────────

elif page == "💬 Ask the Agent":
    st.title("💬 Ask the Agent")
    st.caption("Cross-form Q&A — aggregate counts, semantic search across all forms, or multi-form patterns.")
    st.divider()

    try:
        ag = agent()
        all_ids = ag.list_form_ids()
    except Exception as e:
        st.error(f"Agent error: {e}")
        st.stop()

    if not all_ids:
        st.info("No forms loaded. Go to **⬆️ Upload Forms** first.")
        st.stop()

    tab_auto, tab_multi = st.tabs(["🤖 Auto-Route Q&A", "🌐 Multi-Form Semantic"])

    with tab_auto:
        st.write(
            "The router automatically classifies your question: "
            "`aggregate` → SQL, `single_form_lookup` → exact field, "
            "`single_form_semantic` → vector search on one form, "
            "`multi_form_semantic` → hybrid retrieval across all forms."
        )

        st.write("**📊 Aggregate examples (SQL):**")
        aggregate_examples = [
            "How many membership applications are approved?",
            "How many hospital forms are in Cardiology?",
            "How many applications are pending?",
            "How many forms are rejected?",
            "List all membership forms.",
        ]
        cols = st.columns(2)
        for i, ex in enumerate(aggregate_examples):
            if cols[i % 2].button(ex, key=f"agg_{ex}"):
                st.session_state["cross_question"] = ex

        st.write("**🔍 Semantic examples (hybrid retrieval):**")
        semantic_examples = [
            "What patterns show up across all rejected applicants?",
            "Which applicants have credit defaults?",
            "What are the most common diagnoses?",
            "Summarize the clinical notes across all hospital forms.",
        ]
        cols2 = st.columns(2)
        for i, ex in enumerate(semantic_examples):
            if cols2[i % 2].button(ex, key=f"sem_{ex}"):
                st.session_state["cross_question"] = ex

        st.divider()
        cross_q = st.text_area(
            "Your question",
            value=st.session_state.get("cross_question", ""),
            height=90,
            placeholder="Ask anything about the loaded forms…",
        )

        if st.button("🚀 Ask", key="btn_cross_ask") and cross_q.strip():
            with st.spinner("Routing & answering…"):
                try:
                    import router as rt
                    t0 = time.time()
                    decision = rt.route_question(cross_q.strip())
                    result = ag.answer_question(cross_q.strip())
                    elapsed = time.time() - t0

                    st.caption(f"Route: `{decision.route.value}` via {decision.detection_method} · {elapsed:.1f}s")
                    st.info(result.answer)
                    if result.cited_form_ids:
                        st.caption(f"📎 Sources: {', '.join(result.cited_form_ids)}")
                    if not result.grounded:
                        st.warning("⚠️ Answer could not be fully grounded.")
                except Exception as e:
                    st.error(f"❌ {e}")

    with tab_multi:
        st.write(
            "This tab **always** routes to multi-form semantic retrieval (hybrid BM25 + "
            "embeddings across all forms) — bypasses the auto-router for explicit cross-corpus search."
        )

        if not ollama_ok:
            st.warning("⚠️ Ollama is offline — semantic search requires the LLM for synthesis.")

        multi_q = st.text_area(
            "Cross-form semantic question",
            height=100,
            placeholder="e.g. What patterns show up across all rejected applicants?",
        )

        if st.button("🌐 Search All Forms") and multi_q.strip():
            with st.spinner("Running hybrid retrieval + synthesis…"):
                try:
                    t0 = time.time()
                    result = ag.multi_form_query(multi_q.strip())
                    elapsed = time.time() - t0

                    st.caption(f"Route: `multi_form_semantic` (forced) · {elapsed:.1f}s")
                    st.info(result.answer)
                    if result.cited_form_ids:
                        st.caption(f"📎 Sources: {', '.join(result.cited_form_ids)}")
                    if not result.grounded:
                        st.warning("⚠️ Answer could not be fully grounded.")
                except Exception as e:
                    st.error(f"❌ {e}")
