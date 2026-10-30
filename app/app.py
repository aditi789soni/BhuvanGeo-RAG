"""
BhuvanGeo-RAG: Streamlit Geospatial Document & Mission Intelligence Application.
Provides an interactive chat interface with verified source citations,
geospatial sensor comparison matrix, official document library, and dynamic PDF ingestion.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from src.config import (
    DOCUMENTS_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    TOP_K_RETRIEVAL,
)
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine
from src.geospatial import GeospatialSensorCatalog
from src.pdf_processor import PDFProcessor

# Streamlit Page Setup
st.set_page_config(
    page_title="BhuvanGeo-RAG | ISRO Mission Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, responsive UI
st.markdown(
    """
    <style>
    /* Theme Header */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f2b48;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #3b5266;
        margin-bottom: 1.5rem;
    }
    .badge-isro {
        background: linear-gradient(135deg, #ff9933 0%, #138808 100%);
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .citation-card {
        background-color: #f8fafc;
        border-left: 4px solid #2563eb;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 0 8px 8px 0;
        font-size: 0.92rem;
    }
    .citation-header {
        font-weight: 600;
        color: #1e3a8a;
        margin-bottom: 4px;
    }
    .citation-excerpt {
        color: #334155;
        font-style: italic;
        background: #ffffff;
        padding: 6px 10px;
        border-radius: 4px;
        border: 1px dashed #cbd5e1;
    }
    .metric-box {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .verified-pill {
        background-color: #dcfce7;
        color: #166534;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.78rem;
    }
    .recommendation-pill {
        background-color: #fef3c7;
        color: #92400e;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.78rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_rag_pipeline():
    """Cache the VectorStore and RAGEngine instances for fast inference."""
    try:
        vs = VectorStore()
        rag = RAGEngine(vector_store=vs)
        return vs, rag
    except Exception as e:
        st.error(f"Failed to load RAG engine: {e}")
        return None, None


vector_store, rag_engine = load_rag_pipeline()

# Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Namaste! I am **BhuvanGeo-RAG**, your mission intelligence assistant for "
                "ISRO Earth Observation satellites and the Bhuvan geoportal.\n\n"
                "Ask me any verified technical question regarding **Resourcesat-2/2A (LISS-IV, LISS-III, AWiFS)**, "
                "**Cartosat missions (Stereo PAN, DEMs)**, **Oceansat-2/3 (OCM, OSCAT)**, or sensor suitability."
            ),
            "citations": [],
            "grounded": True,
        }
    ]

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Indian_Space_Research_Organisation_Logo.svg/200px-Indian_Space_Research_Organisation_Logo.svg.png",
        width=80,
    )
    st.markdown("### 🛰️ BhuvanGeo-RAG")
    st.caption("National Remote Sensing Centre (NRSC) / ISRO")

    st.markdown("---")
    st.subheader("📊 Knowledge Base Status")
    if vector_store:
        stats = vector_store.get_document_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Chunks", stats["total_chunks"])
        with col2:
            st.metric("Documents", len(stats["documents"]))

        st.caption(f"**Embeddings:** `{EMBEDDING_MODEL_NAME.split('/')[-1]}`")
        st.caption(f"**Chroma Collection:** `{CHROMA_COLLECTION_NAME}`")
    else:
        st.warning("Vector store is not initialized.")

    st.markdown("---")
    st.subheader("📄 Upload & Ingest PDF")
    uploaded_file = st.file_uploader(
        "Add official satellite document", type=["pdf"], help="Upload an ISRO/NRSC PDF manual"
    )
    if uploaded_file is not None:
        if st.button("Process & Ingest Uploaded PDF", use_container_width=True):
            with st.spinner("Extracting and embedding chunks..."):
                target_path = DOCUMENTS_DIR / uploaded_file.name
                with open(target_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                processor = PDFProcessor()
                chunks = processor.process_document(
                    pdf_path=target_path,
                    source_url="Local Upload",
                    extra_metadata={"category": "User Uploaded"},
                )
                added = vector_store.add_chunks(chunks)
                st.success(f"Ingested {added} new chunks from '{uploaded_file.name}'!")
                st.rerun()

    st.markdown("---")
    st.markdown(
        """
        **Strict Anti-Hallucination Guardrails:**
        - ✅ Grounded in official ISRO/NRSC literature
        - ✅ Zero invented specifications
        - ✅ Mandatory source citations & excerpts
        - ✅ Fully local, open-source stack
        """
    )


# ----------------- MAIN INTERFACE -----------------
st.markdown('<div class="badge-isro">NRSC / ISRO GEOSPATIAL INTELLIGENCE</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-title">BhuvanGeo-RAG: Mission & Geospatial Document Assistant</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Semantic document retrieval, technical payload analysis, and satellite sensor intelligence powered by local embeddings.</div>',
    unsafe_allow_html=True,
)

tab_chat, tab_sensors, tab_library, tab_system = st.tabs(
    [
        "💬 Mission Intelligence Chat",
        "🔍 Sensor Comparison & Specs",
        "📚 Ingested Document Library",
        "⚙️ Architecture & Diagnostics",
    ]
)

# ==================== TAB 1: CHAT ====================
with tab_chat:
    col_chat, col_info = st.columns([7, 3])

    with col_info:
        st.markdown("#### 💡 Sample Queries")
        sample_queries = [
            "What are the spatial resolution and swath of LISS-IV on Resourcesat-2?",
            "What spectral bands are supported by AWiFS and what is its repeat cycle?",
            "How does Cartosat-1 generate Digital Elevation Models (CartoDEM)?",
            "What is the swath width and purpose of OCM on Oceansat-2?",
            "Which sensor is best suited for village cadastral boundary mapping?",
            "What is the orbital altitude and inclination of Chandrayaan-3?",  # Unsupported sample
        ]

        for sq in sample_queries:
            if st.button(sq, key=f"btn_{hash(sq)}", use_container_width=True):
                st.session_state.active_query = sq

    with col_chat:
        # Display chat conversation
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("citations"):
                        with st.expander(f"📌 Verified Sources & Excerpts ({len(msg['citations'])})"):
                            for cit in msg["citations"]:
                                st.markdown(
                                    f"""
                                    <div class="citation-card">
                                        <div class="citation-header">📄 {cit['document']} — Page {cit['page']} (Similarity: {cit.get('similarity', 0.0):.2f})</div>
                                        <div class="citation-excerpt">"{cit['excerpt']}"</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

        # Input query handling
        user_input = st.chat_input("Ask about Resourcesat, Cartosat, Oceansat, or Bhuvan sensors...")
        if "active_query" in st.session_state:
            user_input = st.session_state.pop("active_query")

        if user_input:
            # Append user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Generate Assistant Answer
            with st.chat_message("assistant"):
                with st.spinner("Retrieving verified ISRO document chunks..."):
                    start_time = time.time()
                    response = rag_engine.answer_question(query=user_input, top_k=TOP_K_RETRIEVAL)
                    elapsed = time.time() - start_time

                st.markdown(response["answer"])

                if response["citations"]:
                    with st.expander(f"📌 Verified Sources & Excerpts ({len(response['citations'])})"):
                        for cit in response["citations"]:
                            st.markdown(
                                f"""
                                <div class="citation-card">
                                    <div class="citation-header">📄 {cit['document']} — Page {cit['page']} (Similarity: {cit.get('similarity', 0.0):.2f})</div>
                                    <div class="citation-excerpt">"{cit['excerpt']}"</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                st.caption(
                    f"⏱️ Retrieval & answering latency: {elapsed:.2f}s | "
                    f"Model: `{response.get('model_used', 'Local RAG')}` | "
                    f"Evidence Grounded: `{response.get('is_grounded', False)}`"
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response["answer"],
                        "citations": response["citations"],
                        "grounded": response.get("is_grounded", False),
                    }
                )


# ==================== TAB 2: SENSORS ====================
with tab_sensors:
    st.subheader("🛰️ ISRO Earth Observation Sensor Catalog")
    st.markdown(
        "Compare official technical parameters for operational Indian optical and radar payloads."
    )

    all_sensor_keys = list(GeospatialSensorCatalog.get_all_sensors().keys())
    selected_sensors = st.multiselect(
        "Select Sensors to Compare",
        options=all_sensor_keys,
        default=["LISS-IV", "LISS-III", "AWiFS", "Cartosat-1 PAN"],
    )

    if selected_sensors:
        comparison_data = GeospatialSensorCatalog.compare_sensors(selected_sensors)
        df_comp = pd.DataFrame(comparison_data)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🎯 Application-Guided Sensor Recommendation")
    st.markdown(
        "Select your geospatial workflow to receive an operational sensor recommendation with verified technical specifications."
    )

    app_options = [
        "Cadastral and parcel-level land records mapping (Village scale)",
        "National crop acreage estimation & drought vegetation monitoring (FASAL)",
        "Digital Elevation Model (DEM) and stereoscopic terrain modeling (CartoDEM)",
        "Ocean chlorophyll concentration & Potential Fishing Zone (PFZ) advisory",
        "Cyclone tracking and ocean surface wind vector retrieval",
        "District level Land Use / Land Cover (LULC) and forest monitoring",
    ]

    selected_app = st.selectbox("Select Application Domain", options=app_options)
    if st.button("Generate Recommendation", type="primary"):
        rec = GeospatialSensorCatalog.recommend_sensor(selected_app)

        c1, c2 = st.columns([6, 4])
        with c1:
            st.markdown(f"### Recommended: `{rec['recommended_sensor']}`")
            st.markdown(f"**Rationale:** {rec['recommendation_rationale']}")
            st.markdown(
                f'<span class="recommendation-pill">Operational Guidance</span> '
                f'<span class="verified-pill">Verified ISRO Specs</span>',
                unsafe_allow_html=True,
            )
            st.info(rec["disclaimer"])

        with c2:
            st.markdown("#### Primary Sensor Specs")
            p_specs = rec["verified_sensor_specs"]
            st.write(f"- **Resolution:** {p_specs['spatial_resolution_m']} m")
            st.write(f"- **Swath:** {p_specs['swath_km']}")
            st.write(f"- **Revisit:** {p_specs['revisit_days']} days")
            st.write(f"- **Radiometry:** {p_specs['radiometric_resolution_bits']} bits")
            st.write(f"- **Missions:** {', '.join(p_specs['missions'][:2])}")


# ==================== TAB 3: DOCUMENT LIBRARY ====================
with tab_library:
    st.subheader("📚 Ingested ISRO & NRSC Documentation Library")
    st.markdown(
        "Official technical brochures and user handbooks ingested into the local vector database."
    )

    manifest_file = DOCUMENTS_DIR / "documents_manifest.json"
    if manifest_file.exists():
        import json

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        docs = manifest_data.get("documents", [])
        if docs:
            doc_rows = []
            for d in docs:
                p = DOCUMENTS_DIR / d["filename"]
                size_kb = round(p.stat().st_size / 1024, 1) if p.exists() else 0.0
                doc_rows.append(
                    {
                        "Document Title": d["title"],
                        "Filename": d["filename"],
                        "Category": d["category"],
                        "File Size (KB)": size_kb,
                        "Official URL": d["primary_url"],
                    }
                )
            st.dataframe(pd.DataFrame(doc_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Manifest file not found. Documents will appear after initial ingestion.")


# ==================== TAB 4: ARCHITECTURE ====================
with tab_system:
    st.subheader("⚙️ System Architecture & Engineering Diagnostics")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            ### Technical Stack
            - **PDF Extraction:** `PyMuPDF` (High-speed C-native text parsing & coordinate tracking)
            - **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors, CPU-optimized)
            - **Vector Database:** `ChromaDB` (Persistent, serverless, local SQLite + HNSW index)
            - **RAG Architecture:** Dense semantic retrieval + grounded extractive synthesis fallback
            - **Interface:** `Streamlit` (Interactive stateful web app)
            - **Zero Paid APIs:** Completely runnable offline on local CPU hardware
            """
        )

    with col_b:
        st.markdown(
            """
            ### Anti-Hallucination Guardrails
            1. **Similarity Confidence Filter:** Queries yielding top distance below threshold are rejected with explicit insufficiency notice.
            2. **Mandatory Excerpt Attribution:** Every factual claim cites document name, page number, and literal excerpt.
            3. **Verified Sensor Catalog:** Hardened catalog of official ISRO parameters with strict verification flags.
            4. **Idempotent Ingestion:** Deterministic chunk hashing prevents duplicate embeddings on re-runs.
            """
        )

