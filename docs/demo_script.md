# BhuvanGeo-RAG: Live Demonstration Script

This demo script guides an evaluator, interviewer, or portfolio reviewer through a complete 5-minute interactive walkthrough of **BhuvanGeo-RAG**.

---

## Act 1: Launch & Architectural Briefing (1 Minute)

1. **Start the Application:**
   ```bash
   .\venv\Scripts\streamlit run app\app.py
   ```
2. **Reviewer Note:** Highlight that BhuvanGeo-RAG runs **100% locally on CPU** using open-source components (`PyMuPDF`, `ChromaDB`, `sentence-transformers`) with zero dependency on paid cloud APIs.
3. Observe the sidebar:
   - Collection size & total chunks indexed.
   - Verified ISRO compliance banner.

---

## Act 2: Technical Precision Querying (1.5 Minutes)

In **Tab 1: Mission Intelligence Chat**, execute these queries:

### Query A: Payload Technical Specification
> **Prompt:** *"What are the spatial resolution and swath width of LISS-IV on Resourcesat-2?"*
- **Observe:**
  - Fast response (< 1.5s).
  - Identifies **5.8 meters** spatial resolution and **70 km swath** (in multispectral mode).
  - Open the **📌 Verified Sources & Excerpts** dropdown:
    - Points directly to `ISRO_Resourcesat-2_Brochure.pdf` with the exact page number and text quote.

### Query B: Stereoscopic Cartography
> **Prompt:** *"How does Cartosat-1 generate Digital Elevation Models (CartoDEM)?"*
- **Observe:**
  - Explains the two along-track panchromatic cameras: **Fore camera (+26°)** and **Aft camera (-5°)** producing stereo pairs with 2.5m resolution.
  - Cites `ISRO_Cartosat-1_Brochure.pdf`.

---

## Act 3: Anti-Hallucination & Guardrail Verification (1 Minute)

Ask an out-of-domain or fabricated query to demonstrate strict anti-hallucination guardrails:

> **Prompt:** *"What is the engine thrust and fuel tank volume of the Chandrayaan-3 lander?"*
- **Observe:**
  - Instead of guessing or hallucinating facts, the assistant cleanly and honestly states:
    > *"Insufficient evidence in the ingested official ISRO/NRSC documentation to verify this specification."*
  - This demonstrates production-grade safety for mission-critical engineering applications.

---

## Act 4: Geospatial Sensor Comparison Matrix (1 Minute)

Switch to **Tab 2: Sensor Comparison & Specs**:
1. Select **LISS-IV, LISS-III, AWiFS, and Cartosat-1 PAN** in the multiselect box.
2. Review the structured matrix:
   - Spatial resolution gradient from 0.8m / 2.5m to 56m.
   - Swath width comparison: 9.6km to 740km.
   - Spectral band differences (Green, Red, NIR, SWIR).
3. Test the **Application-Guided Sensor Recommendation**:
   - Select *"Cadastral and parcel-level land records mapping (Village scale)"*.
   - Click **Generate Recommendation**.
   - Note the recommendation of **LISS-IV** (≤ 5.8m) with the operational rationale and the visual separation between verified facts and recommendations.

---

## Act 5: Dynamic Document Ingestion (30 Seconds)

1. In the sidebar, drag and drop any official PDF (e.g. Cartosat-3 or an ISRO technical bulletin).
2. Click **Process & Ingest Uploaded PDF**.
3. Watch the system extract pages via PyMuPDF, chunk, generate dense embeddings, and update the ChromaDB vector count without restarting the server!

