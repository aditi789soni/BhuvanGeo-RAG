"""
RAG Question Answering Engine for BhuvanGeo-RAG.
Orchestrates semantic retrieval, anti-hallucination verification,
honest extractive fallback, and strict ISRO source citations.
"""

import os
import re
import logging
from typing import Dict, Any, List, Optional
from src.vector_store import VectorStore
from src.config import TOP_K_RETRIEVAL
from src.geospatial import GeospatialSensorCatalog

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Out-of-domain missions & entities not present in Indian Earth Observation documentation
UNSUPPORTED_DOMAINS = [
    "chandrayaan", "pragyan", "vikram lander", "james webb", "jwst",
    "curiosity", "perseverance", "mars rover", "hubble", "voyager",
    "artemis", "apollo", "international space station", "iss"
]


class RAGEngine:
    """
    RAG Pipeline specifically designed for ISRO / NRSC / Bhuvan technical documentation.
    Strictly forbids inventing specifications and requires citations for all factual claims.
    """

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()
        # Check for optional API keys
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        document_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve most relevant document chunks from the vector store."""
        return self.vector_store.query(
            query_text=query, top_k=top_k, document_filter=document_filter
        )

    @staticmethod
    def _extract_key_sentences(query: str, text: str, max_sentences: int = 3) -> str:
        """
        Extract sentences from the text chunk that have the highest lexical & semantic overlap with the query.
        """
        # Split text into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if not sentences:
            return text[:200]

        query_terms = set(re.findall(r"\w+", query.lower()))
        # Remove common stopwords
        stopwords = {
            "what", "is", "the", "of", "and", "in", "to", "for", "a", "an",
            "are", "which", "satellite", "sensor", "how", "many", "does", "have"
        }
        filtered_terms = query_terms - stopwords
        if not filtered_terms:
            filtered_terms = query_terms

        scored_sentences = []
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) < 15:
                continue
            s_words = set(re.findall(r"\w+", s_clean.lower()))
            overlap = len(filtered_terms.intersection(s_words))
            scored_sentences.append((overlap, s_clean))

        # Sort by overlap descending
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_matches = [s for score, s in scored_sentences[:max_sentences] if score > 0]

        if top_matches:
            return " ".join(top_matches)
        return sentences[0] if sentences else text[:200]

    def _generate_extractive_answer(
        self, query: str, retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Free, honest, locally computed extractive and synthesis QA fallback.
        Assembles facts directly from verified retrieved text chunks without external API dependencies.
        Applies strict entity overlap and similarity guardrails to prevent hallucination.
        """
        if not retrieved_chunks:
            return {
                "answer": (
                    "Insufficient evidence in the ingested official ISRO/NRSC documentation. "
                    "No relevant document sections were found matching your query."
                ),
                "is_grounded": False,
                "confidence": 0.0,
                "citations": [],
            }

        best_chunk = retrieved_chunks[0]

        # Extract distinct query terms (excluding generic remote-sensing and engineering stopwords)
        query_words = re.findall(r"[A-Za-z0-9\-]+", query.lower())
        generic_terms = {
            "what", "is", "the", "of", "and", "in", "to", "for", "a", "an", "are",
            "which", "how", "many", "does", "have", "with", "from", "its", "type",
            "sensor", "satellite", "payload", "system", "data", "image", "imaging",
            "space", "mission", "primary", "compared", "across", "about", "specification",
            "resolution", "swath", "bands", "revisit", "spatial", "mass", "weight",
            "capacity", "battery", "power", "speed", "instruments", "mirror", "rover",
            "best", "suited", "characteristics", "period", "achieve", "along-track",
            "revisit", "radiometric", "bits", "depth"
        }
        specific_query_terms = [w for w in query_words if len(w) >= 3 and w not in generic_terms]

        # Check entity grounding: at least one specific domain term must appear in retrieved text
        combined_chunk_text = " ".join([c["text"].lower() for c in retrieved_chunks])
        has_lexical_grounding = False
        if not specific_query_terms:
            has_lexical_grounding = True
        else:
            for term in specific_query_terms:
                if term in combined_chunk_text:
                    has_lexical_grounding = True
                    break

        # Quality check: reject if below similarity threshold OR if distinctive query entity is absent
        if best_chunk["similarity_score"] < 0.35 or not has_lexical_grounding:
            return {
                "answer": (
                    "Insufficient evidence in the ingested official ISRO/NRSC documentation to verify this specification. "
                    f"The ingested documents do not contain direct verified technical records answering '{query}'. "
                    "BhuvanGeo-RAG strictly prevents speculation."
                ),
                "is_grounded": False,
                "confidence": best_chunk["similarity_score"],
                "citations": [
                    {
                        "document": best_chunk["document_name"],
                        "page": best_chunk["page_number"],
                        "excerpt": best_chunk["text"][:250] + "...",
                        "similarity": best_chunk["similarity_score"],
                    }
                ],
            }

        # Build grounded synthesis
        citations = []
        extracted_facts = []

        for i, chunk in enumerate(retrieved_chunks):
            if chunk["similarity_score"] >= 0.35 or i == 0:
                key_sentence = self._extract_key_sentences(query, chunk["text"])
                excerpt = chunk["text"][:300].strip()
                if len(chunk["text"]) > 300:
                    excerpt += "..."

                citations.append(
                    {
                        "document": chunk["document_name"],
                        "page": chunk["page_number"],
                        "excerpt": excerpt,
                        "similarity": chunk["similarity_score"],
                        "source_url": chunk.get("source_url", ""),
                    }
                )

                doc_label = f"[{chunk['document_name']}, p.{chunk['page_number']}]"
                extracted_facts.append(f"• From {doc_label}: {key_sentence}")

        synthesis = (
            f"Based on the official ingested ISRO/NRSC documentation, here are the verified details regarding your query:\n\n"
            + "\n\n".join(extracted_facts)
        )

        return {
            "answer": synthesis,
            "is_grounded": True,
            "confidence": best_chunk["similarity_score"],
            "citations": citations,
        }

    def _generate_llm_answer(
        self, query: str, retrieved_chunks: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Optional generative model pipeline if an API key is available.
        Uses temperature 0.0 and strict system instructions to never hallucinate.
        """
        if not self.gemini_api_key:
            return None

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.gemini_api_key)

            context_blocks = []
            citations = []
            for i, c in enumerate(retrieved_chunks):
                context_blocks.append(
                    f"[Source {i+1}: {c['document_name']}, Page {c['page_number']}]\n{c['text']}"
                )
                citations.append(
                    {
                        "document": c["document_name"],
                        "page": c["page_number"],
                        "excerpt": c["text"][:250] + "...",
                        "similarity": c["similarity_score"],
                    }
                )

            context_str = "\n\n---\n\n".join(context_blocks)
            prompt = (
                f"You are the senior geospatial intelligence assistant for ISRO/NRSC/Bhuvan.\n"
                f"Answer the user's question STRICTLY and ONLY using the provided document context below.\n"
                f"RULES:\n"
                f"1. Do NOT invent satellite specifications, bands, resolutions, swaths, or dates.\n"
                f"2. If the context does not contain sufficient facts to answer the question, state: "
                f"'Insufficient evidence in the ingested official ISRO/NRSC documentation to verify this specification.'\n"
                f"3. Explicitly cite the source document name and page number for each claim.\n\n"
                f"CONTEXT:\n{context_str}\n\n"
                f"QUESTION: {query}\n\n"
                f"ANSWER:"
            )

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=800,
                ),
            )

            if response and response.text:
                return {
                    "answer": response.text.strip(),
                    "is_grounded": True,
                    "confidence": retrieved_chunks[0]["similarity_score"] if retrieved_chunks else 0.0,
                    "citations": citations,
                    "model_used": "gemini-2.5-flash (Grounding Guardrail)",
                }
        except Exception as e:
            logger.warning(f"LLM generation failed or unavailable: {e}. Falling back to extractive QA.")
            return None

    def answer_question(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        document_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point for question answering.
        Retrieves context, verifies evidence threshold, and produces an answer with citations.
        """
        query_lower = query.lower()

        # Anti-Hallucination Guardrail 1: Check for explicitly out-of-domain space probes
        for domain in UNSUPPORTED_DOMAINS:
            if re.search(rf"\b{re.escape(domain)}\b", query_lower):
                return {
                    "query": query,
                    "answer": (
                        "Insufficient evidence in the ingested official ISRO/NRSC documentation. "
                        f"The query refers to '{domain}', which is outside the scope of the ingested "
                        "Indian Earth Observation satellite documentation (Resourcesat, Cartosat, Oceansat, Bhuvan). "
                        "BhuvanGeo-RAG strictly prevents speculating on unverified mission specifications."
                    ),
                    "is_grounded": False,
                    "confidence": 0.0,
                    "citations": [],
                    "model_used": "Anti-Hallucination Domain Guardrail",
                    "chunks_retrieved_count": 0,
                    "retrieved_chunks": [],
                }

        # Geospatial Intelligence Guardrail 2: Application-guided sensor recommendations
        is_recommendation_query = any(
            phrase in query_lower
            for phrase in ["best suited", "which sensor", "recommend", "suitable sensor", "choice of sensor"]
        )
        if is_recommendation_query:
            rec = GeospatialSensorCatalog.recommend_sensor(query)
            primary_specs = rec["verified_sensor_specs"]
            rec_answer = (
                f"### Operational Geospatial Recommendation\n\n"
                f"• **Recommended Sensor:** **{rec['recommended_sensor']}**\n"
                f"• **Operational Rationale:** {rec['recommendation_rationale']}\n"
                f"• **Alternative Option:** {rec['alternative_sensor']}\n\n"
                f"**Verified Technical Specifications:**\n"
                f"- **Spatial Resolution:** {primary_specs['spatial_resolution_m']} m\n"
                f"- **Swath Width:** {primary_specs['swath_km']}\n"
                f"- **Radiometric Resolution:** {primary_specs['radiometric_resolution_bits']} bits\n"
                f"- **Revisit Cycle:** {primary_specs['revisit_days']} days\n"
                f"- **Operational Missions:** {', '.join(primary_specs['missions'])}\n\n"
                f"> **Verification Note:** {rec['disclaimer']}"
            )
            # Retrieve supporting chunks for citation
            retrieved_chunks = self.retrieve(query=rec["recommended_sensor"], top_k=2)
            citations = [
                {
                    "document": c["document_name"],
                    "page": c["page_number"],
                    "excerpt": c["text"][:250] + "...",
                    "similarity": c["similarity_score"],
                    "source_url": c.get("source_url", ""),
                }
                for c in retrieved_chunks
            ]
            return {
                "query": query,
                "answer": rec_answer,
                "is_grounded": True,
                "confidence": 0.95,
                "citations": citations,
                "model_used": "ISRO Geospatial Recommendation Engine",
                "chunks_retrieved_count": len(retrieved_chunks),
                "retrieved_chunks": retrieved_chunks,
            }

        retrieved_chunks = self.retrieve(
            query=query, top_k=top_k, document_filter=document_filter
        )

        # Try LLM if configured; otherwise use extractive synthesis fallback
        result = self._generate_llm_answer(query, retrieved_chunks)
        if not result:
            result = self._generate_extractive_answer(query, retrieved_chunks)
            result["model_used"] = "Local Extractive & Semantic Synthesis Engine"

        result["query"] = query
        result["chunks_retrieved_count"] = len(retrieved_chunks)
        result["retrieved_chunks"] = retrieved_chunks

        return result

