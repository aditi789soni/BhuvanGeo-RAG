"""
Data Collection Script for BhuvanGeo-RAG.
Downloads legitimate, publicly accessible official ISRO / NRSC / Bhuvan documents.
Records document names, source URLs, and metadata into documents_manifest.json.
"""

import json
import logging
from pathlib import Path
import requests
import urllib3
from src.config import DOCUMENTS_DIR

urllib3.disable_warnings()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Verified authentic official ISRO & NRSC / Bhuvan publications
DOCUMENT_SOURCES = [
    {
        "id": "resourcesat_2_handbook",
        "title": "NRSC / ISRO Resourcesat-2 Data Users' Handbook (Doc No: NRSC:SDAPSA:NDC:DEC11-364)",
        "filename": "NRSC_ISRO_Resourcesat-2_Data_Users_Handbook.pdf",
        "primary_url": "https://earth.esa.int/eogateway/documents/20142/37627/ResourceSat-2-Data-User-Handbook.pdf",
        "category": "Resourcesat",
        "description": "Comprehensive 142-page official technical handbook compiled by NRSC / ISRO detailing LISS-IV, LISS-III, and AWiFS sensor specifications, calibration, and data products.",
    },
    {
        "id": "cartosat1_dem_bhuvan",
        "title": "NRSC / Bhuvan Cartosat-1 Stereo Data & CartoDEM Generation Technical Guide",
        "filename": "NRSC_Bhuvan_Cartosat1_DEM_Generation.pdf",
        "primary_url": "https://bhuvan-app3.nrsc.gov.in/data/download/tools/document/SISDP-DEM-GENERATION-BHUVAN.pdf",
        "category": "Cartosat",
        "description": "Official NRSC / Bhuvan technical guide detailing Cartosat-1 Fore (+26°) and Aft (-5°) stereo panchromatic cameras, along-track stereoscopy, and high-precision CartoDEM extraction.",
    },
    {
        "id": "oceansat2_oscat_bhuvan",
        "title": "NRSC / Bhuvan Oceansat-2 Scatterometer (OSCAT) Global Wind Fields Technical Report",
        "filename": "NRSC_Bhuvan_Oceansat2_OSCAT_Winds.pdf",
        "primary_url": "https://bhuvan-app3.nrsc.gov.in/data/download/tools/document/Oscat_Global_composites_Bhuvan_winds.pdf",
        "category": "Oceansat",
        "description": "Official NRSC documentation detailing Oceansat-2 OSCAT Ku-band radar scatterometer observations, daily composite wind vectors, OCM synergy, and oceanic meteorology.",
    },
    {
        "id": "cartosat_2_series_c40",
        "title": "ISRO Cartosat-2 Series Mission Brochure (PSLV-C40)",
        "filename": "ISRO_Cartosat-2_Series_PSLV-C40_Brochure.pdf",
        "primary_url": "https://www.isro.gov.in/media_isro/pdf/Publications/Brochures/PSLVC40Cartosat2SeriesMission.pdf",
        "category": "Cartosat",
        "description": "Official ISRO publication on Cartosat-2 series spacecraft, high-resolution Panchromatic (PAN) and Multispectral (MX) imaging, and cartography applications.",
    },
    {
        "id": "cartosat_2_series_c37",
        "title": "ISRO Cartosat-2 Series Satellite Brochure (PSLV-C37)",
        "filename": "ISRO_Cartosat-2_Series_PSLV-C37_Brochure.pdf",
        "primary_url": "https://www.isro.gov.in/media_isro/pdf/Publications/Brochures/PSLV-C37.pdf",
        "category": "Cartosat",
        "description": "Official ISRO publication on Cartosat-2 series satellite, agile roll/pitch steering, urban mapping, and land information systems.",
    },
]


def download_document(doc_info: dict, output_dir: Path) -> bool:
    """Download a single document from official ISRO portal."""
    target_path = output_dir / doc_info["filename"]
    if target_path.exists() and target_path.stat().st_size > 10000:
        logger.info(f"'{doc_info['filename']}' already exists ({target_path.stat().st_size} bytes). Skipping.")
        return True

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,*/*",
    }

    url = doc_info["primary_url"]
    try:
        logger.info(f"Downloading {doc_info['filename']} from: {url}")
        response = requests.get(url, headers=headers, timeout=30, verify=False)
        if response.status_code == 200 and len(response.content) > 5000:
            with open(target_path, "wb") as f:
                f.write(response.content)
            logger.info(f"Saved {doc_info['filename']} ({len(response.content):,} bytes).")
            doc_info["downloaded_from"] = url
            return True
        else:
            logger.warning(f"Download failed with status {response.status_code} ({len(response.content)} bytes)")
    except Exception as e:
        logger.warning(f"Failed to fetch from {url}: {e}")

    return False


def collect_isro_documents() -> dict:
    """Fetch all official documents and generate the manifest."""
    output_dir = Path(DOCUMENTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "collection_title": "ISRO & NRSC Official Mission Documentation",
        "total_documents": len(DOCUMENT_SOURCES),
        "documents": [],
    }

    downloaded_count = 0
    for doc_info in DOCUMENT_SOURCES:
        success = download_document(doc_info, output_dir)
        if success:
            downloaded_count += 1
        doc_entry = {
            "id": doc_info["id"],
            "title": doc_info["title"],
            "filename": doc_info["filename"],
            "category": doc_info["category"],
            "description": doc_info["description"],
            "primary_url": doc_info["primary_url"],
            "downloaded": success,
            "path": str(output_dir / doc_info["filename"]),
        }
        manifest["documents"].append(doc_entry)

    manifest_path = output_dir / "documents_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Downloaded {downloaded_count}/{len(DOCUMENT_SOURCES)} official documents.")
    logger.info(f"Manifest written to: {manifest_path}")
    return manifest


if __name__ == "__main__":
    collect_isro_documents()
