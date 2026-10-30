"""
Geospatial Intelligence Module for BhuvanGeo-RAG.
Provides verified technical specifications for ISRO Earth Observation satellite sensors:
LISS-III, LISS-IV, AWiFS, Cartosat PAN, OCM-2/3, Scatterometer (OSCAT).
Enables parametric comparisons and application-guided sensor recommendations
with strict verification boundaries.
"""

from typing import Dict, List, Any, Optional


# Verified sensor specifications strictly derived from official ISRO/NRSC documentation
VERIFIED_SENSORS: Dict[str, Dict[str, Any]] = {
    "LISS-IV": {
        "sensor_name": "LISS-IV (Linear Imaging Self-Scanning Sensor-4)",
        "short_name": "LISS-IV",
        "missions": ["Resourcesat-1 (IRS-P6)", "Resourcesat-2", "Resourcesat-2A"],
        "spatial_resolution_m": 5.8,
        "swath_km": "70 km (Resourcesat-2/2A in multispectral mode); 23 km (Mono mode)",
        "swath_numeric_km": 70.0,
        "spectral_bands": [
            {"band": "B2 (Green)", "wavelength_um": "0.52 - 0.59"},
            {"band": "B3 (Red)", "wavelength_um": "0.62 - 0.68"},
            {"band": "B4 (NIR)", "wavelength_um": "0.77 - 0.86"},
        ],
        "radiometric_resolution_bits": 10,
        "revisit_days": 24,  # Off-nadir steering provides 5 days
        "steerability": "±26° across-track",
        "applications": [
            "Cadastral and parcel-level mapping",
            "Precision agriculture and crop condition assessment",
            "Urban infrastructure planning",
            "Forest canopy patch analysis",
            "Micro-watershed development",
        ],
        "verified_source": "ISRO Resourcesat-2 Mission Brochure & User Handbook",
    },
    "LISS-III": {
        "sensor_name": "LISS-III (Linear Imaging Self-Scanning Sensor-3)",
        "short_name": "LISS-III",
        "missions": ["IRS-1C", "IRS-1D", "Resourcesat-1", "Resourcesat-2", "Resourcesat-2A"],
        "spatial_resolution_m": 23.5,
        "swath_km": "141 km (Resourcesat-2); 140 km (Resourcesat-1)",
        "swath_numeric_km": 141.0,
        "spectral_bands": [
            {"band": "B2 (Green)", "wavelength_um": "0.52 - 0.59"},
            {"band": "B3 (Red)", "wavelength_um": "0.62 - 0.68"},
            {"band": "B4 (NIR)", "wavelength_um": "0.77 - 0.86"},
            {"band": "B5 (SWIR)", "wavelength_um": "1.55 - 1.70"},
        ],
        "radiometric_resolution_bits": 10,  # 7-bit in IRS-1C/1D, 10-bit in Resourcesat-2
        "revisit_days": 24,
        "steerability": "None (Nadir viewing)",
        "applications": [
            "National and district level land use / land cover (LULC)",
            "Forest cover density mapping",
            "Snow cover and glacier monitoring",
            "Soil salinity and waterlogging mapping",
            "Geological and mineral lineament mapping",
        ],
        "verified_source": "ISRO Resourcesat-2 Data User's Handbook",
    },
    "AWiFS": {
        "sensor_name": "AWiFS (Advanced Wide Field Sensor)",
        "short_name": "AWiFS",
        "missions": ["Resourcesat-1", "Resourcesat-2", "Resourcesat-2A"],
        "spatial_resolution_m": 56.0,
        "swath_km": "740 km (Combined across two camera heads AWiFS-A and AWiFS-B)",
        "swath_numeric_km": 740.0,
        "spectral_bands": [
            {"band": "B2 (Green)", "wavelength_um": "0.52 - 0.59"},
            {"band": "B3 (Red)", "wavelength_um": "0.62 - 0.68"},
            {"band": "B4 (NIR)", "wavelength_um": "0.77 - 0.86"},
            {"band": "B5 (SWIR)", "wavelength_um": "1.55 - 1.70"},
        ],
        "radiometric_resolution_bits": 12,
        "revisit_days": 5,
        "steerability": "Fixed wide field (two camera heads tilted ±11.9°)",
        "applications": [
            "National-scale crop acreage and production estimation (FASAL)",
            "Drought vulnerability assessment and vegetation vigor (NDVI time composites)",
            "Rapid disaster monitoring (floods, cyclones, forest fires)",
            "Snow melt runoff modeling",
            "National biomass assessment",
        ],
        "verified_source": "ISRO Resourcesat-2 Payloads Specification",
    },
    "Cartosat-1 PAN": {
        "sensor_name": "Cartosat-1 PAN (Stereo Panchromatic Cameras: Fore & Aft)",
        "short_name": "Cartosat-1 PAN",
        "missions": ["Cartosat-1 (IRS-P5)"],
        "spatial_resolution_m": 2.5,
        "swath_km": "30 km (Fore camera +26°, Aft camera -5° along-track)",
        "swath_numeric_km": 30.0,
        "spectral_bands": [
            {"band": "PAN (Panchromatic)", "wavelength_um": "0.50 - 0.85"},
        ],
        "radiometric_resolution_bits": 10,
        "revisit_days": 5,  # with pitch/roll steering, 126 days nadir
        "steerability": "Fore (+26°) & Aft (-5°) stereoscopic configuration",
        "applications": [
            "High-accuracy Digital Elevation Models (CartoDEM on Bhuvan)",
            "Topographic and contour mapping (1:25,000 to 1:10,000 scale)",
            "Major corridor and infrastructure alignment (highways, rail)",
            "Urban 3D terrain modeling",
        ],
        "verified_source": "ISRO Cartosat-1 Mission Publication & Bhuvan Documentation",
    },
    "Cartosat-2 PAN/MX": {
        "sensor_name": "Cartosat-2 Series Panchromatic & Multispectral HR Imagers",
        "short_name": "Cartosat-2 PAN/MX",
        "missions": ["Cartosat-2", "Cartosat-2A", "Cartosat-2B", "Cartosat-2C/2D/2E/2F"],
        "spatial_resolution_m": 0.8,  # PAN: 0.8 m - 0.6 m; MX: 2.0 m
        "swath_km": "9.6 km (nominal nadir) to ~10 km",
        "swath_numeric_km": 9.6,
        "spectral_bands": [
            {"band": "PAN", "wavelength_um": "0.45 - 0.90"},
            {"band": "Blue", "wavelength_um": "0.45 - 0.52"},
            {"band": "Green", "wavelength_um": "0.52 - 0.59"},
            {"band": "Red", "wavelength_um": "0.62 - 0.68"},
            {"band": "NIR", "wavelength_um": "0.77 - 0.86"},
        ],
        "radiometric_resolution_bits": 10,
        "revisit_days": 4,  # with agile roll/pitch steering (up to ±45°)
        "steerability": "Agile spacecraft steering up to ±45° across and along track",
        "applications": [
            "Large-scale cadastral and municipal mapping (1:4,000 scale)",
            "Urban layout and master planning",
            "Detailed utility asset mapping (water lines, road networks)",
            "Disaster damage assessment at building level",
        ],
        "verified_source": "ISRO Cartosat-2 Series Documentation",
    },
    "OCM-2 / OCM-3": {
        "sensor_name": "OCM (Ocean Colour Monitor)",
        "short_name": "OCM",
        "missions": ["Oceansat-2", "EOS-06 (Oceansat-3)"],
        "spatial_resolution_m": 360.0,
        "swath_km": "1420 km",
        "swath_numeric_km": 1420.0,
        "spectral_bands": [
            {"band": "B1", "wavelength_um": "0.402 - 0.422 (Violet)"},
            {"band": "B2", "wavelength_um": "0.433 - 0.453 (Blue)"},
            {"band": "B3", "wavelength_um": "0.480 - 0.500 (Blue-Green)"},
            {"band": "B4", "wavelength_um": "0.500 - 0.520 (Green)"},
            {"band": "B5", "wavelength_um": "0.545 - 0.565 (Yellow-Green)"},
            {"band": "B6", "wavelength_um": "0.610 - 0.630 (Orange)"},
            {"band": "B7", "wavelength_um": "0.725 - 0.755 (NIR-1)"},
            {"band": "B8", "wavelength_um": "0.845 - 0.885 (NIR-2)"},
        ],
        "radiometric_resolution_bits": 12,
        "revisit_days": 2,
        "steerability": "Along-track tilt ±20° to avoid sun glint over ocean",
        "applications": [
            "Ocean chlorophyll and phytoplankton concentration mapping",
            "Potential Fishing Zone (PFZ) advisory generation (with INCOIS)",
            "Suspended sediment dynamics in coastal waters",
            "Atmospheric aerosol optical depth (AOD) characterization",
        ],
        "verified_source": "ISRO Oceansat-2 / EOS-06 Technical Brochure",
    },
    "Scatterometer (OSCAT)": {
        "sensor_name": "Ku-band Scanning Pencil-Beam Scatterometer",
        "short_name": "OSCAT",
        "missions": ["Oceansat-2", "SCATSAT-1", "EOS-06"],
        "spatial_resolution_m": 25000.0,  # 25 km x 25 km wind vector cells
        "swath_km": "1400 km (inner beam) / 1840 km (outer beam)",
        "swath_numeric_km": 1400.0,
        "spectral_bands": [
            {"band": "Ku-band Radar", "wavelength_um": "Frequency: 13.515 GHz"},
        ],
        "radiometric_resolution_bits": 12,
        "revisit_days": 2,
        "steerability": "Rotating conically scanning pencil beam reflector",
        "applications": [
            "Ocean surface wind vector (speed & direction) measurement",
            "Cyclone tracking, intensity estimation and landfall prediction",
            "Global weather forecasting data assimilation (IMD / ECMWF)",
            "Polar ice sheet boundary detection",
        ],
        "verified_source": "ISRO Oceansat-2 OSCAT Payload Specifications",
    },
}


class GeospatialSensorCatalog:
    """Provides querying, parametric comparison, and recommendation workflows."""

    @staticmethod
    def get_all_sensors() -> Dict[str, Dict[str, Any]]:
        """Return the dictionary of all verified sensors."""
        return VERIFIED_SENSORS

    @staticmethod
    def get_sensor(name: str) -> Optional[Dict[str, Any]]:
        """Find sensor specifications by exact or fuzzy name."""
        name_clean = name.strip().lower()
        for key, specs in VERIFIED_SENSORS.items():
            if name_clean in key.lower() or name_clean in specs["short_name"].lower():
                return specs
        return None

    @staticmethod
    def compare_sensors(sensor_names: List[str]) -> List[Dict[str, Any]]:
        """
        Produce a side-by-side comparison matrix for the given sensor names.
        """
        comparison = []
        for name in sensor_names:
            specs = GeospatialSensorCatalog.get_sensor(name)
            if specs:
                bands_summary = ", ".join(
                    [f"{b['band']} ({b['wavelength_um']} µm)" for b in specs["spectral_bands"][:3]]
                )
                if len(specs["spectral_bands"]) > 3:
                    bands_summary += f" + {len(specs['spectral_bands']) - 3} more"

                comparison.append(
                    {
                        "Sensor": specs["short_name"],
                        "Spatial Resolution": f"{specs['spatial_resolution_m']} m",
                        "Swath Width": specs["swath_km"],
                        "Bands Count": len(specs["spectral_bands"]),
                        "Bands Overview": bands_summary,
                        "Radiometric Bits": f"{specs['radiometric_resolution_bits']} bits",
                        "Revisit": f"{specs['revisit_days']} days",
                        "Missions": ", ".join(specs["missions"][:2]),
                        "Key Application": specs["applications"][0],
                        "Verified Source": specs["verified_source"],
                    }
                )
        return comparison

    @staticmethod
    def recommend_sensor(use_case: str) -> Dict[str, Any]:
        """
        Application recommendation engine based on operational ISRO remote sensing practices.
        Clearly separates verified factual specs from operational recommendations.
        """
        query = use_case.lower()

        if any(w in query for w in ["cadastral", "parcel", "plot", "boundary", "land record"]):
            primary = "LISS-IV"
            alternative = "Cartosat-2 PAN/MX"
            rationale = (
                "Cadastral mapping requires high spatial resolution (≤ 5.8m) to delineate individual "
                "field boundaries and property ownership plots."
            )
        elif any(w in query for w in ["dem", "elevation", "topography", "contour", "3d", "terrain"]):
            primary = "Cartosat-1 PAN"
            alternative = "Cartosat-2 PAN/MX"
            rationale = (
                "Stereo along-track cameras (+26° Fore and -5° Aft) enable high-fidelity photogrammetric "
                "Digital Elevation Model (CartoDEM) generation without temporal decorrelation."
            )
        elif any(w in query for w in ["crop acreage", "drought", "macro", "regional", "state", "vegetation index", "ndvi time", "fasal"]):
            primary = "AWiFS"
            alternative = "LISS-III"
            rationale = (
                "AWiFS combines a wide 740 km swath with a 5-day repeat cycle and 12-bit radiometry, "
                "making it ideal for dynamic agricultural monitoring and nationwide NDVI time-series."
            )
        elif any(w in query for w in ["ocean", "chlorophyll", "phytoplankton", "fish", "marine", "coastal waters", "pfz"]):
            primary = "OCM-2 / OCM-3"
            alternative = "Scatterometer (OSCAT)"
            rationale = (
                "OCM has specialized narrow spectral bands in violet/blue/green and 2-day revisit "
                "specifically tuned to quantify ocean chlorophyll-a concentrations and Potential Fishing Zones."
            )
        elif any(w in query for w in ["wind", "cyclone", "sea surface", "atmospheric"]):
            primary = "Scatterometer (OSCAT)"
            alternative = "OCM-2 / OCM-3"
            rationale = (
                "Ku-band microwave scatterometer operates day and night through cloud cover "
                "to measure ocean capillary wave scattering and retrieve sea-surface wind vectors."
            )
        else:
            # General multi-purpose Earth observation
            primary = "LISS-III"
            alternative = "LISS-IV"
            rationale = (
                "LISS-3 provides standard 23.5 m multi-spectral imaging (Green, Red, NIR, SWIR) "
                "which is the benchmark for district-level Land Use / Land Cover (LULC) and forestry."
            )

        primary_specs = VERIFIED_SENSORS[primary]
        alt_specs = VERIFIED_SENSORS.get(alternative, {})

        return {
            "query_use_case": use_case,
            "recommended_sensor": primary,
            "alternative_sensor": alternative,
            "recommendation_rationale": rationale,
            "is_verified_fact": False,  # Recommendations are operational guidance
            "verified_sensor_specs": primary_specs,
            "alternative_sensor_specs": alt_specs,
            "disclaimer": (
                "NOTE: Sensor specifications are verified official ISRO facts; "
                "application suitability is operational geospatial guidance based on NRSC best practices."
            ),
        }

