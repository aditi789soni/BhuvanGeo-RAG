"""
Unit tests for the Geospatial Intelligence module.
Verifies official ISRO sensor parameters, comparisons, and recommendations.
"""

import pytest
from src.geospatial import GeospatialSensorCatalog, VERIFIED_SENSORS


def test_verified_sensors_presence():
    """Ensure all core ISRO sensors are present with required technical parameters."""
    required_sensors = ["LISS-IV", "LISS-III", "AWiFS", "Cartosat-1 PAN", "OCM-2 / OCM-3"]
    all_sensors = GeospatialSensorCatalog.get_all_sensors()

    for s_name in required_sensors:
        assert s_name in all_sensors, f"Missing required sensor: {s_name}"
        sensor = all_sensors[s_name]
        assert "spatial_resolution_m" in sensor
        assert "swath_km" in sensor
        assert "spectral_bands" in sensor
        assert "missions" in sensor
        assert len(sensor["spectral_bands"]) > 0


def test_liss4_specifications():
    """Verify official LISS-IV sensor specifications."""
    liss4 = GeospatialSensorCatalog.get_sensor("LISS-IV")
    assert liss4 is not None
    assert liss4["spatial_resolution_m"] == 5.8
    assert liss4["radiometric_resolution_bits"] == 10
    assert "Resourcesat-2" in liss4["missions"]
    assert "B2 (Green)" in [b["band"] for b in liss4["spectral_bands"]]


def test_awifs_specifications():
    """Verify official AWiFS sensor specifications."""
    awifs = GeospatialSensorCatalog.get_sensor("AWiFS")
    assert awifs is not None
    assert awifs["spatial_resolution_m"] == 56.0
    assert awifs["swath_numeric_km"] == 740.0
    assert awifs["revisit_days"] == 5
    assert awifs["radiometric_resolution_bits"] == 12


def test_cartosat1_stereo_pan():
    """Verify Cartosat-1 stereoscopic along-track camera configuration."""
    carto1 = GeospatialSensorCatalog.get_sensor("Cartosat-1 PAN")
    assert carto1 is not None
    assert carto1["spatial_resolution_m"] == 2.5
    assert "Fore" in carto1["steerability"]
    assert "Aft" in carto1["steerability"]


def test_sensor_comparison():
    """Verify side-by-side parametric comparison functionality."""
    comp = GeospatialSensorCatalog.compare_sensors(["LISS-IV", "AWiFS"])
    assert len(comp) == 2
    sensor_names = [row["Sensor"] for row in comp]
    assert "LISS-IV" in sensor_names
    assert "AWiFS" in sensor_names


def test_recommendation_cadastral():
    """Verify operational recommendation for cadastral parcel mapping."""
    rec = GeospatialSensorCatalog.recommend_sensor("cadastral boundary and land record parcel mapping")
    assert rec["recommended_sensor"] == "LISS-IV"
    assert "5.8" in rec["recommendation_rationale"]
    assert rec["is_verified_fact"] is False  # Explicitly distinguished from raw facts


def test_recommendation_crop_drought():
    """Verify operational recommendation for national crop / drought assessment."""
    rec = GeospatialSensorCatalog.recommend_sensor("national crop acreage estimation and drought NDVI time composite")
    assert rec["recommended_sensor"] == "AWiFS"
    assert "740" in rec["recommendation_rationale"]

