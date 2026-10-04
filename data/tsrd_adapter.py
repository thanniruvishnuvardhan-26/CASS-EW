"""
TSRD Unified Adapter & Normalized Data Contract
CASS-EW SIH Problem Statement 26055

Re-exports the canonical PDW representation, ValidationReport, and Adapter classes
from `data.dataset_adapter` while maintaining full backwards compatibility.
"""

from data.dataset_adapter import (
    ObservablePDW,
    GroundTruthPDW,
    DatasetValidationReport,
    BaseDatasetAdapter,
    CSVDatasetAdapter,
    JSONDatasetAdapter,
    HDF5DatasetAdapter,
    TSRDAdapter,
    load_dataset,
    DatasetConfig,
    UnitConverter
)

__all__ = [
    "ObservablePDW",
    "GroundTruthPDW",
    "DatasetValidationReport",
    "BaseDatasetAdapter",
    "CSVDatasetAdapter",
    "JSONDatasetAdapter",
    "HDF5DatasetAdapter",
    "TSRDAdapter",
    "load_dataset",
    "DatasetConfig",
    "UnitConverter"
]
