# Decision Log

Every non-trivial analytical or engineering choice, recorded when it is made.
Entries are never deleted; a reversed decision gets a new entry that supersedes the old one.

| ID | Date | Phase | Decision | Alternatives considered | Rationale | Reversible? |
|----|------|-------|----------|-------------------------|-----------|-------------|
| DEC-001 | 2026-11-11 | 0 | Raw data is not version-controlled; provenance is tracked via a SHA-256 manifest | Commit data to git; DVC | File sizes (ERA5 NetCDF), EM-DAT forbids redistribution, checksums give integrity without storage | Yes (DVC can be added later) |
| DEC-002 | 2026-11-11 | 0 | Python environment managed with conda (conda-forge channel) | uv / pip | Windows 11: compiled geospatial stack (GDAL, GEOS, PROJ, netCDF4/HDF5) installs reliably from conda-forge binaries | Yes |
| DEC-003 | 2026-11-11 | 0 | Python and R exchange data only via Parquet files in `data/processed/` | rpy2 / reticulate in-memory bridge | Explicit, language-neutral, typed interface; keeps inference and prediction pipelines independent | Yes |
| DEC-004 | 2026-11-11 | 0 | Line endings normalized to LF via `.gitattributes` | Default Windows CRLF | CI runs on Linux; Makefile and shell scripts break with CRLF | Yes |