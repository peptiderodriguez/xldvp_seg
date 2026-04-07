# Omic Linker

Bridge morphological features to mass-spec proteomics data (DVP workflow).
Supports direct search engine report parsing via dvp-io (AlphaDIA, DIANN,
MaxQuant, Spectronaut, etc.) or plain CSV loading.

```python
from xldvp_seg.analysis.omic_linker import OmicLinker

linker = OmicLinker.from_detections(detections)
# Option A: plain CSV
linker.load_proteomics("proteomics.csv")
# Option B: search engine report (dvp-io, included)
linker.load_proteomics_report("diann_report.tsv", search_engine="diann")

linker.load_well_mapping("lmd_export/")
linked = linker.link()  # DataFrame: aggregated features + proteins per well
diff = linker.differential_features("marker_profile", "NeuN+", "NeuN-")
```

::: xldvp_seg.analysis.omic_linker
    options:
      show_root_heading: false
      members_order: source
