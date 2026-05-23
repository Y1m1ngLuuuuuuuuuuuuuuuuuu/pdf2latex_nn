# Obsolete Middle Fragment Experiment

`middle_fragment_view.py` was used to test whether middle-derived fragment MERGE
supervision could make the GNN useful. That experiment is no longer on the main
path.

The replacement direction is `src/perception/mineru_v8_reflow.py`, which builds
a standalone v8 content layer from MinerU `middle.json`:

```text
middle.json preproc blocks
-> page/column reading-order reflow
-> conservative continuation merge
-> content_list_v8
```

This folder is intentionally kept as a soft archive, not deleted.

