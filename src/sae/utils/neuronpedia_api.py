
from typing import Dict
import json
import requests

_REQUEST_TIMEOUT = 10  # seconds


def fetch_feature_explanation(
    feature_idx: int,
    expl_cache: Dict[int, str],
    neronepedia_api_base: str = "https://www.neuronpedia.org/api/feature",
    neronepedia_model_id: str = "gemma-2-2b",
    neronepedia_sae_id: str = "24-gemmascope-res-16k",
    timeout: int = _REQUEST_TIMEOUT,
) -> str:
    """
    Get the explanation of a feature based on neuronpedia annotations.
    Usamos caché para no repetir peticiones.
    """
    if feature_idx in expl_cache:
        return expl_cache[feature_idx]

    url = f"{neronepedia_api_base}/{neronepedia_model_id}/{neronepedia_sae_id}/{feature_idx}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    explanation = None
    if "explanations" in data and isinstance(data["explanations"], list) and data["explanations"]:
        for cand in data["explanations"]:
            if isinstance(cand, dict):
                explanation = cand.get("description", None)
                if explanation is not None:
                    break

    if explanation is None:
        explanation = "<no explanation field found in JSON>"

    expl_cache[feature_idx] = explanation
    return explanation


def load_bulk_explanations(
    model_id: str,
    sae_id: str,
    layer_index: int,
    s3_base: str = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com",
    n_batches: int = 16,
    timeout: int = 300,
) -> Dict[int, str]:
    """
    Download all bulk explanation batches for a given model/SAE layer from the
    Neuronpedia S3 bucket and return a feature_id -> explanation dict.

    S3 path pattern::

        {s3_base}/index.html?prefix=v1/{model_id}/{layer_index}-{sae_suffix}/batch-{N}.jsonl.gz

    "layer_index" is the authoritative layer number.  Any leading ""{N}-""
    prefix in "sae_id" is stripped and replaced with "layer_index".

    Each record in the JSONL has the shape::

        {"index": "<str>", "description": "<str>", ...}

    Features without a description get the label "<no explanation>".
    """
    import gzip
    import io
    import re

    sae_suffix = re.sub(r"^\d+-", "", sae_id)
    s3_sae_id = f"{layer_index}-{sae_suffix}"
    s3_prefix = f"v1/{model_id}/{s3_sae_id}/explanations"

    import logging
    _log = logging.getLogger(__name__)
    _log.info("Bulk explanations prefix: %s/%s  (batches 0..%d)",
              s3_base, s3_prefix, n_batches - 1)

    cache: Dict[int, str] = {}
    for batch_idx in range(n_batches):
        url = f"{s3_base}/{s3_prefix}/batch-{batch_idx}.jsonl.gz"
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 404:
            _log.info("batch-%d → 404, stopping at %d total features",
                      batch_idx, len(cache))
            break  # no more batches
        resp.raise_for_status()

        before = len(cache)
        with gzip.open(io.BytesIO(resp.content), "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                fid = record.get("index")
                if fid is None:
                    continue
                description = record.get("description") or "<no explanation>"
                cache[int(fid)] = description

        _log.info("batch-%d: loaded %d features (running total %d)",
                  batch_idx, len(cache) - before, len(cache))

    return cache
