
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List
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


def prefetch_feature_explanations(
    feature_ids: List[int],
    neronepedia_api_base: str = "https://www.neuronpedia.org/api/feature",
    neronepedia_model_id: str = "gemma-2-2b",
    neronepedia_sae_id: str = "24-gemmascope-res-16k",
    max_workers: int = 16,
    timeout: int = _REQUEST_TIMEOUT,
) -> Dict[int, str]:
    """
    Fetch Neuronpedia explanations for a list of feature ids in parallel.
    Returns a dict {feature_id -> explanation}.
    Features that fail get the label "<fetch error>".
    """
    cache: Dict[int, str] = {}
    unique_ids = list(set(feature_ids))

    def _fetch(fid: int) -> tuple:
        try:
            label = fetch_feature_explanation(
                feature_idx=fid,
                expl_cache={},  # no shared cache needed; we aggregate below
                neronepedia_api_base=neronepedia_api_base,
                neronepedia_model_id=neronepedia_model_id,
                neronepedia_sae_id=neronepedia_sae_id,
                timeout=timeout,
            )
        except Exception:
            label = "<fetch error>"
        return fid, label

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch, fid): fid for fid in unique_ids}
        for future in as_completed(futures):
            fid, label = future.result()
            cache[fid] = label

    return cache
