import uuid
from typing import Dict, Any
from transformers import PreTrainedTokenizerBase


def chunk_batch_fn(
    batch: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int,
    chunk_overlap: int,
    text_col: str,
    id_col: str,
    return_text: bool = True,
) -> Dict[str, Any]:
    """
    Tokenize texts in batch and create overlapping token chunks.
    If return_text=False, skips decoding to save time.

    Parameters
    ----------
    batch : dict
        A batch of documents containing text and IDs.
    tokenizer : AutoTokenizer
        The tokenizer to use for chunking the text.
    chunk_size : int
        The size of each chunk in tokens.
    chunk_overlap : int
        The number of overlapping tokens between consecutive chunks.
    text_col : str
        The name of the column containing the text.
    id_col : str
        The name of the column containing the document IDs.
    return_text : bool, optional
        Whether to return the text of each chunk, by default True.

    Returns
    -------
    dict
        A dictionary with lists of chunk IDs, document IDs, texts, and token IDs.
    """
    all_chunk_ids = []
    all_doc_ids = []
    all_texts = []
    all_token_ids = []

    stride = chunk_size - chunk_overlap
    if stride <= 0:
        stride = max(1, chunk_size // 2)

    texts = batch.get(text_col, [])
    doc_ids = batch.get(id_col, [])
    if not texts:
        return {"chunk_id": [], "doc_id": [], "text": [], "token_ids": []}

    # Batch tokenization (fast)
    enc = tokenizer(
        texts,
        truncation=False,
        add_special_tokens=False,
        return_attention_mask=False,
    )

    for input_ids, doc_id, text in zip(enc["input_ids"], doc_ids, texts):
        if not text or not input_ids:
            continue

        for i in range(0, len(input_ids), stride):
            start_idx = i
            end_idx = i + chunk_size
            this_chunk_ids = input_ids[start_idx:end_idx]
            if not this_chunk_ids:
                continue

            all_chunk_ids.append(str(uuid.uuid4()))
            all_doc_ids.append(doc_id)
            all_token_ids.append(this_chunk_ids)

            if return_text:
                all_texts.append(
                    tokenizer.decode(
                        this_chunk_ids,
                        clean_up_tokenization_spaces=False,
                        skip_special_tokens=True,
                    )
                )
            else:
                all_texts.append("")  # keep columns consistent
            if end_idx >= len(input_ids):
                break

    return {
        "chunk_id": all_chunk_ids,
        "doc_id": all_doc_ids,
        "text": all_texts,
        "token_ids": all_token_ids,
    }
