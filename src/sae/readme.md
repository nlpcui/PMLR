# psychic-fiesta

Two methods for Mechanistic Topic Extraction (MTE):

- **`llm`** - uses pretrained SAE activations from a large language model (GemmaScope, LlamaScope)
- **`emb`** - trains a local sparse autoencoder on top of sentence embeddings

---

## LLM-based MTE

### Presets

```bash
# llama-3.1-8b + 32k
python main.py llm --preset llamascope --n_samples 50000 --save_path ./mte_output/llamascope

# gemma-2-2b + 16k
python main.py llm --preset gemmascope --n_samples 50000 --save_path ./mte_output/gemmascope
```

### Manual config

```bash
python main.py llm \
  --dataset slimpajama \
  --llm_model meta-llama/Meta-Llama-3.1-8B \
  --layer_index 20 \
  --sae_release llamascope-res-32k \
  --sae_id layers.20 \
  --n_samples 5
```

### Bigger configs from Paulo et al.

#### Gemma 2 9B + 131k (main config in the paper)

```bash
python main.py llm \
  --llm_model google/gemma-2-9b \
  --layer_index 20 \
  --sae_release gemma-scope-9b-pt-res \
  --sae_id layer_20/width_131k/average_l0_34 \
  --neuronpedia_model_id gemma-2-9b \
  --neuronpedia_sae_id 20-gemmascope-res-131k \
  --dataset pile \
  --batch_size 1 \
  --chunk_size 64 \
  --do_chunking \
  --save_path ./out_gemma9b_131k
```

#### Gemma 2 9B + 16k residual

```bash
python main.py llm \
  --llm_model google/gemma-2-9b \
  --layer_index 20 \
  --sae_release gemma-scope-9b-pt-res \
  --sae_id layer_20/width_16k/average_l0_36 \
  --neuronpedia_model_id gemma-2-9b \
  --neuronpedia_sae_id 20-gemmascope-res-16k \
  --dataset pile \
  --batch_size 1 \
  --chunk_size 64 \
  --do_chunking \
  --save_path ./out_gemma9b_16k
```

#### Llama 3.1 8B + 65k

```bash
python main.py llm \
  --llm_model meta-llama/Meta-Llama-3.1-8B \
  --layer_index 15 \
  --sae_release llamascope-res-65k \
  --sae_id layers.15 \
  --neuronpedia_model_id llama3.1-8b \
  --neuronpedia_sae_id 15-llamascope-res-65k \
  --dataset slimpajama \
  --batch_size 1 \
  --chunk_size 64 \
  --do_chunking \
  --save_path ./out_llama_65k
```

### Output (`--save_path`)

| File | Saved when |
|---|---|
| `llm_document_level_dataset.json` | always |
| `llm_chunk_level_dataset.json` | only if `--save_chunks` |

Each row of `llm_document_level_dataset.json` contains:

| Field | Type | Description |
|---|---|---|
| `doc_id` | str | Original document ID |
| `text` | str | Document text |
| `theta` | List[float] | Dense SAE activation vector aggregated over the document. Length = SAE width (16k / 65k / 131k). Most values are 0.0 |
| `num_features` | int | Length of `theta` (= SAE width) |
| `theta_sparse` | List[dict] | Active features only, with per-token detail (see below) |

Each entry in `theta_sparse`:

```json
{
  "feature_id": 4821,
  "feature_weight": 2.34,
  "token_pos": [12, 7],
  "token_id": [1023, 842],
  "token_activation": [3.1, 1.8],
  "feature_label": "references to legal proceedings"
}
```

| Field | Description |
|---|---|
| `feature_id` | Index of the feature in the SAE |
| `feature_weight` | Aggregated activation value for this feature (its value in `theta`) |
| `token_pos` | Positions of the top-k tokens that activated this feature the most |
| `token_id` | Token IDs at those positions (decodable with the tokenizer) |
| `token_activation` | Raw SAE activation value for each of those tokens |
| `feature_label` | Human-readable explanation from Neuronpedia. Empty if `--no_enrich_neuronpedia` |

---

## Embedding-based MTE

Trains a local TopK sparse autoencoder on sentence embeddings.

### Quick start

```bash
python main.py emb \
  --dataset wikipedia \
  --save_path ./mte_output/emb
```

### Full config

```bash
python main.py emb \
  --dataset wikipedia \
  --embeddings_model sentence-transformers/all-MiniLM-L6-v2 \
  --M 256 \
  --K 8 \
  --embed_batch_size 32 \
  --do_chunking \
  --chunk_size 128 \
  --chunk_overlap 32 \
  --checkpoint_dir ./checkpoints \
  --cache_name mte_sae_cache \
  --save_path ./mte_output/emb
```

### Key parameters

| Parameter | Default | Description |
|---|---|---|
| `--embeddings_model` | `all-MiniLM-L6-v2` | Sentence-transformers model used to encode text |
| `--M` | 256 | Total number of SAE features (dictionary size) |
| `--K` | 8 | Number of active features per document (sparsity) |
| `--embed_batch_size` | 32 | Batch size for embedding inference |
| `--checkpoint_dir` | `./checkpoints` | Where to save the trained SAE weights |
| `--cache_name` | `mte_sae_cache` | Name for the embedding cache and checkpoint subfolder |

### Output (`--save_path`)

| File | Saved when |
|---|---|
| `emb_document_level_dataset.json` | always |
| `emb_chunk_level_dataset.json` | only if `--save_chunks` |

Each row of `emb_document_level_dataset.json` contains:

| Field | Type | Description |
|---|---|---|
| `doc_id` | str | Original document ID |
| `text` | str | Document text |
| `embeddings` | List[float] | Dense sentence embedding vector (input to the SAE) |
| `theta` | List[float] | Sparse SAE activation vector. Length = `M`. At most `K` values are non-zero |