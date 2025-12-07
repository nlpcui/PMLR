

local_config = {
    'data': {
        'wikitext': 'data/wikitext/train.metadata.jsonl',
        'bill': 'data/bill/train.metadata.jsonl'
    },
    'output': {
        'topic_models'
    },
    'embeddings': {
        'wikitext': 'data/wikitext/train.metadata.embeddings.jsonl.all-MiniLM-L6-v2.parquet',
        'bill': 'data/bill/train.metadata.embeddings.jsonl.all-MiniLM-L6-v2.parquet'
    },
    'vocab': {
        'wikitext': 'data/wikitext/vocab.json',
        'bill': 'data/bill/vocab.json'
    }
}

openai_key = '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'

api_price = {
    'gpt-4o-mini': {'input': 0.15 / 1000000, 'output': 0.6 / 1000000}
}