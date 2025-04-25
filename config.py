

local_config = {
    'data': {
        'wikitext': 'data/wikitext/train.metadata.jsonl',
        'bill': 'data/bill/train.metadata.jsonl'
    }
}

openai_key = ''

api_price = {
    'gpt-4o-mini': {'input': 0.15 / 1000000, 'output': 0.06 / 1000000}
}