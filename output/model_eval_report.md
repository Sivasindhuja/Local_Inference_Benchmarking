# Structured JSON Extraction Evaluation Report

This report compares three local models on refund-ticket structured extraction using Pydantic validation and a single retry policy.

## Overall results

| Model | Total | Pass 1st try | Pass after retry | Parse fails | Content fails | 1st try rate | With retry rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen2.5:1.5b | 24 | 17 | 2 | 0 | 5 | 70.83% | 79.17% |

## Error breakdown by model

### qwen2.5:1.5b

| Error type | Count |
|---|---:|
| hallucinated_currency | 1 |
| hallucinated_email | 2 |
| hallucinated_refund_amount | 1 |
| missed_currency | 1 |
| wrong_email | 1 |
| wrong_name | 1 |
| wrong_refund_amount | 1 |

## Edge-case coverage

Test categories include happy path, missing fields, multiple candidates, forwarded/quoted chains, signature noise, negation, multilingual text, OCR noise, written amounts, currency variants, special-character names, fake JSON bait, and irrelevant refund mentions.

## Notes

- A pass requires valid JSON, valid Pydantic schema, and exact match against expected normalized fields.

- If the first response fails syntax or schema validation, the harness reprompts the model once before marking failure.

- `refund_amount` is normalized to 2 decimals; currency symbols are mapped to ISO codes when possible.
