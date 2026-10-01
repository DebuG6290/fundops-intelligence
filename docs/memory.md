# Case Memory

Case memory stores resolved operational cases as structured records.

Each case contains:
- exception type
- symptoms
- root cause
- resolution
- evidence
- supporting/contradictory signals

The current implementation uses transparent keyword retrieval as a baseline. It is intentionally simple.

Later phases can add embeddings/vector retrieval and compare whether semantic retrieval improves investigation quality. Historical cases should not be treated as ground truth for a new exception; they are evidence that informs the investigation.
