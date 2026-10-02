# Case Memory

Case memory stores resolved operational cases as structured records.

Each case contains:
- exception type
- symptoms
- root cause
- resolution
- evidence
- supporting/contradictory signals
- investigation path
- useful evidence types
- human validation and review metadata

The current implementation uses transparent keyword retrieval as a baseline. It is intentionally simple. Memory influences investigation planning but never establishes truth. Historical cases are analogies, not evidence about the current exception. Only an explicit human ACCEPT creates validated operational memory; REJECT and INVESTIGATE_FURTHER do not.

Later phases can add embeddings/vector retrieval and compare whether semantic retrieval improves investigation quality. Historical cases should not be treated as ground truth for a new exception; they are evidence that informs the investigation.

