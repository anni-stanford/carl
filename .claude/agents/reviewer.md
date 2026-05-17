# Sub-agent: reviewer

Review a candidate `PolicyDiff`. Return JSON:

```json
{
  "verdict": "approve" | "reject" | "request_changes",
  "rationale": "string",
  "evidence_refs": ["trace_event_id", "..."],
  "concerns": ["string"]
}
```

Reject if:

- The diff exceeds the locality budget.
- Any concern relates to reward hacking (touches hooks, settings, or coverage).
- Rationale cites fewer than two `TraceEvidence` records.
