# NICE Guidelines Processing System

## Overview

This system processes NICE (National Institute for Health and Care Excellence) clinical guidelines from PDF flowcharts and extracts them into machine-readable formats.

## Output Format

When you upload a NICE guideline PDF, the system extracts:

### 1. IF-THEN Rules
Concise logical rules that express the clinical decision pathways:
```
IF BP ≥ 180/120 AND retinal haemorrhage OR papilloedema OR life-threatening symptoms THEN refer for same-day specialist review.
IF BP ≥ 180/120 AND target-organ damage (no emergency) THEN start treatment immediately.
```

### 2. JSON Decision Graph
A structured representation with nodes and edges:
```json
{
  "guideline_id": "nice_hypertension_ng136",
  "name": "Hypertension in adults: diagnosis and management",
  "version": "NG136 (2023)",
  "citation": "NICE guideline [NG136]",
  "citation_url": "https://www.nice.org.uk/guidance/ng136",
  "rules": [
    "IF BP ≥ 180/120 AND emergency signs THEN refer for same-day specialist review.",
    "IF BP ≥ 180/120 AND no emergency THEN start treatment immediately."
  ],
  "nodes": [
    {"id": "n1", "type": "condition", "text": "BP ≥ 180/120?"},
    {"id": "n2", "type": "condition", "text": "Emergency signs present?"},
    {"id": "n3", "type": "action", "text": "Refer for same-day specialist review"},
    {"id": "n4", "type": "action", "text": "Start treatment immediately"}
  ],
  "edges": [
    {"from": "n1", "to": "n2", "label": "yes"},
    {"from": "n2", "to": "n3", "label": "yes"},
    {"from": "n2", "to": "n4", "label": "no"}
  ]
}
```

## Configuration

### Model Selection
Edit `.env.local` to set your preferred OpenAI model:
```bash
OPENAI_MODEL=gpt-4.5-turbo-preview
```

### API Key
Add your OpenAI API key to `.env.local`:
```bash
OPENAI_API_KEY=sk-proj-...
```

## Extraction Rules

The system follows these principles when processing NICE guidelines:

1. **Preserve clinical logic exactly** - All thresholds, comparators, and numerical values are kept precise
2. **Merge redundant branches** - Parallel pathways with identical outcomes are combined
3. **Avoid unnecessary qualifiers** - Demographic splits that don't change outcomes are simplified
4. **Maintain NICE-style phrasing** - Uses clinical verbs like "Offer", "Consider", "Refer", "Assess"
5. **Strip irrelevant text** - Removes footers, references, URLs, and formatting noise
6. **Compress decision logic** - Expresses equivalent decisions concisely
7. **Preserve quantitative precision** - Keeps all numbers, ranges, units, and comparators exact
8. **No commentary** - Outputs only clean logical structure

## Node Types

- **condition**: Decision points (diamond shapes in flowcharts)
- **action**: Recommendations or actions (rectangles in flowcharts)

## Edge Labels

- **yes**: Condition is true
- **no**: Condition is false
- **otherwise**: Alternative pathway
- **(empty)**: Linear progression with no branching

## Usage

1. Upload a NICE guideline PDF through the application interface
2. The system will extract the IF-THEN rules and decision graph
3. The structured output can be used for:
   - Computational reasoning
   - Clinical decision support systems
   - Visualization tools
   - Integration with other healthcare systems

## Backwards Compatibility

The system maintains support for the legacy guideline format with `inputs` and decision tree nodes. Existing guidelines like `nice-hypertension.ts` will continue to work with the legacy `DecisionEngine`.

New NICE guidelines extracted using this system will use the simpler graph-based format optimized for flowchart representation.
