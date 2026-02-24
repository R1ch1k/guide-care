# NICE Guidelines — Data Format

## Overview

The system uses 10 NICE clinical guidelines, each stored as two JSON files:

- **Guideline JSON** (`backend/data/guidelines/<id>.json`) — the decision tree structure
- **Evaluator JSON** (`backend/data/evaluators/<id>_eval.json`) — condition evaluation logic

## Guideline JSON Structure

```json
{
  "guideline_id": "NG84",
  "name": "Sore throat (acute): antimicrobial prescribing",
  "version": "NG84 (2018)",
  "citation": "NICE guideline [NG84]",
  "citation_url": "https://www.nice.org.uk/guidance/ng84",
  "nodes": [
    {
      "id": "n1",
      "type": "condition",
      "text": "FeverPAIN score ≥ 4 or Centor score ≥ 3?"
    },
    {
      "id": "n5",
      "type": "action",
      "text": "Consider delayed antibiotic prescription..."
    }
  ],
  "edges": [
    {"from": "n1", "to": "n2", "label": "yes"},
    {"from": "n1", "to": "n3", "label": "no"}
  ]
}
```

### Node Types

- **condition** — decision point evaluated against patient variables
- **action** — terminal recommendation node (the output shown to the user)

### Edge Labels

- **yes** — condition evaluates to true
- **no** — condition evaluates to false
- **next** — unconditional progression

## Evaluator JSON Structure

Maps each condition node to its evaluation logic:

```json
{
  "n1": {
    "variable": "feverpain_score",
    "operator": ">=",
    "value": 4
  },
  "n2": {
    "and": [
      {"variable": "age", "operator": ">=", "value": 18},
      {"variable": "immunocompromised", "operator": "==", "value": false}
    ]
  }
}
```

### Supported Condition Types

| Type | Example |
|------|---------|
| Simple comparison | `{"variable": "age", "operator": ">=", "value": 65}` |
| Boolean check | `{"variable": "pregnant", "operator": "==", "value": true}` |
| BP comparison | `{"variable": "bp", "operator": ">=", "value": "180/120"}` |
| BP range | `{"variable": "bp", "operator": "range", "value": "140/90-179/119"}` |
| AND compound | `{"and": [{...}, {...}]}` |
| OR compound | `{"conditions": [{...}, {...}], "match": "any"}` |
| Treatment type | `{"variable": "treatment_type", "operator": "==", "value": "monotherapy"}` |

## Guidelines Included

| File | ID | Guideline |
|------|----|-----------|
| ng84.json | NG84 | Sore throat — antimicrobial prescribing |
| ng91.json | NG91 | Otitis media — ear infection |
| ng112.json | NG112 | UTI (lower) — urinary tract infection |
| ng133.json | NG133 | Hypertension in pregnancy |
| ng136.json | NG136 | Hypertension in adults |
| ng184.json | NG184 | Bite wounds |
| ng222.json | NG222 | Depression in adults |
| ng232.json | NG232 | Head injury |
| ng81_chronic_glaucoma.json | NG81_GLAUCOMA | Chronic open-angle glaucoma |
| ng81_ocular_hypertension.json | NG81_HYPERTENSION | Ocular hypertension |

## How Traversal Works

1. Start at the first node (usually `n1`)
2. If condition node: evaluate using the evaluator spec and known variables
3. Follow the matching edge (yes/no/next)
4. If a required variable is missing: record it and stop
5. If action node: record the recommendation text
6. Continue until all reachable paths are exhausted

The engine returns:
- **reached_actions** — list of action node texts (the recommendations)
- **path** — ordered list of `(node_id, next_node_id, edge_label)` tuples
- **missing_variables** — variables needed but not yet known
