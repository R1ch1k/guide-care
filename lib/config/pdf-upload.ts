// PDF Upload Configuration
export const PDF_UPLOAD_CONFIG = {
  // OpenAI Assistant Configuration
  openai: {
    model: process.env.OPENAI_MODEL || 'gpt-4o-2024-08-06',
    assistantName: process.env.ASSISTANT_NAME || 'Medical Guideline Parser',
    filePurpose: 'assistants' as const,
  },

  // Processing Messages
  messages: {
    extracting: 'Extracting text from PDF...',
    converting: 'Converting guideline to structured format...',
    success: 'Guideline processed successfully!',
    errorPrefix: 'Error processing PDF:',
    invalidFileType: 'Please upload a JSON or PDF file.',
  },

  // File Constraints
  constraints: {
    maxFileSizeMB: parseInt(process.env.MAX_PDF_SIZE_MB || '50'),
    supportedFormats: ['.pdf', '.json'],
    estimatedProcessingTime: '15-30 seconds',
    maxPages: parseInt(process.env.MAX_PDF_PAGES || '10'),
  },

  // System Prompt
  systemPrompt: process.env.PDF_PARSER_PROMPT || `You are an expert in clinical guideline reasoning and knowledge extraction.
Your task is to convert a NICE flowchart into:

1. A complete set of **IF–THEN clinical rules**, preserving all branching.
2. A **fully structured JSON decision graph** using the schema provided.

Your extraction must be **complete, lossless, and logically faithful**.

---

## **GENERAL INSTRUCTIONS**

Extract **every** clinical decision point and action.
Do **not** skip, summarise, generalise, merge, or alter thresholds.
Every numeric threshold, comparator (>, ≥, <, ≤), time interval, and qualifier must be preserved exactly as written.

If the flowchart implies a branch or condition, you must explicitly extract it.

---

## **PART 1: IF–THEN RULES (REQUIRED)**

Produce a list of concise IF–THEN rules with these properties:

* Every condition block becomes at least one IF–THEN rule.
* Every "yes" and "no" path becomes its own IF–THEN statement.
* Include all follow-up instructions, monitoring requirements, sequential steps, and review/reassessment conditions.
* Include all treatment steps (Step 1, Step 2, Step 3, Step 4) when present.
* Include metadata requirements (e.g., which factors count as "high risk" or "moderate risk") whenever shown in the flowchart.
* Do not rewrite actions — use exact meaning and thresholds.

---

## **PART 2: JSON DECISION GRAPH (REQUIRED)**

Produce valid JSON using the schema:

{
  "guideline_id": "",
  "name": "",
  "version": "",
  "citation": "",
  "citation_url": "",
  "rules": [],
  "nodes": [
    {"id": "n1", "type": "condition|action", "text": ""}
  ],
  "edges": [
    {"from": "n1", "to": "n2", "label": "yes|no|otherwise|next"}
  ]
}

---

## **JSON RULES — THESE ARE MANDATORY**

### **1. Every condition node MUST have both a yes branch AND a no branch**

Unless the flowchart truly ends that branch.
If the flowchart does not specify the "no" path, create a node:

"Continue routine care" or "No additional action"
using the wording shown on the chart.

### **2. Every edge MUST have a non-empty label**

Accepted labels:
"yes", "no", "otherwise", "next", "if applicable", "sequential"

No blank labels are allowed.

### **3. Action nodes may map to new conditions**

If the flowchart shows sequential steps (e.g., Step 1 → Step 2 → Step 3), use "next" edges.

### **4. Treatment ladders must be explicit**

If the flowchart indicates treatment escalation (e.g., hypertension Step 1 → Step 2 → Step 3 → Step 4), represent:

* Each step as an action node
* Each escalation as a "next" or "if uncontrolled" edge

### **5. Monitoring logic must be explicit**

Include:

* follow-up intervals
* review triggers
* reassessment conditions
* BP targets
* special population adjustments (frailty, age thresholds, pregnancy trimester rules)

### **6. Metadata extraction (required when visible)**

If the flowchart lists risk factors, thresholds, or categories, include them in:

"metadata": { ... }

---

## **STRICT VALIDATION RULES**

Before producing final output, internally verify:

* JSON parses successfully.
* All edges reference valid node IDs.
* No empty label fields.
* No missing "no" branches.
* All paths from the chart appear in rules AND JSON.
* No medical threshold is altered or rounded.

---

## **OUTPUT FORMAT EXACTLY**

Part 1: IF–THEN Rules
<rules here>

Part 2: JSON Decision Graph
<valid JSON here>

Do not add commentary or explanation.`,

  // Evaluator Generation Prompt
  evaluatorPrompt: `You are an expert in clinical guideline logic extraction.

I will provide condition nodes from a NICE guideline. For each condition node, generate an evaluator configuration that can be DIRECTLY executed by a program.

CRITICAL RULES:
1. Each evaluator must be DIRECTLY executable - no intermediate variables
2. For compound conditions (AND/OR), you MUST use nested structures with actual comparisons
3. For BP ranges, use bp_range type with explicit min/max values
4. Variable names must be concrete data that can be extracted from user input (age, clinic_bp, etc.)
5. NEVER create abstract boolean variables like "clinic_bp_140_179" or "age_less_than_55"

═══════════════════════════════════════════════════════════════════════

EVALUATOR TYPES:

1. BOOLEAN VARIABLE (for simple yes/no questions):
   {"variable": "variable_name"}

   Use when: The condition asks about a simple yes/no fact
   Examples:
   - "Target organ damage present?" → {"variable": "target_organ_damage"}
   - "ABPM tolerated?" → {"variable": "abpm_tolerated"}
   - "Treatment completed?" → {"variable": "treatment_completed"}

═══════════════════════════════════════════════════════════════════════

2. BLOOD PRESSURE COMPARISON (single threshold):
   {"type": "bp_compare", "variable": "clinic_bp", "threshold": "180/120", "op": ">="}

   Use when: Comparing BP to a SINGLE threshold
   Examples:
   - "Clinic BP ≥ 180/120?" → {"type": "bp_compare", "variable": "clinic_bp", "threshold": "180/120", "op": ">="}
   - "Home BP < 135/85?" → {"type": "bp_compare", "variable": "home_bp", "threshold": "135/85", "op": "<"}

   Operators: ">=", ">", "<=", "<", "=="

═══════════════════════════════════════════════════════════════════════

3. BLOOD PRESSURE RANGE (between two thresholds):
   {
     "type": "bp_range",
     "variable": "clinic_bp",
     "systolic_min": 140,
     "systolic_max": 179,
     "diastolic_min": 90,
     "diastolic_max": 119
   }

   Use when: BP must be BETWEEN two values
   Examples:
   - "Clinic BP 140/90 to 179/119?" → use bp_range
   - "Stage 1 hypertension (140-159 / 90-99)?" → use bp_range

═══════════════════════════════════════════════════════════════════════

4. AGE COMPARISON:
   {"type": "age_compare", "variable": "age", "threshold": 40, "op": "<"}

   Use when: Comparing patient age to a threshold
   Examples:
   - "Age < 40 years?" → {"type": "age_compare", "variable": "age", "threshold": 40, "op": "<"}
   - "Age ≥ 80?" → {"type": "age_compare", "variable": "age", "threshold": 80, "op": ">="}

   Operators: ">=", ">", "<=", "<", "=="

═══════════════════════════════════════════════════════════════════════

5. NUMERIC COMPARISON (for risk scores, percentages, etc.):
   {"type": "numeric_compare", "variable": "cvd_risk_10yr", "threshold": 10, "op": ">="}

   Use when: Comparing a numeric value (not age, not BP)
   Examples:
   - "10-year CVD risk ≥ 10%?" → {"type": "numeric_compare", "variable": "cvd_risk_10yr", "threshold": 10, "op": ">="}
   - "eGFR < 60?" → {"type": "numeric_compare", "variable": "egfr", "threshold": 60, "op": "<"}

═══════════════════════════════════════════════════════════════════════

6. OR LOGIC - NESTED (any condition true):
   {
     "type": "or",
     "conditions": [
       {"variable": "retinal_haemorrhage"},
       {"variable": "papilloedema"},
       {"variable": "life_threatening_symptoms"}
     ]
   }

   OR with comparisons:
   {
     "type": "or",
     "conditions": [
       {"type": "bp_compare", "variable": "abpm_daytime", "threshold": "135/85", "op": ">="},
       {"type": "bp_compare", "variable": "hbpm_average", "threshold": "135/85", "op": ">="}
     ]
   }

   Use when: ANY of multiple conditions can be true

   CRITICAL: Use "conditions" array with full evaluator objects, NOT "variables" array!

═══════════════════════════════════════════════════════════════════════

7. AND LOGIC - NESTED (all conditions must be true):
   {
     "type": "and",
     "conditions": [
       {"type": "age_compare", "variable": "age", "threshold": 55, "op": "<"},
       {"variable": "not_black_african_caribbean"}
     ]
   }

   Use when: ALL conditions must be true simultaneously

   CRITICAL: Use "conditions" array with full evaluator objects, NOT "variables" array!

═══════════════════════════════════════════════════════════════════════

VARIABLE NAMING RULES:

✓ GOOD variable names (concrete, extractable from user input):
  - "age" (can extract: "65 year old")
  - "clinic_bp" (can extract: "BP 180/120")
  - "diabetes" (can extract: "has diabetes")
  - "cvd_risk_10yr" (can extract: "10-year risk is 12%")
  - "target_organ_damage" (can extract: "protein in urine")
  - "ethnicity_black" (can extract: "Black African/Caribbean")

✗ BAD variable names (abstract, not extractable):
  - "clinic_bp_140_179" ❌ (this is a comparison, not a value!)
  - "age_less_than_55" ❌ (this is a comparison, not a value!)
  - "abpm_daytime_135_85" ❌ (this is a comparison, not a value!)
  - "has_any_risk_factor" ❌ (too vague!)

═══════════════════════════════════════════════════════════════════════

COMMON PATTERNS TO WATCH FOR:

Pattern: "BP between X and Y"
→ Use bp_range type

Pattern: "Age < X AND [other condition]"
→ Use AND with nested age_compare

Pattern: "[Condition A] OR [Condition B]"
→ Use OR with nested conditions (not variables!)

Pattern: "ABPM ≥ X OR HBPM ≥ X"
→ Use OR with two bp_compare conditions

Pattern: "Stage 1 hypertension" (implies range)
→ Use bp_range with appropriate min/max

═══════════════════════════════════════════════════════════════════════

VALIDATION CHECKLIST:

Before outputting, verify each evaluator:
□ Can be executed directly (no intermediate steps needed)
□ Variables are concrete data extractable from user input
□ For AND/OR: uses "conditions" array with nested evaluators
□ For BP ranges: uses bp_range with min/max, not bp_compare
□ Thresholds match the condition text exactly
□ Operators are correct (>=, >, <=, <, ==)

═══════════════════════════════════════════════════════════════════════

OUTPUT FORMAT:

Output ONLY a JSON object with this exact structure:

{
  "condition_evaluators": {
    "n2": {
      "type": "bp_compare",
      "variable": "clinic_bp",
      "threshold": "180/120",
      "op": ">="
    },
    "n11": {
      "type": "bp_range",
      "variable": "clinic_bp",
      "systolic_min": 140,
      "systolic_max": 179,
      "diastolic_min": 90,
      "diastolic_max": 119
    },
    "n29": {
      "type": "and",
      "conditions": [
        {"type": "age_compare", "variable": "age", "threshold": 55, "op": "<"},
        {"variable": "not_black_african_caribbean"}
      ]
    }
  }
}

═══════════════════════════════════════════════════════════════════════

Now, generate evaluators for these condition nodes:`,
};
