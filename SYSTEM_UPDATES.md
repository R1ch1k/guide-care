# System Updates - NICE Guidelines Integration

## Changes Made

### 1. **Environment Configuration**
- ✅ Updated `.env.local` to use GPT-5 model
- ✅ API key is protected (already in `.gitignore`)

### 2. **PDF Parsing System** 
- ✅ Updated system prompt in `lib/config/pdf-upload.ts` with NICE-specific extraction rules
- ✅ Updated Zod schema to extract:
  - `rules`: Array of IF-THEN rule strings
  - `nodes`: Array of {id, type, text} for flowchart visualization
  - `edges`: Array of {from, to, label} for decision pathways

### 3. **Type System**
- ✅ Added `NICEGuideline` type for new format
- ✅ Added `AnyGuideline` union type to support both formats
- ✅ Kept legacy `Guideline` type for backwards compatibility

### 4. **Chat API**
- ✅ Fixed error where `inputs` was being accessed on NICE guidelines
- ✅ Added type guards to detect format
- ✅ Updated both `buildStrictPrompt` and `buildExplainPrompt` to handle both formats
- ✅ NICE format prompts focus on IF-THEN rules
- ✅ Legacy format prompts use decision tree structure

### 5. **New Feature: Guideline Viewer Component**
- ✅ Created `GuidelineViewer.tsx` with two tabs:
  - **IF-THEN Rules tab**: Displays all extracted rules in a clean, numbered format
  - **JSON Structure tab**: Shows complete JSON for testing/verification
- ✅ Copy functionality for both rules and JSON
- ✅ Shows decision graph summary (condition count, action count)
- ✅ Modal overlay design

### 6. **Updated Components**
- ✅ `GuidelineSelector.tsx`: Added "View JSON & Rules" button for NICE guidelines
- ✅ `ChatPanel.tsx`: Updated to accept both format types
- ✅ `app/page.tsx`: Updated to use `AnyGuideline` type
- ✅ Validation handles both NICE and legacy JSON formats

## How It Works

### For Users:
1. **Upload NICE PDF**: The system extracts IF-THEN rules and decision graph
2. **View Button Appears**: When a NICE guideline is loaded, a "View JSON & Rules" button shows
3. **Inspect Extraction**: Click the button to see:
   - All IF-THEN rules extracted from the flowchart
   - Complete JSON structure
   - Copy either format for testing
4. **Chat with AI**: The AI strictly follows the extracted IF-THEN rules
5. **Testing Mode**: You can verify the AI follows the JSON exactly by comparing responses to the rules

### For Testing:
```typescript
// NICE Format (New)
{
  "guideline_id": "nice_xxx",
  "name": "...",
  "rules": ["IF ... THEN ..."],
  "nodes": [{id, type, text}],
  "edges": [{from, to, label}]
}

// Legacy Format (Still supported)
{
  "guideline_id": "nice_xxx",
  "name": "...",
  "inputs": [...],
  "nodes": [{if, then, else, actions}]
}
```

## Current Workflow

1. **PDF Upload** → OpenAI extracts using NICE prompt → Returns JSON with rules + graph
2. **View JSON & Rules** → Modal shows IF-THEN rules and complete JSON structure
3. **Chat** → AI uses IF-THEN rules to guide clinical decisions
4. **Verification** → Compare AI responses with displayed rules to ensure strict adherence

## Model Configuration

Currently set to: `OPENAI_MODEL=gpt-5`

Recommendations:
- **Development/Testing**: Use `gpt-4o-mini` (60-80% cheaper)
- **Production**: Use `gpt-4o` or `gpt-4o-2024-08-06` (more accurate)
- **GPT-5**: May need to verify exact model name with OpenAI API docs

## Next Steps

1. Test PDF upload with a NICE guideline
2. Verify IF-THEN rules extraction quality
3. Use "View JSON & Rules" to inspect output
4. Test chatbot adherence to rules
5. Adjust model if needed based on cost/quality tradeoff
