To create a concise prompt for each ambiguity type and implement a model-based approach for ambiguity detection, let's start by summarizing each ambiguity type into a shorter definition for use in the prompt. Then, I'll outline how you might use a model like Llama to judge each sentence against these types of ambiguity based on the generated knowledge graph.

### Simplified Ambiguity Definitions for Prompt:
- **T1 (Unclear Relations)**: Ambiguity arising from undefined or vague relationships between process elements.
- **T2 (Unclear References)**: Ambiguity due to vague descriptions that don't clearly identify process elements.
- **T3 (Underspecifications)**: Ambiguity from omitted details or partially described process aspects, leading to gaps in understanding.
- **T4 (Inconsistent Specifications)**: Ambiguity from conflicting requirements or descriptions within the process.
