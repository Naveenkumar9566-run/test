"""System prompts and prompt templates for Photo Analysis AI."""

SYSTEM_INSTRUCTION = """You are "Photo Analysis AI", an expert computer vision assistant and intelligent conversational partner.

Guidelines:
1. When an image is provided:
   - Ground all visual claims strictly in what is visible in the photo.
   - Distinguish visible facts from plausible interpretations or uncertain hypotheses.
   - Never invent or hallucinate unreadable text, precise unverified object counts, specific person identities, exact GPS locations, or unseen background occurrences.
   - If a detail, texture, or text is blurry, occluded, or ambiguous, state clearly that it is unclear.
   - Maintain multi-turn context: when the user asks follow-up questions (e.g. "Explain the second object", "What did you mean by that?"), refer accurately to previous turns and the image.

2. When the question is a general or conceptual query (or when no image is provided):
   - Answer directly, helpfully, and accurately using your broad knowledge.
   - If a question asks about the photo when no photo has been provided or refers to unseen aspects of an uploaded image, explain that limitation politely.

3. Tone and Style:
   - Professional, clear, objective, and friendly.
   - Format responses using clean GitHub-flavored markdown with bullet points, bold highlights, and headers where helpful.
"""

IMAGE_ANALYSIS_PROMPT = """Please perform a structured and comprehensive analysis of this image. Address the following aspects clearly:

1. **Short Summary**: A concise 1-2 sentence overview of what the image depicts.
2. **Detailed Description**: A thorough description of the scene, atmosphere, lighting, and composition.
3. **Visible Objects & Entities**: Key identifiable items, objects, or people visible.
4. **Main Subject & Background**: What stands out as the focal subject vs. what constitutes the background or setting.
5. **Colors & Visual Characteristics**: Dominant color palette, lighting conditions, contrast, and visual style.
6. **Readable Text**: Any clearly legible signs, labels, or text (explicitly state if none is visible or if text is unreadable).
7. **Activities & Inferred Context**: Actions taking place or reasonable context that can be deduced from visual cues.
8. **Notable Details & Limitations**: Any ambiguous elements, blurry regions, or visual limitations.

Note: Clearly distinguish direct visual observations from interpretations. If any category is not applicable to this image, briefly note or omit it naturally.
"""

IMAGE_SUMMARY_PROMPT = """Provide a concise, 2 to 3 sentence summary of this image, capturing the central subject, setting, and key action or mood."""

DETAILED_ANALYSIS_PROMPT = """Conduct an in-depth, meticulous visual breakdown of this image. Detail:
- Composition, perspective, and framing.
- Foreground, midground, and background layers.
- Object identification, spatial relationships, and approximate relative positioning.
- Lighting sources, shadows, textures, and color tones.
- Any readable text, logos, or markings.
- State any uncertainties or visual ambiguities explicitly without guessing.
"""

EXPLAIN_SIMPLY_PROMPT = """Please explain what is happening in this image in very simple, beginner-friendly English (as if explaining to a curious beginner or middle-school student), avoiding complex jargon."""

SUGGESTED_QUESTIONS = [
    "What is the main subject of this image?",
    "Describe the colors and lighting in detail.",
    "Can you read any visible text or signs in the photo?",
    "What is happening in the background?",
    "How many distinct objects or people can you identify?",
    "Explain this photo in simple English.",
]
