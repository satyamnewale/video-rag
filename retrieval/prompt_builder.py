from langchain_core.prompts import ChatPromptTemplate


class PromptBuilder:

    def __init__(self):

        self.prompt = ChatPromptTemplate.from_messages([

            (
                "system",
                """
You are a VideoRAG assistant.

Your task is to answer user questions using
only the supplied video context.

Rules:

1. Use the retrieved context as your evidence.

2. Do not invent events, dialogue, objects,
   characters, or actions.

3. If the context does not contain enough
   information, clearly state that.

4. When making a factual statement, cite the
   supporting source using its source number.

5. Use citations in this format:
   [Source 1], [Source 2].

6. If multiple sources support an answer,
   cite all relevant sources.

7. Include video timestamps when relevant.

8. Distinguish between:
   - Spoken dialogue
   - Text visible on screen
   - Visual descriptions
   - Object detections

9. Object detection alone does not prove
   an action or event occurred.

10. Do not interpret missing information
    as proof that something did not happen.

Keep the answer clear and concise.
"""
            ),

            (
                "human",
                """
Video context:

{context}

User question:

{question}

Provide an evidence-based answer with citations.
"""
            )

        ])

    # -----------------------------------------
    # Create formatted prompt
    # -----------------------------------------

    def build_prompt(
        self,
        context,
        question
    ):

        return self.prompt.invoke({

            "context": context,

            "question": question

        })