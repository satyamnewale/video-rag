from langchain_openai import ChatOpenAI


class ResponseGenerator:

    def __init__(
        self,
        model="gpt-4o-mini",
        temperature=0
    ):

        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature
        )

    # -----------------------------------------
    # Generate response
    # -----------------------------------------

    def generate(self, prompt_value):

        response = self.llm.invoke(
            prompt_value
        )

        return response.content