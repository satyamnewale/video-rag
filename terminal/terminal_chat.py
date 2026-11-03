import argparse
import os

from dotenv import load_dotenv

from embeddings.embedding_processor import EmbeddingProcessor
from storage.chrome_storage import ChromaStorage

from retrieval.vector_retriever import VectorRetriever
from retrieval.reranker import Reranker
from retrieval.context_builder import ContextBuilder
from retrieval.prompt_builder import PromptBuilder
from retrieval.response_generator import ResponseGenerator


class TerminalChat:

    def __init__(
        self,
        video_id=None,
        retrieval_top_k=20,
        rerank_top_n=5,
        model="gpt-4o-mini"
    ):

        load_dotenv()

        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError(
                "OPENAI_API_KEY not found. "
                "Add it to your .env file."
            )

        # --------------------------------
        # Initialize shared dependencies
        # --------------------------------

        self.embedding_processor = EmbeddingProcessor()

        self.chroma_storage = ChromaStorage()

        # --------------------------------
        # Initialize retrieval components
        # --------------------------------

        self.vector_retriever = VectorRetriever(
            embedding_processor=self.embedding_processor,
            chroma_storage=self.chroma_storage,
            top_k=retrieval_top_k
        )

        self.reranker = Reranker(
            top_n=rerank_top_n
        )

        self.context_builder = ContextBuilder()

        self.prompt_builder = PromptBuilder()

        self.response_generator = ResponseGenerator(
            model=model,
            temperature=0
        )

        self.video_id = video_id

        self.retrieval_top_k = retrieval_top_k

        self.rerank_top_n = rerank_top_n

    # -----------------------------------------
    # Display application information
    # -----------------------------------------

    def display_header(self):

        total_records = (
            self.chroma_storage.count_records()
        )

        print("\n" + "=" * 55)

        print("                 VIDEO RAG CHAT")

        print("=" * 55)

        print(f"Available records : {total_records}")

        print(
            f"Video filter      : "
            f"{self.video_id or 'All videos'}"
        )

        print(
            f"Retrieval top-k   : {self.retrieval_top_k}"
        )

        print(
            f"Reranker top-n    : {self.rerank_top_n}"
        )

        print("Generation model  : gpt-4o-mini")

        print("=" * 55)

        print("Type 'exit' or 'quit' to close the application.")

        print("=" * 55)

    # -----------------------------------------
    # Display source information
    # -----------------------------------------

    @staticmethod
    def display_sources(sources):

        if not sources:
            print("\nNo sources available.")
            return

        print("\n" + "-" * 55)

        print("RETRIEVED SOURCES")

        print("-" * 55)

        for source in sources:

            print(
                f"\nSource {source['source_number']}"
            )

            print(
                f"Record ID : {source['record_id']}"
            )

            print(
                f"Video ID  : {source['video_id']}"
            )

            print(
                f"Scene ID  : {source['scene_id']}"
            )

            print(
                f"Timestamp : "
                f"{source['start_timestamp']} - "
                f"{source['end_timestamp']}"
            )

            print(
                f"Rerank score : "
                f"{source['rerank_score']:.4f}"
            )

            print(
                f"Frames : {source['frame_references']}"
            )

    # -----------------------------------------
    # Process user query
    # -----------------------------------------

    def ask(self, question):

        if not question or not question.strip():
            return

        print("\n[1/5] VECTOR RETRIEVAL")

        retrieved_documents = (
            self.vector_retriever.retrieve(
                query=question,
                video_id=self.video_id
            )
        )

        print(
            f"Retrieved candidates: "
            f"{len(retrieved_documents)}"
        )

        if not retrieved_documents:

            print(
                "\nNo relevant records found "
                "in the selected video collection."
            )

            return

        print("\n[2/5] RERANKING")

        reranked_documents = self.reranker.rerank(
            query=question,
            documents=retrieved_documents,
            top_n=self.rerank_top_n
        )

        print(
            f"Reranked candidates: "
            f"{len(reranked_documents)}"
        )

        if not reranked_documents:

            print("\nReranker returned no results.")

            return

        print("\n[3/5] CONTEXT BUILDING")

        context, sources = (
            self.context_builder.build_context(
                reranked_documents
            )
        )

        print(
            f"Context records: {len(sources)}"
        )

        print("\n[4/5] PROMPT BUILDING")

        prompt_value = self.prompt_builder.build_prompt(
            context=context,
            question=question
        )

        print("\n[5/5] GENERATING ANSWER")

        answer = self.response_generator.generate(
            prompt_value
        )

        print("\n" + "=" * 55)

        print("                  ANSWER")

        print("=" * 55)

        print(answer)

        self.display_sources(sources)

        print("\n" + "=" * 55)

    # -----------------------------------------
    # Interactive terminal loop
    # -----------------------------------------

    def run(self):

        self.display_header()

        while True:

            try:

                question = input(
                    "\nYou: "
                ).strip()

                if question.lower() in (
                    "exit",
                    "quit",
                    "q"
                ):

                    print(
                        "\nClosing VideoRAG. Goodbye!"
                    )

                    break

                if not question:
                    continue

                self.ask(question)

            except KeyboardInterrupt:

                print(
                    "\n\nInterrupted. Closing VideoRAG."
                )

                break

            except Exception as error:

                print(
                    f"\nError: {type(error).__name__}: {error}"
                )

                print(
                    "Please check the error and try again."
                )


# -----------------------------------------
# Application entry point
# -----------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Terminal-based VideoRAG"
    )

    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Filter retrieval by video ID"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of vector candidates"
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of reranked records"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for answer generation"
    )

    args = parser.parse_args()

    if args.top_k < 1 or args.top_n < 1:
        parser.error("--top-k and --top-n must be positive")

    if args.top_n > args.top_k:
        parser.error("--top-n cannot exceed --top-k")

    try:

        chat = TerminalChat(
            video_id=args.video_id,
            retrieval_top_k=args.top_k,
            rerank_top_n=args.top_n,
            model=args.model
        )

        chat.run()

    except Exception as error:

        print(
            f"\nApplication failed: "
            f"{type(error).__name__}: {error}"
        )


if __name__ == "__main__":
    main()