import math

from foundry_local_sdk import Configuration, FoundryLocalManager


MODEL_ALIAS = "qwen3-embedding-0.6b"
EXPECTED_DIMENSION = 1024

QUERY_TASK = (
    "Given a legal AI governance question, retrieve relevant passages "
    "from authoritative AI governance and data protection documents."
)


def format_query(question: str) -> str:
    """
    Qwen3 embedding models benefit from a task instruction on queries.
    Retrieval documents are embedded without this instruction.
    """
    return f"Instruct: {QUERY_TASK}\nQuery: {question}"


def cosine_similarity(
    first_vector: list[float],
    second_vector: list[float],
) -> float:
    """
    Compute cosine similarity without adding a NumPy dependency.
    """
    dot_product = sum(
        first * second
        for first, second in zip(
            first_vector,
            second_vector,
            strict=True,
        )
    )

    first_norm = math.sqrt(
        sum(value * value for value in first_vector)
    )
    second_norm = math.sqrt(
        sum(value * value for value in second_vector)
    )

    if first_norm == 0 or second_norm == 0:
        raise ValueError("Embedding vector must not have zero magnitude.")

    return dot_product / (first_norm * second_norm)


def main():
    question = (
        "What risk management duties apply to providers "
        "of high-risk AI systems?"
    )

    relevant_passage = (
        "Providers of high-risk AI systems shall establish, implement, "
        "document and maintain a risk management system."
    )

    unrelated_passage = (
        "Personal data shall be processed lawfully, fairly and in a "
        "transparent manner in relation to the data subject."
    )

    config = Configuration(app_name="govrag_embedding_smoke_test")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    model = manager.catalog.get_model(MODEL_ALIAS)

    print(f"Requested model alias: {MODEL_ALIAS}")
    print(f"Selected model alias:  {model.alias}")
    print(f"Selected model ID:     {model.id}")

    model.download(
        lambda progress: print(
            f"\rDownloading model: {progress:.2f}%",
            end="",
            flush=True,
        )
    )
    print()

    model.load()
    print("Model loaded.")

    try:
        client = model.get_embedding_client()

        response = client.generate_embeddings(
            [
                format_query(question),
                relevant_passage,
                unrelated_passage,
            ]
        )

        embeddings = [
            item.embedding
            for item in response.data
        ]

        if len(embeddings) != 3:
            raise ValueError(
                f"Expected 3 embeddings, received {len(embeddings)}."
            )

        dimensions = {
            len(embedding)
            for embedding in embeddings
        }

        if dimensions != {EXPECTED_DIMENSION}:
            raise ValueError(
                f"Expected {EXPECTED_DIMENSION} dimensions, "
                f"received {sorted(dimensions)}."
            )

        relevant_score = cosine_similarity(
            embeddings[0],
            embeddings[1],
        )
        unrelated_score = cosine_similarity(
            embeddings[0],
            embeddings[2],
        )

        print(f"Embedding count:       {len(embeddings)}")
        print(f"Embedding dimension:   {len(embeddings[0])}")
        print(f"Relevant similarity:   {relevant_score:.4f}")
        print(f"Unrelated similarity:  {unrelated_score:.4f}")

        if relevant_score <= unrelated_score:
            raise ValueError(
                "Semantic smoke test failed: the relevant passage "
                "did not score above the unrelated passage."
            )

        print("Semantic embedding smoke test passed.")

    finally:
        model.unload()
        print("Model unloaded.")


if __name__ == "__main__":
    main()
