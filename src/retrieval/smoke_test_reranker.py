import numpy as np
import torch
from sentence_transformers import CrossEncoder


MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print(f"Loading reranker: {MODEL_ID}")
    print(f"Requested device: cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    torch.cuda.reset_peak_memory_stats()

    model = CrossEncoder(
        MODEL_ID,
        device="cuda",
    )

    model_device = next(model.model.parameters()).device
    print(f"Actual model device: {model_device}")

    question = (
        "When is a data protection impact assessment "
        "required under the GDPR?"
    )

    candidates = [
        {
            "label": "GDPR Article 35",
            "text": (
                "Where a type of processing, in particular using new "
                "technologies, is likely to result in a high risk to "
                "the rights and freedoms of natural persons, the "
                "controller shall, prior to the processing, carry out "
                "a data protection impact assessment."
            ),
        },
        {
            "label": "GDPR Recital 90",
            "text": (
                "A data protection impact assessment should be carried "
                "out by the controller prior to processing in order to "
                "assess the likelihood and severity of the high risk."
            ),
        },
        {
            "label": "Unrelated AI Act passage",
            "text": (
                "Providers of general-purpose AI models must prepare "
                "and maintain technical documentation concerning the "
                "model and provide information to downstream providers."
            ),
        },
    ]

    sentence_pairs = [
        (question, candidate["text"])
        for candidate in candidates
    ]

    scores = model.predict(
        sentence_pairs,
        batch_size=3,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    torch.cuda.synchronize()

    score_values = np.asarray(scores).reshape(-1).tolist()

    ranked_results = sorted(
        zip(candidates, score_values),
        key=lambda item: item[1],
        reverse=True,
    )

    print()
    print("Reranker results:")

    for rank, (candidate, score) in enumerate(
        ranked_results,
        start=1,
    ):
        print(
            f"{rank}. {candidate['label']} | "
            f"score={score:.4f}"
        )

    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024
    )

    print()
    print(f"Peak CUDA memory: {peak_memory_mb:.2f} MB")

    relevant_scores = score_values[:2]
    unrelated_score = score_values[2]

    if max(relevant_scores) <= unrelated_score:
        raise RuntimeError(
            "Reranker failed to rank a relevant GDPR "
            "passage above the unrelated passage."
        )

    if model_device.type != "cuda":
        raise RuntimeError(
            f"Reranker is running on {model_device}, not CUDA."
        )

    print("Cross-Encoder GPU smoke test passed.")


if __name__ == "__main__":
    main()