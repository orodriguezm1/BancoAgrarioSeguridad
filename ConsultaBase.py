import os
import json
import warnings
from pathlib import Path
from typing import Dict, Tuple

import deeplake
import numpy as np

warnings.filterwarnings("ignore")
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(path: str | Path) -> Dict:
    """Carga el archivo config.json."""
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_vectorstore(cfg: Dict) -> Tuple[object, HuggingFaceEmbeddings]:
    """Carga la base DeepLake local y el modelo de embeddings."""
    dl_cfg = cfg["deeplake"]
    emb_cfg = cfg["embedding"]

    dataset_path = Path(dl_cfg["dataset_path"]).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    dataset_path = dataset_path.resolve()

    if not dataset_path.is_dir():
        raise FileNotFoundError(
            f"No se encontró la base DeepLake en: {dataset_path}. "
            "Ejecuta TensorialBase.py para crearla."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=emb_cfg["model_name"],
        model_kwargs={"device": emb_cfg.get("device", "cpu")},
        encode_kwargs={
            "normalize_embeddings": bool(emb_cfg.get("normalize_embeddings", True))
        },
    )

    dataset = deeplake.load(
        str(dataset_path),
        read_only=bool(dl_cfg.get("read_only", True)),
        verbose=False,
    )
    dataset.checkout("main")

    required_tensors = {"text", "metadata", "embedding"}
    missing_tensors = required_tensors.difference(dataset.tensors)
    if missing_tensors:
        missing = ", ".join(sorted(missing_tensors))
        raise ValueError(f"La base DeepLake no contiene los tensores: {missing}")

    return dataset, embeddings


def similarity_search_with_score(
    dataset: object,
    embeddings: HuggingFaceEmbeddings,
    query: str,
    k: int,
) -> list[Tuple[Document, float]]:
    """Busca los documentos más cercanos usando los embeddings ya almacenados."""
    if k <= 0:
        raise ValueError("retrieval.k debe ser mayor que cero")

    query_vector = np.asarray(embeddings.embed_query(query), dtype=np.float32)
    document_vectors = np.asarray(dataset["embedding"].numpy(), dtype=np.float32)
    document_vectors = document_vectors / (
        np.linalg.norm(document_vectors, axis=1, keepdims=True) + 1e-12
    )
    query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-12)
    scores = document_vectors @ query_vector

    texts = dataset["text"].data(aslist=True)["value"]
    metadata = dataset["metadata"].data(aslist=True)["value"]
    top_indices = np.argsort(scores)[::-1][: min(k, len(scores))]

    return [
        (
            Document(
                page_content=str(texts[index]),
                metadata=metadata[index] if isinstance(metadata[index], dict) else {},
            ),
            float(scores[index]),
        )
        for index in top_indices
    ]


def mostrar_chunks_recuperados(cfg: Dict, query: str):
    """Muestra los k chunks más similares al prompt del usuario."""
    k = int(cfg["retrieval"].get("k", 5))

    dataset, embeddings = load_vectorstore(cfg)

    results = similarity_search_with_score(dataset, embeddings, query, k)

    print("\n" + "=" * 100)
    print(f"Prompt del usuario:\n{query}")
    print("=" * 100)
    print(f"\nTop {k} chunks recuperados:\n")

    for i, (doc, score) in enumerate(results, start=1):
        print("-" * 100)
        print(f"Chunk #{i}")
        print(f"Score / similitud: {score}")
        print("-" * 100)

        print(doc.page_content)

        if doc.metadata:
            print("\nMetadata:")
            for key, value in doc.metadata.items():
                print(f"  {key}: {value}")

        print()

    print("=" * 100)


if __name__ == "__main__":
    config_file = PROJECT_ROOT / "config.json"
    cfg = load_config(config_file)

    print("\nBuscador de chunks DeepLake")
    print("Escribe 'salir' para terminar.\n")

    while True:
        user_prompt = input("Prompt del usuario: ").strip()

        if user_prompt.lower() in {"salir", "exit", "quit"}:
            print("Fin.")
            break

        if not user_prompt:
            continue

        mostrar_chunks_recuperados(cfg, user_prompt)
