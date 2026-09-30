# ======================================================================
# CÁLCULO DE SIMILITUD DEL COSENO ENTRE TRES TEXTOS
# ======================================================================
# Este script permite ingresar tres textos desde la consola y calcula la
# similitud del coseno entre ellos, comparando cada par.
#
# FUNCIONALIDAD:
#   - Pide tres textos al usuario.
#   - Genera los embeddings de cada texto utilizando un modelo HuggingFace.
#   - Calcula la similitud del coseno entre cada par de textos.
#   - Muestra una matriz de similitud (3x3) con los resultados.
#
# CONFIGURACIÓN:
#   - Puedes cambiar el modelo de embeddings, el dispositivo (CPU/GPU)
#     y la normalización directamente en el bloque de parámetros.
#
#
# ======================================================================

import os
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

# ----------------------------------------------------------------------
# DESACTIVAR TENSORFLOW
# ----------------------------------------------------------------------
# Estas variables de entorno garantizan que HuggingFace use PyTorch
# en lugar de intentar inicializar TensorFlow.
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"


# ----------------------------------------------------------------------
# CONFIGURACIÓN DEL MODELO DE EMBEDDINGS
# ----------------------------------------------------------------------
# Aquí puedes ajustar el modelo, el dispositivo y la normalización.
# - model_name: nombre del modelo de sentence-transformers.
# - device: "cpu" o "cuda" (si tienes GPU disponible).
# - normalize_embeddings: normaliza los vectores (recomendado para coseno).
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cpu"
NORMALIZE = True


# ----------------------------------------------------------------------
# FUNCIÓN: CALCULAR SIMILITUD DEL COSENO
# ----------------------------------------------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcula la similitud del coseno entre dos vectores numéricos.

    Fórmula:
        cos(θ) = (a · b) / (||a|| * ||b||)

    Retorna:
        float: valor entre -1 y 1
               1 -> vectores idénticos o muy similares
               0 -> sin relación
              -1 -> opuestos
    """
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12  # evita división por cero
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ----------------------------------------------------------------------
# FUNCIÓN PRINCIPAL
# ----------------------------------------------------------------------
def main() -> None:
    """
    Permite al usuario ingresar tres textos y calcula la similitud del coseno
    entre todos los pares (T1-T2, T1-T3, T2-T3).
    """

    print("=" * 70)
    print("CÁLCULO DE SIMILITUD DEL COSENO ENTRE TRES TEXTOS")
    print("=" * 70)
    print("Ingrese tres textos diferentes para comparar su similitud semántica.\n")

    # --------------------------------------------------------------
    # ENTRADA DE TEXTOS
    # --------------------------------------------------------------
    text1 = input("Texto 1: ").strip()
    text2 = input("Texto 2: ").strip()
    text3 = input("Texto 3: ").strip()

    textos = [text1, text2, text3]

    # --------------------------------------------------------------
    # INICIALIZACIÓN DEL MODELO DE EMBEDDINGS
    # --------------------------------------------------------------
    print("\nCargando modelo de embeddings...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": NORMALIZE}
    )

    # --------------------------------------------------------------
    # GENERACIÓN DE EMBEDDINGS PARA CADA TEXTO
    # --------------------------------------------------------------
    print("Generando embeddings de los textos...")
    vectores = np.array(embeddings_model.embed_documents(textos), dtype=np.float32)

    # vectores es una matriz de tamaño (3, n_dim)
    # donde n_dim depende del modelo, por ejemplo 384 para all-MiniLM-L6-v2

    # --------------------------------------------------------------
    # CÁLCULO DE SIMILITUD DEL COSENO ENTRE TODOS LOS PARES
    # --------------------------------------------------------------
    print("\nCalculando similitudes del coseno:\n")

    n = len(vectores)
    matriz_sim = np.zeros((n, n), dtype=float)

    # Calcula la similitud para cada par (i, j)
    for i in range(n):
        for j in range(n):
            matriz_sim[i, j] = cosine_similarity(vectores[i], vectores[j])

    # --------------------------------------------------------------
    # PRESENTACIÓN DE RESULTADOS
    # --------------------------------------------------------------
    print("MATRIZ DE SIMILITUD DEL COSENO (valores entre -1 y 1):\n")
    print("        T1        T2        T3")
    for i in range(n):
        fila = f"T{i+1}  " + "  ".join(f"{matriz_sim[i,j]:8.4f}" for j in range(n))
        print(fila)

    print("\nINTERPRETACIÓN:")
    print("  - Valores cercanos a 1 indican alta similitud semántica.")
    print("  - Valores cercanos a 0 indican baja o nula similitud.")
    print("  - La diagonal (T1-T1, T2-T2, T3-T3) siempre será 1.0.")
    print("=" * 70)


# ----------------------------------------------------------------------
# BLOQUE DE EJECUCIÓN DIRECTA
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
