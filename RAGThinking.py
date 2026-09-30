from __future__ import annotations

import os
import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import deeplake
import numpy as np
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

warnings.filterwarnings("ignore")

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

EXIT_COMMANDS = {"salir", "exit", "quit"}

# Número máximo de interacciones guardadas en memoria.
MAX_HISTORY_TURNS = 10

# Pausa entre llamadas al modelo.
INTERNAL_DELAY_SECONDS = 1

# Mostrar u ocultar análisis y planificación internos.
SHOW_INTERNAL_STEPS = True


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

ENV_FILE = PROJECT_ROOT / "secrets" / ".env"
CONFIG_FILE = PROJECT_ROOT / "config.json"


# ============================================================
# CARGAR CONFIGURACIÓN
# ============================================================

def load_config(path: str | Path) -> Dict:
    """
    Carga el archivo config.json.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de configuración: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# CONVERSIÓN DE RESPUESTAS DEL LLM
# ============================================================

def response_to_text(response: Any) -> str:
    """
    Convierte diferentes formatos de respuesta de LangChain
    en texto plano.
    """

    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):

        parts: List[str] = []

        for block in content:

            if isinstance(block, str):
                parts.append(block)

            elif isinstance(block, dict):

                text = (
                    block.get("text")
                    or block.get("content")
                )

                if text:
                    parts.append(str(text))

            elif hasattr(block, "text"):
                parts.append(str(block.text))

            else:
                parts.append(str(block))

        return "\n".join(parts).strip()

    if isinstance(content, dict):

        return str(
            content.get("text")
            or content.get("content")
            or content
        ).strip()

    return str(content).strip()


# ============================================================
# MEMORIA CONVERSACIONAL
# ============================================================

def format_history(
    history: List[Dict[str, str]]
) -> str:
    """
    Convierte el historial reciente del chat en texto.
    """

    if not history:
        return "No hay conversaciones anteriores."

    recent_history = history[-MAX_HISTORY_TURNS:]

    formatted_turns = []

    for turn_number, turn in enumerate(
        recent_history,
        start=1
    ):

        formatted_turns.append(
            f"Interacción {turn_number}\n"
            f"Usuario: {turn['user']}\n"
            f"Asistente: {turn['assistant']}"
        )

    return "\n\n".join(formatted_turns)


# ============================================================
# MOSTRAR INTERACCIONES INTERNAS
# ============================================================

def print_internal_step(
    title: str,
    content: str
) -> None:

    if not SHOW_INTERNAL_STEPS:
        return

    separator = "=" * 70

    print(f"\n{separator}")
    print(title)
    print(separator)
    print(content)
    print(separator)


# ============================================================
# SIMILITUD DEL COSENO
# ============================================================

def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray
) -> float:
    """
    Calcula la similitud del coseno.
    """

    denom = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    ) + 1e-12

    if denom == 0:
        return 0.0

    return float(
        np.dot(a, b) / denom
    )


# ============================================================
# CARGAR RAG
# ============================================================

def load_rag_resources(
    cfg: Dict
):
    """
    Carga una sola vez:

    - Modelo de embeddings.
    - Base de datos Deep Lake.
    - Embeddings almacenados.
    - Textos almacenados.

    Evita volver a cargar la base en cada pregunta.
    """

    dl_cfg = cfg["deeplake"]
    emb_cfg = cfg["embedding"]

    # --------------------------------------------------------
    # Resolver ruta de Deep Lake
    # --------------------------------------------------------

    dataset_path = Path(
        dl_cfg["dataset_path"]
    )

    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    dataset_path = (
        dataset_path
        .expanduser()
        .resolve()
    )

    if not dataset_path.is_dir():
        raise FileNotFoundError(
            f"No se encontró la base Deep Lake en: "
            f"{dataset_path}"
        )

    print(
        f"\n📚 Cargando base RAG desde:\n"
        f"{dataset_path}"
    )

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name=emb_cfg["model_name"],
        model_kwargs={
            "device": emb_cfg.get(
                "device",
                "cpu"
            )
        },
        encode_kwargs={
            "normalize_embeddings": bool(
                emb_cfg.get(
                    "normalize_embeddings",
                    True
                )
            )
        }
    )

    # --------------------------------------------------------
    # Deep Lake
    # --------------------------------------------------------

    dataset = deeplake.load(
        str(dataset_path),
        read_only=bool(
            dl_cfg.get(
                "read_only",
                True
            )
        ),
        verbose=False,
    )

    dataset.checkout("main")

    # --------------------------------------------------------
    # Leer vectores y textos una sola vez
    # --------------------------------------------------------

    document_vectors = np.asarray(
        dataset["embedding"].numpy(),
        dtype=np.float32
    )

    texts = dataset["text"].data(
        aslist=True
    )["value"]

    texts = [
        str(text)
        for text in texts
    ]

    if len(texts) == 0:
        raise ValueError(
            "La base Deep Lake no contiene documentos."
        )

    if len(document_vectors) != len(texts):
        raise ValueError(
            "El número de embeddings no coincide "
            "con el número de textos."
        )

    print(
        f"✅ RAG cargado correctamente: "
        f"{len(texts)} fragmentos.\n"
    )

    return (
        embeddings,
        document_vectors,
        texts
    )


# ============================================================
# RECUPERACIÓN DESDE DEEP LAKE
# ============================================================

def retrieve_context(
    cfg: Dict,
    query: str,
    embeddings,
    document_vectors: np.ndarray,
    texts: List[str]
) -> Tuple[str, List[Tuple[int, float]]]:
    """
    Recupera los K fragmentos más similares a la consulta.
    """

    retrieval_cfg = cfg.get(
        "retrieval",
        {}
    )

    k = int(
        retrieval_cfg.get(
            "k",
            5
        )
    )

    max_chars = int(
        retrieval_cfg.get(
            "max_chars_per_chunk",
            4000
        )
    )

    if k <= 0:
        raise ValueError(
            "retrieval.k debe ser mayor que cero."
        )

    # --------------------------------------------------------
    # Embedding de la consulta
    # --------------------------------------------------------

    query_vector = np.asarray(
        embeddings.embed_query(query),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Similitud
    # --------------------------------------------------------

    scores = np.array(
        [
            cosine_similarity(
                query_vector,
                document_vector
            )
            for document_vector
            in document_vectors
        ],
        dtype=np.float32
    )

    top_indices = np.argsort(
        scores
    )[::-1][
        :min(k, len(scores))
    ]

    results = []

    context_parts = []

    print(
        f"\n🔎 Top {len(top_indices)} "
        f"resultados del RAG:"
    )

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        index = int(index)
        score = float(scores[index])

        results.append(
            (index, score)
        )

        text = texts[index]

        if max_chars > 0:
            text = text[:max_chars]

        context_parts.append(
            f"""
DOCUMENTO {rank}
Similitud del coseno: {score:.4f}

{text}
""".strip()
        )

        print(
            f"[{rank}] "
            f"índice={index} "
            f"cos={score:.4f}"
        )

    print("-" * 80)

    rag_context = "\n\n" + (
        "\n\n" + "-" * 70 + "\n\n"
    ).join(context_parts)

    return rag_context, results


# ============================================================
# METAPROMPT GENERAL
# ============================================================

META_PROMPT = """
Actúa como un asistente con capacidad de planificación y consulta
de una base de conocimiento RAG.

Tu objetivo es comprender correctamente la solicitud del usuario,
consultar la información disponible y construir una respuesta
fundamentada.

Debes:

1. Identificar el objetivo principal del usuario.
2. Separar el problema en pasos manejables.
3. Detectar restricciones, condiciones y dependencias.
4. Revisar si existe suficiente información para responder.
5. Utilizar el contexto recuperado del RAG cuando sea relevante.
6. No afirmar que el RAG contiene información que realmente no contiene.
7. Evitar inventar información.
8. Solicitar contexto adicional solamente cuando sea indispensable.
9. Proporcionar respuestas claras, prácticas y orientadas a la acción.

El contexto recuperado puede contener información irrelevante.
Debes evaluar su utilidad antes de utilizarlo.
""".strip()


# ============================================================
# INTERACCIÓN INTERNA 1
# ANÁLISIS DE LA SOLICITUD
# ============================================================

def analyse_request(
    llm,
    user_input: str,
    conversation_history: str,
    system_instruction: str
) -> str:
    """
    Primera interacción del modelo.

    Analiza la solicitud y genera además una consulta optimizada
    para buscar información en el RAG.
    """

    system_prompt = f"""
{META_PROMPT}

INSTRUCCIÓN GENERAL DEL SISTEMA:

{system_instruction}

Eres el primer módulo interno del asistente.

Analiza la solicitud del usuario.

Todavía NO tienes el contexto recuperado desde el RAG,
por lo que no debes redactar una respuesta final.

Genera exactamente esta estructura:

OBJETIVO DEL USUARIO:
Describe brevemente qué quiere conseguir.

INFORMACIÓN DISPONIBLE:
Indica los datos que ya están disponibles.

POSIBLES AMBIGÜEDADES:
Indica posibles ambigüedades.
Si no existen, escribe: Ninguna.

DESCOMPOSICIÓN DEL PROBLEMA:
Divide el problema en componentes manejables.

CONSULTA_RAG:
Escribe una única consulta breve y semánticamente rica
que permita recuperar del RAG la información necesaria.

RESULTADO ESPERADO:
Describe qué debería contener una buena respuesta.

Reglas:

- No respondas directamente al usuario.
- No inventes información.
- No escribas todavía la respuesta final.
- CONSULTA_RAG debe ocupar una sola línea.
""".strip()

    user_prompt = f"""
HISTORIAL RECIENTE:

{conversation_history}


NUEVA SOLICITUD:

{user_input}


Realiza el análisis estructurado.
""".strip()

    response = llm.invoke(
        [
            SystemMessage(
                content=system_prompt
            ),
            HumanMessage(
                content=user_prompt
            )
        ]
    )

    return response_to_text(response)


# ============================================================
# EXTRAER CONSULTA PARA EL RAG
# ============================================================

def extract_rag_query(
    first_analysis: str,
    original_query: str
) -> str:
    """
    Extrae la línea CONSULTA_RAG generada por el primer módulo.

    Si no puede encontrarla, utiliza la consulta original.
    """

    lines = first_analysis.splitlines()

    for i, line in enumerate(lines):

        if line.strip().upper().startswith(
            "CONSULTA_RAG:"
        ):

            query = line.split(
                ":",
                1
            )[1].strip()

            # Si la consulta aparece en la siguiente línea
            if not query and i + 1 < len(lines):
                query = lines[i + 1].strip()

            if query:
                return query

    return original_query


# ============================================================
# INTERACCIÓN INTERNA 2
# REVISIÓN DEL RAG Y PLANIFICACIÓN
# ============================================================

def review_context_and_plan(
    llm,
    user_input: str,
    first_analysis: str,
    rag_context: str,
    conversation_history: str,
    system_instruction: str
) -> str:
    """
    Segunda interacción.

    Evalúa el análisis inicial y la información recuperada
    desde Deep Lake.
    """

    system_prompt = f"""
{META_PROMPT}

INSTRUCCIÓN GENERAL DEL SISTEMA:

{system_instruction}

Eres el segundo módulo interno del asistente.

Tu función es revisar:

1. La solicitud del usuario.
2. El análisis del primer módulo.
3. El historial reciente.
4. La información recuperada desde el RAG.

Debes determinar si la información disponible permite
generar una respuesta adecuada.

Genera exactamente esta estructura:

ESTADO DEL CONTEXTO:
Escribe una de estas opciones:

- SUFICIENTE
- PARCIAL
- FALTA_CONTEXTO

VERIFICACIÓN:
Explica brevemente qué información útil aporta el RAG
y qué información no está respaldada por él.

INFORMACIÓN RELEVANTE DEL RAG:
Resume únicamente la información recuperada que sea útil
para responder.

CONTEXTO FALTANTE:
Si falta información indispensable, enumérala.
En caso contrario escribe: Ninguno.

PLAN DE RESPUESTA:
Presenta entre 2 y 6 pasos concretos para elaborar
la respuesta final.

INDICACIÓN PARA LA RESPUESTA FINAL:
Escribe una de estas opciones:

- Responder directamente.
- Responder indicando las limitaciones del contexto.
- Solicitar información adicional al usuario.

Reglas:

- No redactes todavía la respuesta final.
- No inventes información.
- Ignora fragmentos del RAG que no sean relevantes.
- Diferencia claramente lo que aparece en el RAG de aquello
  que no está respaldado por el contexto.
""".strip()

    user_prompt = f"""
HISTORIAL RECIENTE:

{conversation_history}


SOLICITUD ACTUAL:

{user_input}


ANÁLISIS DEL PRIMER MÓDULO:

{first_analysis}


============================================================
CONTEXTO RECUPERADO DESDE EL RAG
============================================================

{rag_context}

============================================================


Revisa la información disponible y construye
el plan de respuesta.
""".strip()

    response = llm.invoke(
        [
            SystemMessage(
                content=system_prompt
            ),
            HumanMessage(
                content=user_prompt
            )
        ]
    )

    return response_to_text(response)


# ============================================================
# INTERACCIÓN 3
# RESPUESTA FINAL
# ============================================================

def generate_final_response(
    llm,
    user_input: str,
    first_analysis: str,
    context_review: str,
    rag_context: str,
    conversation_history: str,
    system_instruction: str
) -> str:
    """
    Tercera interacción.

    Construye la respuesta final utilizando el análisis,
    la planificación y el contexto recuperado.
    """

    system_prompt = f"""
{META_PROMPT}

INSTRUCCIÓN GENERAL DEL SISTEMA:

{system_instruction}

Eres el módulo encargado de generar la respuesta final
para el usuario.

Dispones de:

1. La solicitud original.
2. El historial reciente.
3. El análisis de la solicitud.
4. Información recuperada desde el RAG.
5. Una revisión crítica de esa información.
6. Un plan de respuesta.

Debes seguir estas reglas:

- Sigue el plan construido por el módulo anterior.
- Utiliza el contexto del RAG cuando sea relevante.
- No utilices como cierta información que no esté respaldada
  por el RAG cuando la respuesta dependa de los documentos.
- Si el contexto es parcial, responde hasta donde permita
  la información disponible.
- Si falta información indispensable, solicita únicamente
  esa información.
- No inventes datos.
- No menciones los módulos internos.
- No menciones que realizaste varias llamadas al modelo.
- No muestres el análisis interno.
- No muestres el plan interno.
- No incluyas etiquetas como OBJETIVO, VERIFICACIÓN,
  CONSULTA_RAG o PLAN.
- Entrega únicamente la respuesta destinada al usuario.
""".strip()

    user_prompt = f"""
HISTORIAL RECIENTE:

{conversation_history}


SOLICITUD ORIGINAL:

{user_input}


============================================================
CONTEXTO RECUPERADO DESDE EL RAG
============================================================

{rag_context}

============================================================


ANÁLISIS INICIAL:

{first_analysis}


REVISIÓN DEL CONTEXTO Y PLAN:

{context_review}


Genera ahora la respuesta final para el usuario.
""".strip()

    response = llm.invoke(
        [
            SystemMessage(
                content=system_prompt
            ),
            HumanMessage(
                content=user_prompt
            )
        ]
    )

    return response_to_text(response)


# ============================================================
# CHAT PRINCIPAL
# ============================================================

def chat_with_deep_rag(
    cfg: Dict
) -> None:
    """
    Chat RAG con planificación profunda.

    Flujo:

    Usuario
       ↓
    Análisis de la solicitud
       ↓
    Generación de consulta RAG
       ↓
    Búsqueda en Deep Lake
       ↓
    Revisión crítica del contexto
       ↓
    Plan de respuesta
       ↓
    Respuesta final
    """

    # --------------------------------------------------------
    # Variables de entorno
    # --------------------------------------------------------

    load_dotenv(ENV_FILE)

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "No se encontró OPENAI_API_KEY "
            "en el archivo .env"
        )

    # --------------------------------------------------------
    # Configurar LLM
    # --------------------------------------------------------

    llm_cfg = cfg["llm"]

    llm = ChatOpenAI(
        openai_api_base=llm_cfg["api_base"],
        openai_api_key=api_key,
        model_name=llm_cfg["model_name"],
        temperature=float(
            llm_cfg.get(
                "temperature",
                0.4
            )
        )
    )

    print(
        f"\n🤖 Modelo utilizado: "
        f"{llm_cfg['model_name']}"
    )

    # --------------------------------------------------------
    # Instrucción general del RAG
    # --------------------------------------------------------

    system_instruction = cfg.get(
        "system_instruction",
        "Responde de forma clara, precisa "
        "y basándote en el contexto disponible."
    )

    # --------------------------------------------------------
    # Cargar Deep Lake UNA VEZ
    # --------------------------------------------------------

    (
        embeddings,
        document_vectors,
        texts
    ) = load_rag_resources(cfg)

    # --------------------------------------------------------
    # Memoria
    # --------------------------------------------------------

    history: List[Dict[str, str]] = []

    print(
        "\n"
        "🧠 RAG con pensamiento profundo listo.\n"
        "Escribe 'salir' para terminar.\n"
    )

    # ========================================================
    # BUCLE PRINCIPAL
    # ========================================================

    while True:

        user_input = input(
            "👤 Usuario: "
        ).strip()

        if not user_input:
            continue

        if user_input.lower() in EXIT_COMMANDS:
            print(
                "\n👋 Fin de la sesión."
            )
            break

        try:

            conversation_history = (
                format_history(history)
            )

            # =================================================
            # 1. ANÁLISIS INICIAL
            # =================================================

            first_analysis = analyse_request(
                llm=llm,
                user_input=user_input,
                conversation_history=conversation_history,
                system_instruction=system_instruction
            )

            print_internal_step(
                title=(
                    "🧠 INTERACCIÓN INTERNA 1: "
                    "ANÁLISIS DE LA SOLICITUD"
                ),
                content=first_analysis
            )

            time.sleep(
                INTERNAL_DELAY_SECONDS
            )

            # =================================================
            # 2. EXTRAER CONSULTA OPTIMIZADA PARA RAG
            # =================================================

            rag_query = extract_rag_query(
                first_analysis=first_analysis,
                original_query=user_input
            )

            if SHOW_INTERNAL_STEPS:
                print(
                    "\n🔎 Consulta generada "
                    "para el RAG:"
                )
                print(rag_query)

            # =================================================
            # 3. RECUPERAR INFORMACIÓN
            # =================================================

            rag_context, retrieval_results = (
                retrieve_context(
                    cfg=cfg,
                    query=rag_query,
                    embeddings=embeddings,
                    document_vectors=document_vectors,
                    texts=texts
                )
            )

            if SHOW_INTERNAL_STEPS:

                print_internal_step(
                    title=(
                        "📚 CONTEXTO RECUPERADO "
                        "DESDE DEEP LAKE"
                    ),
                    content=rag_context
                )

            time.sleep(
                INTERNAL_DELAY_SECONDS
            )

            # =================================================
            # 4. REVISIÓN Y PLANIFICACIÓN
            # =================================================

            context_review = (
                review_context_and_plan(
                    llm=llm,
                    user_input=user_input,
                    first_analysis=first_analysis,
                    rag_context=rag_context,
                    conversation_history=conversation_history,
                    system_instruction=system_instruction
                )
            )

            print_internal_step(
                title=(
                    "🔍 INTERACCIÓN INTERNA 2: "
                    "REVISIÓN DEL RAG Y PLAN"
                ),
                content=context_review
            )

            time.sleep(
                INTERNAL_DELAY_SECONDS
            )

            # =================================================
            # 5. RESPUESTA FINAL
            # =================================================

            final_response = (
                generate_final_response(
                    llm=llm,
                    user_input=user_input,
                    first_analysis=first_analysis,
                    context_review=context_review,
                    rag_context=rag_context,
                    conversation_history=conversation_history,
                    system_instruction=system_instruction
                )
            )

            print(
                "\n" + "=" * 70
            )
            print(
                "🤖 RESPUESTA FINAL"
            )
            print(
                "=" * 70
            )
            print(
                final_response
            )
            print(
                "=" * 70 + "\n"
            )

            # =================================================
            # 6. GUARDAR EN MEMORIA
            # =================================================

            history.append(
                {
                    "user": user_input,
                    "assistant": final_response
                }
            )

            if (
                len(history)
                > MAX_HISTORY_TURNS
            ):
                history = history[
                    -MAX_HISTORY_TURNS:
                ]

        except KeyboardInterrupt:

            print(
                "\n\n👋 Conversación terminada."
            )
            break

        except Exception as error:

            print(
                f"\n❌ Error al procesar "
                f"la solicitud: {error}\n"
            )


# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":

    cfg = load_config(
        CONFIG_FILE
    )

    chat_with_deep_rag(
        cfg
    )