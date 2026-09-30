from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

warnings.filterwarnings("ignore")

EXIT_COMMANDS = {"salir", "exit", "quit"}

# Número máximo de turnos almacenados en memoria.
MAX_HISTORY_TURNS = 10

# Pausa entre llamadas para evitar saturar modelos gratuitos.
INTERNAL_DELAY_SECONDS = 1

# Permite mostrar u ocultar las interacciones internas.
SHOW_INTERNAL_STEPS = True


# ============================================================
# CARGAR VARIABLES DE ENTORNO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / "secrets" / ".env")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "❌ No se encontró OPENAI_API_KEY en el archivo .env"
    )


# ============================================================
# CONFIGURACIÓN DEL MODELO
# ============================================================

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    model="nvidia/nemotron-3-ultra-550b-a55b:free",
    temperature=0.4,
)


# ============================================================
# METAPROMPT PRINCIPAL
# ============================================================

META_PROMPT = """
Actúa como un asistente planeador.

Tu objetivo es comprender la solicitud del usuario, identificar qué
resultado quiere obtener y construir un plan adecuado antes de responder.

Debes:

1. Identificar el objetivo principal del usuario.
2. Separar el problema en pasos manejables.
3. Detectar restricciones, condiciones y dependencias.
4. Revisar si existe suficiente contexto para responder.
5. Evitar inventar información.
6. Solicitar contexto adicional solamente cuando sea indispensable.
7. Proporcionar respuestas claras, prácticas y orientadas a la acción.

No eres experto en un área específica. Debes adaptar tu planificación
al problema planteado por el usuario.
""".strip()


# ============================================================
# FUNCIONES AUXILIARES
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
        parts: list[str] = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)

            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")

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


def format_history(history: list[dict[str, str]]) -> str:
    """
    Convierte la memoria reciente del chat en texto.
    """

    if not history:
        return "No hay conversaciones anteriores."

    recent_history = history[-MAX_HISTORY_TURNS:]

    formatted_turns: list[str] = []

    for turn_number, turn in enumerate(recent_history, start=1):
        formatted_turns.append(
            f"Interacción {turn_number}\n"
            f"Usuario: {turn['user']}\n"
            f"Asistente: {turn['assistant']}"
        )

    return "\n\n".join(formatted_turns)


def print_internal_step(title: str, content: str) -> None:
    """
    Imprime de forma clara una interacción interna.
    """

    if not SHOW_INTERNAL_STEPS:
        return

    separator = "=" * 70

    print(f"\n{separator}")
    print(title)
    print(separator)
    print(content)
    print(separator)


# ============================================================
# PRIMERA INTERACCIÓN INTERNA
# ANÁLISIS DE LA SOLICITUD
# ============================================================

def analyse_request(
    user_input: str,
    conversation_history: str,
) -> str:
    """
    Analiza la solicitud del usuario.

    Esta es la primera interacción del modelo consigo mismo.
    """

    system_prompt = f"""
{META_PROMPT}

Eres el primer módulo interno del asistente.

Analiza la solicitud del usuario, pero no redactes todavía la respuesta
final.

Genera un resumen de análisis con esta estructura:

OBJETIVO DEL USUARIO:
Describe brevemente qué quiere conseguir.

INFORMACIÓN DISPONIBLE:
Enumera los datos y condiciones presentes en la solicitud.

POSIBLES AMBIGÜEDADES:
Indica qué aspectos podrían tener varias interpretaciones.

DESCOMPOSICIÓN DEL PROBLEMA:
Divide la solicitud en pasos o componentes manejables.

RESULTADO ESPERADO:
Explica qué debería contener una buena respuesta.

No respondas directamente al usuario.
No inventes información.
No escribas una respuesta final.
""".strip()

    user_prompt = f"""
Historial reciente:

{conversation_history}

Nueva solicitud del usuario:

{user_input}

Realiza el análisis estructurado de la solicitud.
""".strip()

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    return response_to_text(response)


# ============================================================
# SEGUNDA INTERACCIÓN INTERNA
# REVISIÓN DEL CONTEXTO Y CREACIÓN DEL PLAN
# ============================================================

def review_context_and_plan(
    user_input: str,
    first_analysis: str,
    conversation_history: str,
) -> str:
    """
    Revisa el análisis anterior, determina si falta contexto
    y construye el plan de respuesta.

    Esta es la segunda interacción del modelo consigo mismo.
    """

    system_prompt = f"""
{META_PROMPT}

Eres el segundo módulo interno del asistente.

Tu función es revisar el análisis elaborado por el primer módulo y
determinar si existe suficiente información para responder correctamente.

Debes generar una revisión con esta estructura exacta:

ESTADO DEL CONTEXTO:
Escribe únicamente una de estas opciones:

- SUFICIENTE
- FALTA_CONTEXTO

VERIFICACIÓN:
Explica brevemente por qué el contexto es suficiente o insuficiente.

CONTEXTO FALTANTE:
Si falta contexto, enumera únicamente la información indispensable.
Si no falta contexto, escribe: Ninguno.

PLAN DE RESPUESTA:
Presenta entre 2 y 6 pasos concretos para elaborar la respuesta final.

INDICACIÓN PARA LA RESPUESTA FINAL:
Indica una de las siguientes acciones:

- Responder directamente.
- Solicitar información adicional al usuario.

No redactes todavía la respuesta final.
No inventes datos.
No repitas innecesariamente la solicitud.
""".strip()

    user_prompt = f"""
Historial reciente:

{conversation_history}

Solicitud actual:

{user_input}

Análisis producido por el primer módulo:

{first_analysis}

Revisa el contexto y construye el plan de respuesta.
""".strip()

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    return response_to_text(response)


# ============================================================
# TERCERA INTERACCIÓN
# RESPUESTA FINAL PARA EL USUARIO
# ============================================================

def generate_final_response(
    user_input: str,
    first_analysis: str,
    context_review: str,
    conversation_history: str,
) -> str:
    """
    Genera la respuesta final basándose en las dos
    interacciones internas anteriores.
    """

    system_prompt = f"""
{META_PROMPT}

Eres el módulo que redacta la respuesta final para el usuario.

Dispones de:

1. La solicitud original.
2. Un análisis estructurado.
3. Una revisión del contexto.
4. Un plan de respuesta.

Sigue estas reglas:

- Si el estado del contexto es SUFICIENTE, responde directamente.
- Sigue el plan elaborado en la revisión.
- Si el estado es FALTA_CONTEXTO, solicita únicamente la información
  indispensable.
- Cuando falte información, formula preguntas concretas y fáciles
  de responder.
- No inventes datos.
- No menciones los módulos internos.
- No menciones que realizaste varias llamadas al modelo.
- No repitas el análisis interno.
- No incluyas etiquetas como OBJETIVO, VERIFICACIÓN o PLAN, salvo
  que sean útiles para responder.
- Entrega únicamente la respuesta destinada al usuario.
""".strip()

    user_prompt = f"""
Historial reciente:

{conversation_history}

Solicitud original del usuario:

{user_input}

Primer análisis:

{first_analysis}

Revisión del contexto y plan:

{context_review}

Genera ahora la respuesta final para el usuario.
""".strip()

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    return response_to_text(response)


# ============================================================
# BUCLE PRINCIPAL DEL CHAT
# ============================================================

def main() -> None:
    """
    Ejecuta el chatbot planeador.

    Por cada mensaje del usuario realiza:

    1. Análisis de la solicitud.
    2. Revisión del contexto y planificación.
    3. Respuesta final.
    """

    history: list[dict[str, str]] = []

    print(
        "\n💬 Chatbot planeador con interacciones internas\n"
        "Escribe 'salir' para terminar.\n"
    )

    while True:
        user_input = input("👤 Tú: ").strip()

        if not user_input:
            continue

        if user_input.lower() in EXIT_COMMANDS:
            print("\n👋 Hasta luego.")
            break

        try:
            conversation_history = format_history(history)

            # =================================================
            # INTERACCIÓN INTERNA 1
            # =================================================

            first_analysis = analyse_request(
                user_input=user_input,
                conversation_history=conversation_history,
            )

            print_internal_step(
                title="🧠 INTERACCIÓN INTERNA 1: ANÁLISIS",
                content=first_analysis,
            )

            time.sleep(INTERNAL_DELAY_SECONDS)

            # =================================================
            # INTERACCIÓN INTERNA 2
            # =================================================

            context_review = review_context_and_plan(
                user_input=user_input,
                first_analysis=first_analysis,
                conversation_history=conversation_history,
            )

            print_internal_step(
                title="🔍 INTERACCIÓN INTERNA 2: REVISIÓN Y PLAN",
                content=context_review,
            )

            time.sleep(INTERNAL_DELAY_SECONDS)

            # =================================================
            # RESPUESTA FINAL
            # =================================================

            final_response = generate_final_response(
                user_input=user_input,
                first_analysis=first_analysis,
                context_review=context_review,
                conversation_history=conversation_history,
            )

            print("\n" + "=" * 70)
            print("🤖 RESPUESTA FINAL")
            print("=" * 70)
            print(final_response)
            print("=" * 70 + "\n")

            # Guardar únicamente la interacción visible.
            history.append(
                {
                    "user": user_input,
                    "assistant": final_response,
                }
            )

            # Limitar el tamaño de la memoria.
            if len(history) > MAX_HISTORY_TURNS:
                history = history[-MAX_HISTORY_TURNS:]

        except KeyboardInterrupt:
            print("\n\n👋 Conversación terminada.")
            break

        except Exception as error:
            print(f"\n❌ Error al procesar la solicitud: {error}\n")


if __name__ == "__main__":
    main()
