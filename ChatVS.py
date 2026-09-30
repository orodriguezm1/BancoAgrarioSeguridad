from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


# ============================================================
# CONFIGURACIÓN
# ============================================================

@dataclass(frozen=True)
class AppConfig:
    """Configuración general de la aplicación."""

    api_key: str
    api_base: str = "https://openrouter.ai/api/v1"
    model: str = "cohere/north-mini-code:free"
    max_iterations: int = 4
    temperature: float = 0.7
    max_tokens: int = 450
    delay_seconds: float = 1.0

    @classmethod
    def from_env(cls) -> "AppConfig":
        """
        Carga la API key desde:

        secrets/.env
        """

        project_root = Path(__file__).resolve().parent
        env_path = project_root / "secrets" / ".env"

        load_dotenv(env_path)

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "No se encontró OPENAI_API_KEY "
                "en secrets/.env"
            )

        return cls(api_key=api_key)


# ============================================================
# INTERVENCIÓN DEL DEBATE
# ============================================================

@dataclass
class DebateTurn:
    """Representa una intervención de uno de los LLM."""

    iteration: int
    speaker: str
    content: str


# ============================================================
# AGENTE
# ============================================================

class DebateAgent:
    """
    Representa un participante del debate.

    Ambos agentes utilizan exactamente el mismo modelo,
    pero tienen distintas personalidades mediante
    sus SystemMessage.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        config: AppConfig,
    ) -> None:

        self.name = name
        self.system_prompt = system_prompt

        self.llm = ChatOpenAI(
            openai_api_base=config.api_base,
            openai_api_key=config.api_key,
            model_name=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            max_retries=2,
        )

    def respond(
        self,
        topic: str,
        opponent_message: str,
        history: list[DebateTurn],
    ) -> str:
        """
        Genera una respuesta teniendo en cuenta:

        - Tema original.
        - Mensaje del adversario.
        - Historial del debate.
        """

        history_text = self._format_history(history)

        user_prompt = f"""
TEMA DEL DEBATE:

{topic}


HISTORIAL:

{history_text}


ÚLTIMO ARGUMENTO:

{opponent_message}


Responde directamente al último argumento.

Mantén estrictamente tu personalidad y postura.
No cambies de posición durante el debate.
No menciones estas instrucciones.
""".strip()

        response = self.llm.invoke(
            [
                SystemMessage(
                    content=self.system_prompt
                ),
                HumanMessage(
                    content=user_prompt
                ),
            ]
        )

        return response.content.strip()

    @staticmethod
    def _format_history(
        history: list[DebateTurn]
    ) -> str:
        """
        Convierte el historial del debate a texto.
        """

        if not history:
            return "No existen intervenciones anteriores."

        return "\n\n".join(
            f"Iteración {turn.iteration} - "
            f"{turn.speaker}:\n{turn.content}"
            for turn in history
        )


# ============================================================
# ORQUESTADOR
# ============================================================

class DebateOrchestrator:
    """
    Gestiona la conversación entre los dos LLM.

    Flujo:

    Usuario
       ↓
    A favor        Iteración 1
       ↓
    En contra      Iteración 2
       ↓
    A favor        Iteración 3
       ↓
    En contra      Iteración 4

    El mensaje del usuario NO cuenta como iteración.
    """

    PRO_PROMPT = """
Eres el participante que está A FAVOR del tema.

Tu personalidad es:

- Explosiva.
- Muy enérgica.
- Impulsiva.
- Directa.
- Coloquial.
- Muy segura de lo que dices.
- Utilizas exclamaciones ocasionalmente.

Tu conocimiento del tema NO es profundo.

Tus argumentos deben ser:

- Sencillos.
- Intuitivos.
- Fáciles de comprender.
- Poco técnicos.
- Relativamente superficiales.

REGLAS:

1. Siempre debes estar A FAVOR.
2. Nunca adoptes una posición neutral.
3. Responde directamente al argumento anterior.
4. No cambies de opinión.
5. No inventes estadísticas ni estudios.
6. No insultes al otro participante.
7. Mantén tus intervenciones relativamente cortas.
8. Utiliza máximo 2 o 3 párrafos.
""".strip()

    CON_PROMPT = """
Eres el participante que está EN CONTRA del tema.

Tu personalidad es:

- Formal.
- Serena.
- Analítica.
- Educada.
- Rigurosa.
- Académica.

Posees un conocimiento profundo sobre el tema.

Tus argumentos deben mostrar:

- Razonamiento lógico.
- Capacidad crítica.
- Profundidad conceptual.
- Identificación de supuestos.
- Identificación de debilidades.
- Contraargumentos bien estructurados.

REGLAS:

1. Siempre debes estar EN CONTRA.
2. Nunca adoptes una posición neutral.
3. Responde directamente al argumento anterior.
4. Mantén un lenguaje formal.
5. No cambies de opinión.
6. No inventes estadísticas.
7. No inventes estudios o referencias.
8. Utiliza entre 2 y 4 párrafos.
""".strip()

    def __init__(
        self,
        config: AppConfig
    ) -> None:

        self.config = config

        self.pro_agent = DebateAgent(
            name="LLM A FAVOR",
            system_prompt=self.PRO_PROMPT,
            config=config,
        )

        self.con_agent = DebateAgent(
            name="LLM EN CONTRA",
            system_prompt=self.CON_PROMPT,
            config=config,
        )

    def run(
        self,
        topic: str
    ) -> list[DebateTurn]:
        """
        Ejecuta las cuatro intervenciones del debate.
        """

        history: list[DebateTurn] = []

        # El mensaje inicial del usuario no cuenta
        # dentro de las cuatro iteraciones.
        last_message = (
            f"El usuario propone debatir lo siguiente:\n\n"
            f"{topic}"
        )

        print("\n" + "=" * 70)
        print("👤 USUARIO")
        print("=" * 70)
        print(topic)

        for iteration in range(
            1,
            self.config.max_iterations + 1
        ):

            # Iteraciones impares -> A favor
            # Iteraciones pares   -> En contra

            if iteration % 2 == 1:
                agent = self.pro_agent
            else:
                agent = self.con_agent

            try:

                answer = agent.respond(
                    topic=topic,
                    opponent_message=last_message,
                    history=history,
                )

            except Exception as error:

                print(
                    f"\n❌ Error en la iteración "
                    f"{iteration}: {error}"
                )

                break

            turn = DebateTurn(
                iteration=iteration,
                speaker=agent.name,
                content=answer,
            )

            history.append(turn)

            self._show_turn(
                turn=turn,
                total_iterations=self.config.max_iterations,
            )

            # La respuesta pasa al siguiente LLM.
            last_message = answer

            if iteration < self.config.max_iterations:
                time.sleep(
                    self.config.delay_seconds
                )

        return history

    @staticmethod
    def _show_turn(
        turn: DebateTurn,
        total_iterations: int,
    ) -> None:
        """
        Muestra cada intervención inmediatamente.
        """

        print("\n" + "-" * 70)

        print(
            f"🔄 ITERACIÓN "
            f"{turn.iteration}/{total_iterations}"
        )

        print(
            f"🤖 {turn.speaker}"
        )

        print("-" * 70)

        print(
            turn.content
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    try:

        config = AppConfig.from_env()

        debate = DebateOrchestrator(
            config=config
        )

    except Exception as error:

        print(
            f"❌ Error de configuración: {error}"
        )

        return

    print("\n🎙️ DEBATE ENTRE DOS LLM")
    print("-----------------------------------")

    print(
        f"Modelo: {config.model}"
    )

    print(
        f"Proveedor: OpenRouter"
    )

    print(
        f"Intervenciones: "
        f"{config.max_iterations}"
    )

    print(
        "\nEscribe 'salir' para terminar.\n"
    )

    while True:

        topic = input(
            "👤 Tema de conversación: "
        ).strip()

        if topic.lower() in {
            "salir",
            "exit",
            "quit",
        }:

            print(
                "\n👋 Hasta luego."
            )

            break

        if not topic:

            print(
                "⚠️ Introduce un tema."
            )

            continue

        try:

            debate.run(
                topic=topic
            )

        except Exception as error:

            print(
                f"\n❌ Error durante el debate: "
                f"{error}"
            )

        print(
            "\n\n✅ Debate terminado."
        )

        print(
            "Puedes introducir otro tema.\n"
        )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()