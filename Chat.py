import os
import warnings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import time
from pathlib import Path

# Espera 2 segundos sin procesar

warnings.filterwarnings("ignore")

# 🔐 Cargar variables desde el archivo .env
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / "secrets" / ".env")
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ No se encontró OPENAI_API_KEY en el archivo .env")

# 🤖 Configuración del modelo OpenRouter
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model_name="cohere/north-mini-code:free",
    temperature=0.5,
)

# 🗨️ Bucle de chat básico
print("💬 Chatbot Mistral vía OpenRouter (escribe 'salir' para terminar)\n")

Meta_promt = """
Eres un profesor de matematicas, que provee definiciones u operaciones básicas de aritmetica,
estas operaciones son suma, resta, multiplicación y división. Interpreta la solicitud del
usuario en si desea sumar, restar, multiplicar, dividir o la definición matemática de alguna
palabra y clasifica su intención como una de las siguientes cinco opciones:
1) Suma, 2) Resta, 3) Multiplicación, 4) División, 5) Definición.
Como respuesta determina solo una de las anteriores cinco opciones y devuelvelas en el 
siguiente formato:
{
 "seleccion": "Suma"|"Resta"|"Multiplicacion"|"Division"|"Definicion",
 "Definicion": "Devuelve vacio si la seleccion no es definicion y devuelve la definicion si la seleccion si lo es"
}

"""

Memo = ''
while True:
    user_input = input("👤 Tú: ")
    if user_input.lower() in ["salir", "exit", "quit"]:
        print("👋 Hasta luego.")
        break

    try:
        response = llm.invoke([HumanMessage(content='Memoria del chat:' + Memo + 'Condiciones' + Meta_promt + 'Usuario' + user_input)])
        #print('Memoria del chat:' + Memo + 'Condiciones' + Meta_promt + 'Usuario' + user_input)
        try:
            # Si es un AIMessage (objeto de mensaje)
            print(f"🤖 Bot: {response.content.strip()}\n")
            Memo = Memo + user_input + response.content.strip()
            time.sleep(2)
        except AttributeError:
            # Si es un dict o lista de mensajes (caso nuevo en langchain_openai)
            if isinstance(response, dict) and "content" in response:
                print(f"🤖 Bot: {response['content'].strip()}\n")
            elif isinstance(response, list) and len(response) > 0:
                print(f"🤖 Bot: {response[0].content.strip()}\n")
            else:
                print(f"🤖 Bot: {response}\n")

    except Exception as e:
        print(f"❌ Error: {e}\n")

