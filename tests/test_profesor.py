import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents import profesor


class FakeAgent:

    def __init__(self, respuesta):
        self.respuesta = respuesta
        self.entrada = None

    async def ainvoke(self, entrada):
        self.entrada = entrada

        return {
            "messages": [
                AIMessage(
                    content=self.respuesta
                )
            ]
        }


@pytest.mark.asyncio
async def test_nodo_profesor(monkeypatch):

    respuesta = (
        "La regresión es una técnica estadística "
        "que permite estudiar la relación entre variables."
    )

    fake_agent = FakeAgent(respuesta)

    monkeypatch.setattr(
        profesor,
        "create_agent",
        lambda **kwargs: fake_agent,
    )

    monkeypatch.setattr(
        profesor,
        "get_model",
        lambda provider: object(),
    )

    state = {
        "messages": [
            HumanMessage(
                content="¿Qué es la regresión?"
            )
        ],
        "contribuciones": [],
        "pasos": 0,
    }

    resultado = await profesor.nodo_profesor(state)

    assert resultado["messages"][0].name == "profesor"

    assert resultado["messages"][0].content == respuesta

    assert resultado["contribuciones"] == [
        {
            "agente": "profesor",
            "aporte": respuesta,
        }
    ]

    assert resultado["pasos"] == 1


@pytest.mark.asyncio
async def test_nodo_profesor_conserva_contribuciones(monkeypatch):

    respuesta = "Respuesta del profesor."

    fake_agent = FakeAgent(respuesta)

    monkeypatch.setattr(
        profesor,
        "create_agent",
        lambda **kwargs: fake_agent,
    )

    monkeypatch.setattr(
        profesor,
        "get_model",
        lambda provider: object(),
    )

    contribucion_anterior = {
        "agente": "evaluador",
        "aporte": "Evaluación anterior.",
    }

    state = {
        "messages": [
            HumanMessage(
                content="¿Qué es la correlación?"
            )
        ],
        "contribuciones": [
            contribucion_anterior
        ],
        "pasos": 1,
    }

    resultado = await profesor.nodo_profesor(state)

    assert resultado["contribuciones"] == [
        {
            "agente": "profesor",
            "aporte": respuesta,
        }
    ]

    assert resultado["pasos"] == 2


@pytest.mark.asyncio
async def test_nodo_profesor_fallback_de_proveedor(
    monkeypatch,
):

    llamadas = []

    class FakeAgent:

        async def ainvoke(self, entrada):
            return {
                "messages": [
                    AIMessage(
                        content="Respuesta desde Gemini simulado."
                    )
                ]
            }

    def fake_get_model(provider):

        llamadas.append(provider)

        if provider in ("openai", "anthropic"):
            raise ValueError(
                f"{provider} simulado como fallido."
            )

        return object()

    monkeypatch.setattr(
        profesor,
        "get_model",
        fake_get_model,
    )

    monkeypatch.setattr(
        profesor,
        "create_agent",
        lambda **kwargs: FakeAgent(),
    )

    state = {
        "messages": [
            HumanMessage(
                content="¿Qué es la correlación?"
            )
        ],
        "contribuciones": [],
        "pasos": 0,
    }

    resultado = await profesor.nodo_profesor(state)

    assert llamadas == [
        "openai",
        "anthropic",
        "gemini",
    ]

    assert resultado["messages"][0].name == "profesor"
    assert resultado["pasos"] == 1