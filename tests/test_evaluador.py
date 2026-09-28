import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents import evaluador


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
async def test_nodo_evaluador(monkeypatch):

    respuesta = (
        "Evaluación: Incompleta\n"
        "La respuesta identifica parcialmente el concepto, "
        "pero faltan algunos elementos importantes."
    )

    fake_agent = FakeAgent(respuesta)

    monkeypatch.setattr(
        evaluador,
        "create_agent",
        lambda **kwargs: fake_agent,
    )

    monkeypatch.setattr(
        evaluador,
        "get_model",
        lambda provider: object(),
    )

    state = {
        "messages": [
            HumanMessage(
                content=(
                    "Mi explicación sobre la regresión es: "
                    "una técnica que relaciona dos variables."
                )
            )
        ],
        "contribuciones": [],
        "pasos": 0,
    }

    resultado = await evaluador.nodo_evaluador(state)

    assert resultado["messages"][0].name == "evaluador"

    assert resultado["messages"][0].content == respuesta

    assert resultado["messages"][0].content.startswith(
        "Evaluación: Incompleta"
    )

    assert resultado["contribuciones"] == [
        {
            "agente": "evaluador",
            "aporte": respuesta,
        }
    ]

    assert resultado["pasos"] == 1


@pytest.mark.asyncio
async def test_nodo_evaluador_conserva_pasos(monkeypatch):

    respuesta = "Evaluación: Bien\nLa respuesta es adecuada."

    fake_agent = FakeAgent(respuesta)

    monkeypatch.setattr(
        evaluador,
        "create_agent",
        lambda **kwargs: fake_agent,
    )

    monkeypatch.setattr(
        evaluador,
        "get_model",
        lambda provider: object(),
    )

    state = {
        "messages": [
            HumanMessage(
                content="Mi explicación sobre la regresión es correcta."
            )
        ],
        "contribuciones": [],
        "pasos": 2,
    }

    resultado = await evaluador.nodo_evaluador(state)

    assert resultado["pasos"] == 3

    assert resultado["messages"][0].name == "evaluador"


@pytest.mark.asyncio
async def test_nodo_evaluador_fallback_de_proveedor(
    monkeypatch,
):

    llamadas = []

    class FakeAgent:

        async def ainvoke(self, entrada):
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "Evaluación: Muy bien\n"
                            "La explicación es adecuada."
                        )
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
        evaluador,
        "get_model",
        fake_get_model,
    )

    monkeypatch.setattr(
        evaluador,
        "create_agent",
        lambda **kwargs: FakeAgent(),
    )

    state = {
        "messages": [
            HumanMessage(
                content="Mi explicación sobre regresión."
            )
        ],
        "contribuciones": [],
        "pasos": 0,
    }

    resultado = await evaluador.nodo_evaluador(state)

    assert llamadas == [
        "openai",
        "anthropic",
        "gemini",
    ]

    assert resultado["messages"][0].name == "evaluador"

    assert resultado["messages"][0].content.startswith(
        "Evaluación: Muy bien"
    )

    assert resultado["pasos"] == 1