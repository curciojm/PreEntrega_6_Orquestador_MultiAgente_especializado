import pytest
from langchain_core.messages import AIMessage, HumanMessage

import agents.sintesis as sintesis


class FakeLLM:

    def __init__(self):
        self.prompt = None

    async def ainvoke(self, prompt):
        self.prompt = prompt

        return AIMessage(
            content="Respuesta final sintetizada."
        )


@pytest.mark.asyncio
async def test_nodo_sintesis(monkeypatch):

    fake_llm = FakeLLM()

    monkeypatch.setattr(
        sintesis,
        "get_model",
        lambda provider: fake_llm,
    )

    state = {
        "messages": [
            HumanMessage(
                content="¿Qué es la regresión?"
            )
        ],
        "contribuciones": [
            {
                "agente": "profesor",
                "aporte": (
                    "La regresión estudia la relación "
                    "entre variables."
                ),
            }
        ],
    }

    resultado = await sintesis.nodo_sintesis(state)

    assert resultado["messages"][0].name == "sintesis"

    assert resultado["messages"][0].content == (
        "Respuesta final sintetizada."
    )

    assert "¿Qué es la regresión?" in fake_llm.prompt

    assert (
        "La regresión estudia la relación entre variables."
        in fake_llm.prompt
    )