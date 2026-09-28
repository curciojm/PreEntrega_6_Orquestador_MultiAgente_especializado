import pytest
from langchain_core.messages import AIMessage, HumanMessage

import graph_config
from agents import supervisor
from agents import profesor
from agents import evaluador
from agents import sintesis
from schemas import DecisionSupervisor


class FakeSupervisorLLM:

    def __init__(self, decisiones):
        self.decisiones = iter(decisiones)

    def with_structured_output(self, schema):
        assert schema is DecisionSupervisor
        return self

    async def ainvoke(self, messages):
        return next(self.decisiones)


class FakeAgent:

    def __init__(self, respuesta):
        self.respuesta = respuesta

    async def ainvoke(self, entrada):
        return {
            "messages": [
                AIMessage(content=self.respuesta)
            ]
        }


class FakeSynthesisLLM:

    async def ainvoke(self, prompt):
        return AIMessage(
            content="Respuesta final sintetizada."
        )


@pytest.mark.asyncio
async def test_graph_profesor_supervisor_sintesis(monkeypatch):

    fake_supervisor = FakeSupervisorLLM(
        [
            DecisionSupervisor(
                next="profesor",
                razon="Es una consulta conceptual.",
            ),
            DecisionSupervisor(
                next="FINISH",
                razon="El profesor ya respondió suficientemente.",
            ),
        ]
    )

    monkeypatch.setattr(
        supervisor,
        "get_model",
        lambda provider: fake_supervisor,
    )

    monkeypatch.setattr(
        profesor,
        "get_model",
        lambda provider: object(),
    )

    monkeypatch.setattr(
        profesor,
        "create_agent",
        lambda **kwargs: FakeAgent(
            "La regresión estudia la relación entre variables."
        ),
    )

    monkeypatch.setattr(
        sintesis,
        "get_model",
        lambda provider: FakeSynthesisLLM(),
    )

    app = graph_config.grafo.compile()

    resultado = await app.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="¿Qué es la regresión?"
                )
            ],
            "contribuciones": [],
            "pasos": 0,
            "next_agent": None,
            "task_completed": False,
        }
    )

    mensajes = resultado["messages"]

    assert mensajes[-1].name == "sintesis"
    assert mensajes[-1].content == (
        "Respuesta final sintetizada."
    )

    assert any(
        getattr(mensaje, "name", None) == "profesor"
        for mensaje in mensajes
    )

    assert any(
        contribucion["agente"] == "profesor"
        for contribucion in resultado["contribuciones"]
    )


@pytest.mark.asyncio
async def test_graph_evaluador_supervisor_sintesis(monkeypatch):

    fake_supervisor = FakeSupervisorLLM(
        [
            DecisionSupervisor(
                next="evaluador",
                razon="El usuario proporciona una respuesta propia.",
            ),
            DecisionSupervisor(
                next="FINISH",
                razon="El evaluador ya completó la evaluación.",
            ),
        ]
    )

    monkeypatch.setattr(
        supervisor,
        "get_model",
        lambda provider: fake_supervisor,
    )

    monkeypatch.setattr(
        evaluador,
        "get_model",
        lambda provider: object(),
    )

    monkeypatch.setattr(
        evaluador,
        "create_agent",
        lambda **kwargs: FakeAgent(
            "Evaluación: Incompleta\n"
            "Faltan algunos elementos conceptuales."
        ),
    )

    monkeypatch.setattr(
        sintesis,
        "get_model",
        lambda provider: FakeSynthesisLLM(),
    )

    app = graph_config.grafo.compile()

    resultado = await app.ainvoke(
        {
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
            "next_agent": None,
            "task_completed": False,
        }
    )

    mensajes = resultado["messages"]

    assert mensajes[-1].name == "sintesis"
    assert mensajes[-1].content == (
        "Respuesta final sintetizada."
    )

    assert any(
        getattr(mensaje, "name", None) == "evaluador"
        for mensaje in mensajes
    )

    assert any(
        contribucion["agente"] == "evaluador"
        for contribucion in resultado["contribuciones"]
    )