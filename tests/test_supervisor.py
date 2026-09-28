import pytest
from langchain_core.messages import HumanMessage

import agents.supervisor as supervisor
from schemas import DecisionSupervisor


class FakeStructuredLLM:

    def __init__(self, decision):
        self.decision = decision
        self.messages = None

    def with_structured_output(self, schema):
        assert schema is DecisionSupervisor
        return self

    async def ainvoke(self, messages):
        self.messages = messages
        return self.decision


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision", "task_completed"),
    [
        ("profesor", False),
        ("evaluador", False),
        ("FINISH", True),
    ],
)
async def test_supervisor_devuelve_decision(
    monkeypatch,
    decision,
    task_completed,
):

    fake_llm = FakeStructuredLLM(
        DecisionSupervisor(
            next=decision,
            razon="Decisión simulada para el test.",
        )
    )

    monkeypatch.setattr(
        supervisor,
        "get_model",
        lambda provider: fake_llm,
    )

    state = {
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

    resultado = await supervisor.nodo_supervisor(state)

    assert resultado["next_agent"] == decision
    assert resultado["task_completed"] is task_completed
    assert fake_llm.messages is not None


@pytest.mark.asyncio
async def test_supervisor_finaliza_al_alcanzar_max_pasos(
    monkeypatch,
):

    def get_model_no_deberia_llamarse(provider):
        raise AssertionError(
            "El LLM no debería llamarse al alcanzar MAX_PASOS."
        )

    monkeypatch.setattr(
        supervisor,
        "get_model",
        get_model_no_deberia_llamarse,
    )

    state = {
        "messages": [
            HumanMessage(
                content="Consulta de prueba."
            )
        ],
        "contribuciones": [],
        "pasos": supervisor.MAX_PASOS,
        "next_agent": None,
        "task_completed": False,
    }

    resultado = await supervisor.nodo_supervisor(state)

    assert resultado["next_agent"] == "FINISH"
    assert resultado["task_completed"] is True


@pytest.mark.asyncio
async def test_supervisor_fallback_de_proveedor(
    monkeypatch,
):

    llamadas = []

    class FakeStructuredLLM:

        def with_structured_output(self, schema):
            assert schema is DecisionSupervisor
            return self

        async def ainvoke(self, messages):
            return DecisionSupervisor(
                next="profesor",
                razon="Gemini simulado.",
            )

    def fake_get_model(provider):

        llamadas.append(provider)

        if provider in ("openai", "anthropic"):
            raise ValueError(
                f"Proveedor {provider} simulado como fallido."
            )

        return FakeStructuredLLM()

    monkeypatch.setattr(
        supervisor,
        "get_model",
        fake_get_model,
    )

    state = {
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

    resultado = await supervisor.nodo_supervisor(state)

    assert llamadas == [
        "openai",
        "anthropic",
        "gemini",
    ]

    assert resultado["next_agent"] == "profesor"
    assert resultado["task_completed"] is False