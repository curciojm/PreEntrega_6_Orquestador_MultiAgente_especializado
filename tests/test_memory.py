import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from graph_config import grafo
import agents.supervisor as supervisor
import agents.profesor as profesor
import agents.sintesis as sintesis
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

    async def ainvoke(self, entrada):
        return {
            "messages": [
                AIMessage(
                    content="La correlación mide la relación entre dos variables."
                )
            ]
        }


class FakeSynthesisLLM:

    async def ainvoke(self, prompt):
        return AIMessage(
            content="Respuesta final de prueba."
        )


@pytest.mark.asyncio
async def test_memory_same_thread(monkeypatch, tmp_path):

    decisiones_supervisor = [
        DecisionSupervisor(
            next="profesor",
            razon="Es una consulta conceptual.",
        ),
        DecisionSupervisor(
            next="FINISH",
            razon="La consulta ya fue respondida.",
        ),
        DecisionSupervisor(
            next="profesor",
            razon="Es una nueva consulta conceptual.",
        ),
        DecisionSupervisor(
            next="FINISH",
            razon="La nueva consulta ya fue respondida.",
        ),
    ]

    fake_supervisor = FakeSupervisorLLM(
        decisiones_supervisor
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
        lambda **kwargs: FakeAgent(),
    )

    monkeypatch.setattr(
        sintesis,
        "get_model",
        lambda provider: FakeSynthesisLLM(),
    )

    database = tmp_path / "memory.sqlite"

    config = {
        "configurable": {
            "thread_id": "test-memory-1"
        },
        "recursion_limit": 10,
    }

    async with AsyncSqliteSaver.from_conn_string(
        str(database)
    ) as checkpointer:

        app = grafo.compile(
            checkpointer=checkpointer
        )

        await app.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content="¿Qué es la correlación?"
                    )
                ],
                "contribuciones": [],
                "pasos": 0,
                "next_agent": None,
                "task_completed": False,
            },
            config=config,
        )

        resultado_segundo = await app.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content="¿Y puede ser positiva?"
                    )
                ],
            },
            config=config,
        )

    mensajes = resultado_segundo["messages"]

    assert any(
        mensaje.content == "¿Qué es la correlación?"
        for mensaje in mensajes
        if isinstance(mensaje, HumanMessage)
    )

    assert any(
        mensaje.content
        == "La correlación mide la relación entre dos variables."
        for mensaje in mensajes
        if isinstance(mensaje, AIMessage)
    )

    assert any(
        mensaje.content == "¿Y puede ser positiva?"
        for mensaje in mensajes
        if isinstance(mensaje, HumanMessage)
    )

    assert any(
        contribucion["agente"] == "profesor"
        for contribucion in resultado_segundo["contribuciones"]
    )