import pytest
from langchain_core.documents import Document

import tools
from schemas import LLMError, ResultadoConcepto, ResultadoFuente, ResultadoEvaluacion

from tools import buscar_concepto, buscar_fuente, evaluar_concepto


@pytest.mark.asyncio
async def test_buscar_concepto(monkeypatch):

    documentos = [
        Document(
            page_content="La correlación estudia la relación entre dos variables.",
            metadata={
                "fuente": "Pagano 2006",
                "pagina": 10,
            },
        ),
        Document(
            page_content="La correlación puede ser positiva o negativa.",
            metadata={
                "fuente": "Pagano 2006",
                "pagina": 11,
            },
        ),
    ]

    class FakeRetriever:

        async def ainvoke(self, query):
            assert query == "¿Qué es la correlación?"
            return documentos

    monkeypatch.setattr(
        tools,
        "retriever_hibrido",
        FakeRetriever(),
    )

    resultado = await buscar_concepto.ainvoke(
        {"consulta": "¿Qué es la correlación?"}
    )

    assert isinstance(resultado, list)
    assert len(resultado) == 2

    assert all(
        isinstance(item, ResultadoConcepto)
        for item in resultado
    )

    assert resultado[0].contenido == (
        "La correlación estudia la relación entre dos variables."
    )


@pytest.mark.asyncio
async def test_buscar_fuente(monkeypatch):

    documentos = [
        Document(
            page_content="Contenido sobre correlación.",
            metadata={
                "fuente": "Pagano 2006",
                "pagina": 10.0,
            },
        ),
        Document(
            page_content="Más contenido sobre correlación.",
            metadata={
                "fuente": "Sampieri",
                "pagina": 25,
            },
        ),
    ]

    class FakeRetriever:

        async def ainvoke(self, query):
            assert query == "correlación"
            return documentos

    monkeypatch.setattr(
        tools,
        "retriever_hibrido",
        FakeRetriever(),
    )

    resultado = await buscar_fuente.ainvoke(
        {"tema": "correlación"}
    )

    assert isinstance(resultado, list)
    assert len(resultado) == 2

    assert all(
        isinstance(item, ResultadoFuente)
        for item in resultado
    )

    assert resultado[0].fuente == "Pagano 2006"
    assert resultado[0].pagina == 10
    assert isinstance(resultado[0].pagina, int)

    assert resultado[1].fuente == "Sampieri"
    assert resultado[1].pagina == 25


@pytest.mark.asyncio
async def test_buscar_concepto_clasifica_error(monkeypatch):

    class FakeRetriever:

        async def ainvoke(self, query):
            raise ValueError("Error de prueba")

    monkeypatch.setattr(
        tools,
        "retriever_hibrido",
        FakeRetriever(),
    )

    with pytest.raises(LLMError) as exc_info:
        await buscar_concepto.ainvoke(
            {"consulta": "¿Qué es la correlación?"}
        )

    assert exc_info.value.error_type.name == "UNKNOWN"


@pytest.mark.parametrize(
    ("similitud", "esperada"),
    [
        (0.40, "Mal"),
        (0.60, "Incompleta"),
        (0.80, "Bien"),
        (0.95, "Muy bien"),
    ],
)
@pytest.mark.asyncio
async def test_evaluar_concepto_clasifica_por_umbral(
    monkeypatch,
    similitud,
    esperada,
):

    documentos = [
        Document(
            page_content=(
                "La regresión estudia la relación entre una "
                "variable dependiente y una o más variables "
                "independientes."
            ),
            metadata={
                "fuente": "Pagano 2006",
                "pagina": 20,
            },
        )
    ]

    class FakeRetriever:

        async def ainvoke(self, query):
            assert query == "regresión"
            return documentos

    class FakeEmbeddings:

        async def aembed_query(self, texto):
            return [1.0, 0.0]

        async def aembed_documents(self, textos):
            return [
                [
                    similitud,
                    (1 - similitud ** 2) ** 0.5,
                ]
            ]

    monkeypatch.setattr(
        tools,
        "retriever_hibrido",
        FakeRetriever(),
    )

    monkeypatch.setattr(
        tools,
        "EMBEDDINGS",
        FakeEmbeddings(),
    )

    resultado = await evaluar_concepto.ainvoke(
        {
            "tema": "regresión",
            "respuesta_usuario": "Una explicación de prueba.",
        }
    )

    assert isinstance(resultado, ResultadoEvaluacion)
    assert resultado.evaluacion == esperada