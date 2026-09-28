import asyncio
from tools import buscar_concepto

async def prueba():
    print("PRIMERA LLAMADA")

    resultado1 = await buscar_concepto.ainvoke({
        "consulta": "regresion lineal"
    })

    print(resultado1)

    print("\nSEGUNDA LLAMADA")

    resultado2 = await buscar_concepto.ainvoke({
        "consulta": "regresion lineal"
    })

    print(resultado2)


asyncio.run(prueba())