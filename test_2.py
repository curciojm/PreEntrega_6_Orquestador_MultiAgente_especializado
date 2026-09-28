import asyncio
from tools import evaluar_concepto, buscar_fuente, buscar_concepto

async def prueba():
    print("=== 1. evaluar_concepto ===")
    resultado1 = await evaluar_concepto.ainvoke({
        "tema": "regresion lineal",
        "respuesta_usuario": (
            "La regresión es una técnica estadística que permite modelar "
            "la relación entre una variable dependiente y una o más "
            "variables independientes. En la regresión lineal se estima "
            "una función que describe esa relación y permite cuantificar "
            "cómo cambia, en promedio, la variable dependiente cuando "
            "cambia una variable independiente. El modelo puede utilizarse "
            "tanto para analizar la relación entre las variables como para "
            "realizar predicciones."
        )
    })
    print(resultado1)

    print("\n=== 2. buscar_fuente ===")
    resultado2 = await buscar_fuente.ainvoke({
        "tema": "regresion lineal"
    })
    print(resultado2)

    print("\n=== 3. buscar_concepto ===")
    resultado3 = await buscar_concepto.ainvoke({
        "consulta": "regresion lineal"
    })
    print(resultado3)

asyncio.run(prueba())