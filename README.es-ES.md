

# Análisis de Ondas de Elliott

Este script de Python, `elliottwaves.py`, se utiliza para realizar análisis de Ondas de Elliott en datos financieros. Las Ondas de Elliott son un método de análisis técnico que busca patrones recurrentes de precios a largo plazo relacionados con cambios persistentes en el sentimiento y la psicología de los inversores.

## Función: ElliottWaveFindPattern

La función principal de este script es `ElliottWaveFindPattern(df_source, measure, granularity, dateStart, dateEnd, extremes=True)`. Esta función encuentra y analiza patrones de Ondas de Elliott en los datos proporcionados.

### Parámetros

- `df_source`: Un DataFrame de pandas que contiene los datos de origen.
- `measure`: La métrica a utilizar para el análisis.
- `granularity`: La granularidad de los datos.
- `dateStart`: La fecha de inicio para el subconjunto de datos a considerar.
- `dateEnd`: La fecha de finalización para el subconjunto de datos a considerar.
- `extremes`: Un valor booleano que indica si se deben considerar o no los valores extremos. El valor predeterminado es `True`.

### Proceso

1. Primero, la función subdivide los datos a considerar basándose en las fechas de inicio y fin proporcionadas.
2. Luego, encuentra los valores mínimo y máximo de la métrica 'Close' en los datos.
3. A continuación, la función dibuja la onda inicial utilizando la función `draw_wave`.
4. Luego, descubre las Ondas de Elliott en los datos utilizando la función `ElliottWaveDiscovery`.
5. Las ondas se filtran luego con la función `filterWaveSet`.
6. Las ondas se dividen en conjuntos basándose en su longitud.
7. Luego, se encuentra la onda de mejor ajuste para cada conjunto utilizando la función `findBestFitWave`.
8. Finalmente, se construye un conjunto de cadenas de ondas utilizando la función `buildWaveChainSet`.

### Salida

La función imprime el número inicial de ondas, las ondas en sí, las ondas seleccionadas y la onda de mejor ajuste. También dibuja las ondas utilizando la función `draw_wave`.

## Dependencias

Este script requiere pandas para la manipulación de datos y matplotlib para dibujar las ondas.

## Uso

Para utilizar este script, ingréselo e invoque la función `ElliottWaveFindPattern` con los parámetros adecuados. Asegúrese de que sus datos estén en el formato correcto y de que tenga instaladas las dependencias necesarias.

```python
import pandas as pd
from elliottwaves import ElliottWaveFindPattern

# Load your data into a pandas DataFrame
df = pd.read_csv('your_data.csv')

# Call the function
ElliottWaveFindPattern(df, 'Close', 'D', '2020-01-01', '2020-12-31')
```


## Instalación de dependencias

El script de análisis de Ondas de Elliott requiere los siguientes paquetes de Python:

- pandas
- matplotlib

Puede instalar estos paquetes usando pip, que es un administrador de paquetes para Python. Abra su terminal y escriba los siguientes comandos:

```bash
pip install pandas
pip install matplotlib

## Visualización de los resultados del análisis de Ondas de Elliott

Para visualizar los resultados del análisis de Ondas de Elliott, puede usar matplotlib para graficar los datos de precios y luego superponer los patrones de ondas identificados. A continuación, se presenta un ejemplo básico de cómo podría hacerlo:

```python
import matplotlib.pyplot as plt

# Assuming `df` is your DataFrame and `waves` is the result from ElliottWaveFindPattern
df['Close'].plot()

for wave in waves:
    start, end = wave['start'], wave['end']
    plt.plot(df.index[start:end], df['Close'][start:end], label=f'Wave {wave["name"]}')

plt.legend()
plt.show()


Este código primero grafica los precios de 'Close' de su DataFrame. Luego, recorre las ondas devueltas por ElliottWaveFindPattern, graficando cada onda en el mismo gráfico. Los puntos de inicio y fin de cada onda se utilizan para segmentar los precios de 'Close' para esa onda. Cada onda se etiqueta con su nombre.

Tenga en cuenta que este es un ejemplo básico y puede necesitar ajustes según la estructura exacta de sus datos de ondas y sus necesidades específicas de visualización.
