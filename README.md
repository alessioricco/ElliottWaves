# Elliott Waves Analysis

This Python script, `elliottwaves.py`, is used for Elliott Waves analysis on financial data. Elliott Waves are a method of technical analysis that look for recurrent long-term price patterns related to persistent changes in investor sentiment and psychology.

## Function: ElliottWaveFindPattern

The main function in this script is `ElliottWaveFindPattern(df_source, measure, granularity, dateStart, dateEnd, extremes=True)`. This function finds and analyzes Elliott Wave patterns in the given data.

### Parameters

- `df_source`: A pandas DataFrame containing the source data.
- `measure`: The measure to be used for the analysis.
- `granularity`: The granularity of the data.
- `dateStart`: The start date for the data subset to consider.
- `dateEnd`: The end date for the data subset to consider.
- `extremes`: A boolean value indicating whether to consider extreme values or not. Default is `True`.

### Process

1. The function first subsets the data to consider based on the provided start and end dates.
2. It then finds the minimum and maximum values of the 'Close' measure in the data.
3. The function then draws the initial wave using the `draw_wave` function.
4. It then discovers the Elliott Waves in the data using the `ElliottWaveDiscovery` function.
5. The waves are then filtered using the `filterWaveSet` function.
6. The waves are then split into sets based on their length.
7. The best fit wave is then found for each set using the `findBestFitWave` function.
8. Finally, a wave chain set is built using the `buildWaveChainSet` function.

### Output

The function prints the initial number of waves, the waves themselves, the selected waves, and the best fit wave. It also draws the waves using the `draw_wave` function.

## Dependencies

This script requires pandas for data manipulation and matplotlib for drawing the waves.

## Usage

To use this script, import it and call the `ElliottWaveFindPattern` function with the appropriate parameters. Make sure your data is in the correct format and that you have the necessary dependencies installed.

```python
import pandas as pd
from elliottwaves import ElliottWaveFindPattern

# Load your data into a pandas DataFrame
df = pd.read_csv('your_data.csv')

# Call the function
ElliottWaveFindPattern(df, 'Close', 'D', '2020-01-01', '2020-12-31')
```


## Installing Dependencies

The Elliott Waves analysis script requires the following Python packages:

- pandas
- matplotlib

You can install these packages using pip, which is a package manager for Python. Open your terminal and type the following commands:

```bash
pip install pandas
pip install matplotlib

## Visualizing Elliott Waves Analysis Results

To visualize the Elliott Waves analysis results, you can use matplotlib to plot the price data and then overlay the identified wave patterns. Here's a basic example of how you might do this:

```python
import matplotlib.pyplot as plt

# Assuming `df` is your DataFrame and `waves` is the result from ElliottWaveFindPattern
df['Close'].plot()

for wave in waves:
    start, end = wave['start'], wave['end']
    plt.plot(df.index[start:end], df['Close'][start:end], label=f'Wave {wave["name"]}')

plt.legend()
plt.show()


This code first plots the 'Close' prices from your DataFrame. It then loops over the waves returned by ElliottWaveFindPattern, plotting each wave on the same graph. The start and end points of each wave are used to subset the 'Close' prices for that wave. Each wave is labeled with its name.

Please note that this is a basic example and may need to be adjusted based on the exact structure of your waves data and your specific visualization needs.

