import pandas as pd

df = pd.read("your_dataset.csv")

#15 period SMA
df['15MA'] = df['close'].rolling(window=time_window).mean()

#creating the data points needed for difference methods
df['f(xi-1)'] = df['15MA'].shift(1)
df['f(xi-2)'] = df['15MA'].shift(2)
df['f(xi-3)'] = df['15MA'].shift(3)
df['f(xi-4)'] = df['15MA'].shift(4)

#two point backward difference
df['first_der'] = df.apply(lambda row: (row['15MA'] - row['f(xi-1)'])/(time_window), axis=1)
#three point backward difference
df['second_der'] = df.apply(lambda row: (row['f(xi-2)']-2*row['f(xi-1)']+row['15MA'])/(time_window**2), axis=1)
#four point backward difference
df['third_der'] = df.apply(lambda row: (-row['f(xi-3)']+3*row['f(xi-2)']-3*row['f(xi-1)']+row['15MA'])/(time_window**3), axis=1)
#five point backward difference
df['fourth_der'] = df.apply(lambda row: (row['f(xi-4)']-4*row['f(xi-3)']+6*row['f(xi-2)']-4*row['f(xi-1)']+row['15MA'])/(time_window**4), axis=1)
df.drop(['f(xi-1)','f(xi-2)','f(xi-3)','f(xi-4)'], inplace=True, axis=1)
#what is the third and fourth derivative? lol

print(df.head())
