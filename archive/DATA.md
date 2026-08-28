# Data files

Raw market data is intentionally excluded from version control. Before running
the notebook, place the following file in this directory:

```text
archive/spy-historical.csv
```

The notebook expects 8,210 daily SPY observations from 1993-02-01 through
2025-09-11 with these columns:

```text
date_id, Date, Open, High, Low, Close, Adj Close, Volume, Dividend, forward_returns
```

Expected SHA-256 checksum for the dataset used to produce the committed
results:

```text
b60ef80723379c268501c6f7b80e08aa18349c867e64f5734a651a6f2135c70e
```

`sp-historical.csv` is not used by the current analysis. The repository's MIT
license applies to the project code only; it does not grant redistribution
rights for third-party market data. Obtain the dataset from its authorized
source and follow the applicable competition or data-provider terms.
