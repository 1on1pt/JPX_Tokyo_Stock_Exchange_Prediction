```python
df_securities['daily_return'] = (df_securities.sort_values(by=['SecuritiesCode','ds'])
                                 .groupby('SecuritiesCode')['yhat'].pct_change())
```

```python
df_securities['rank'] = (df_securities.sort_values(by=['ds'])
                                 .groupby('ds')['daily_return'].rank(method='max',ascending=False))
```

```python
df_securities = df_securities.dropna()
df_securities.head()
```

```python
def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    """
    Args:
        df (pd.DataFrame): predicted results
        portfolio_size (int): # of equities to buy/sell
        toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
    Returns:
        (float): sharpe ratio
    """
    def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
        """
        Args:
            df (pd.DataFrame): predicted results
            portfolio_size (int): # of equities to buy/sell
            toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
        Returns:
            (float): spread return
        """
        assert df['rank'].min() == 0
        assert df['rank'].max() == len(df['rank']) - 1
        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
        purchase = (df.sort_values(by='rank')['Target'][:portfolio_size] * weights).sum() / weights.mean()
        short = (df.sort_values(by='rank', ascending=False)['Target'][:portfolio_size] * weights).sum() / weights.mean()
        return purchase - short

    buf = df.groupby('ds').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio
```

```python
calc_spread_return_sharpe(df_securities, 200, 2)
```

```python
f_securities = df_securities.drop(columns=['avg_rank'], axis=1)
df_securities
```

```python
df_securities['avg_rank'] = (df_securities.sort_values(by=['ds','rank'])
                                 .groupby('ds')['rank'].mean())
df_securities['avg_rank'].unique()
```

