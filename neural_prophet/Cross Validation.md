# Neural Prophet Cross Validation

```python
m = NeuralProphet(
        n_forecasts=20,
        n_lags=60,
        n_changepoints=50,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        batch_size=56,
        epochs=25,
        learning_rate=1.0,
    )

```

```python
df = pd.DataFrame(target_df[6146])
df.reset_index(inplace=True)
df['y'] = df[6146]
df.drop(columns=[6146],inplace=True)
df.dropna(inplace=True)
df.columns
```

```python
# create a test holdout set:
df_train_val, df_test = m.split_df(df=df, freq="D", valid_p=0.2)
# create a validation holdout set:
df_train, df_val = m.split_df(df=df_train_val, freq="D", valid_p=0.2)
```

```python
# fit a model on training data and evaluate on validation set.
metrics_train1 = m.fit(df=df_train, freq="D")
metrics_val = m.test(df=df_val)
```

```python
# refit model on training and validation data and evaluate on test set.
m = NeuralProphet(learning_rate = 1.0)
metrics_train2 = m.fit(df=df_train_val, freq="D")
metrics_test = m.test(df=df_test)
```

```python
metrics_train1["split"]  = "train1"
metrics_train2["split"]  = "train2"
metrics_val["split"] = "validate"
metrics_test["split"] = "test"
metrics_train1.tail(1).append([metrics_train2.tail(1), metrics_val, metrics_test]).drop(columns=['RegLoss'])
```

```python
METRICS = ['SmoothL1Loss', 'MAE', 'RMSE']
params = {"seasonality_mode": "multiplicative", "learning_rate": 1.0}


folds = NeuralProphet(**params).crossvalidation_split_df(df, freq="D", k=5, fold_pct=0.20, fold_overlap_pct=0.5)
```

```python
metrics_train = pd.DataFrame(columns=METRICS)
metrics_test = pd.DataFrame(columns=METRICS)

for df_train, df_test in folds:
    m = NeuralProphet(**params)
    train = m.fit(df=df_train, freq="D")
    test = m.test(df=df_test)
    metrics_train = metrics_train.append(train[METRICS].iloc[-1])
    metrics_test = metrics_test.append(test[METRICS].iloc[-1])
```

```python
metrics_test.describe()
```

```python
metrics_val.describe()
```

```python
m = NeuralProphet(
        n_forecasts=10,
        n_lags=60,
        n_changepoints=50,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        batch_size=56,
        epochs=25,
        learning_rate=1.0,
    )
metrics = m.fit(df,freq='D',progress='plot-all')
```

```python
future = m.make_future_dataframe(df, periods=56, n_historic_predictions=len(df))
forecast = m.predict(future)
fig1 = m.plot(forecast)
```

```python
fig_comp1 = m.plot_components(forecast)
```

```python
fig_param1 = m.plot_parameters()
```

# Prophet with Cross Validation

```python
m = Prophet(
        n_changepoints=50,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True        
    )
```

```python
target_df = prices_target.reset_index()
target_df = target_df.set_index('ds')
```

```python
df = pd.DataFrame(target_df[9983])
df.reset_index(inplace=True)
df['y'] = df[9983]
df.drop(columns=[9983],inplace=True)
df.dropna(inplace=True)
df = df[df['y'] > 0]
df.columns
```

```python
m.fit(df)
future = m.make_future_dataframe(periods=56, freq='D')
future.tail()
```

```python
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(20)
```

```python
fig1 = m.plot(forecast)
```

```python
fig2 = m.plot_components(forecast)
```

```python
fig = plot_plotly(m, forecast)
plot_plotly(m, forecast)
```

```python
plot_components_plotly(m, forecast)
```

```python
forecast.tail()
```

```python
# create cross-validation df
from prophet.diagnostics import cross_validation
df_cv = cross_validation(m, initial='730 days', period='180 days', horizon = '365 days')
df_cv
```

```python
# create performance metrics df
df_p = performance_metrics(df_cv)
df_p.head()
```

```python
# plot cross-validation
fig = plot_cross_validation_metric(df_cv, metric='mape')
```

