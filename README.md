# JPX_Tokyo_Stock_Exchange_Prediction

![tokyo-stock-exchange](https://user-images.githubusercontent.com/94148420/166913015-6cb22041-0dd8-48f7-9b79-f8fef31ab4b9.jpg)

##### University of Wisconsin Extension Campus Data Analytics Bootcamp Final Project

### Quick Links
* [Presentation](#presentation)
* [GitHub](#github)
* [Machine Learning](#machine-learning)
* [Database](#database)
* [Dashboard](#dashboard)
* [Google Slides Presentation](https://docs.google.com/presentation/d/1edVy4hYBCBOtUxXZ-iFzxX-yDt1bUzcv-C_nE8Z_uqs/edit?usp=sharing)
* [Tableau Public Dashboard](https://public.tableau.com/app/profile/nate.millmann/viz/TokyoStockDashboard/Dashboard1)
* [Summary](#summary)


### Team Members

| **Member**                                            |    **Primary Role**    | **Responsibilities**                                       |
| ----------------------------------------------------- | :--------------------: | ---------------------------------------------------------- |
| **[Aslesha Vangareddy](https://github.com/AsleshaV)** |       Dashboard        | Manage the development of the dashboard                    |
| **[Jerri Morales](https://github.com/jerrimor)**      |        Database        | Manage the development of the database                    |
| **[Carl Stewart](https://github.com/CarlS2rt)**       | Maching Learning Model | Manage the development of the machine learning model      |
| **[Eric Himburg](https://github.com/eric-himburg)**   | Machine Learning Model | Manage the development of the machine learning model       |
| **[Nate Millmann](https://github.com/millmannnate)**  | Machine Learning Model; Dashboard | Manage the development of the machine learning model; dashboard development       |
| **[John Beauchamp](https://github.com/1on1pt)**       |    GitHub; Database    | Manage GitHub repository; assist with database development |

**Although Team Members had a *Primary Role*, each contributed to all aspects of this final project.**
                                                                        
## Presentation
### Selected Topic - Analysis of Stock Performance
Using machine learning models to predict the performance of stocks from the JPX Tokyo Stock Exchange and rank the stocks from highest to lowest expected returns.  

### Reason for Selected Topic
The data scientists in our group are interested in exploring quantitative trading where decisions to buy or sell stocks are made based on predictions from trained models.  Historically, finance decisions have been made manually by professionals who decide whether a stock or derivative is undervalued or overvalued.  Our goal is to use machine learning to quickly evaluate a large set of financial data in order to make a portfolio of predicted stock outcomes.

### Description of Source of Data
This dataset contains historic data for a variety of Japanese stocks and options obtained from the Japan Exchange Group.  The Japan Exchange Group, Inc. (JPX) is a holding company operating one of the largest stock exchanges in the world, Tokyo Stock Exchange (TSE).  In the dataset are the financial data of around 2,000 stocks traded on the TSE, which includes information on the opening and closing prices of the stocks, daily high and low prices, and the volume of the stock transactions.  

### Questions to Answer with Data
1.	How accurately can machine learning models predict the outcome of stocks using historical data?  
2.	Which machine learning model makes the most accurate predictions of the stock market? 
3.	Are hybrid machine learning models more accurate than one simple model?

### Description of Data Exploration Phase of Project
![etl_process](https://user-images.githubusercontent.com/94148420/168453788-457db515-af66-4767-acad-abb4d0056eb1.PNG)

#### Data Exploration Summary
Ultimately, two of the files from the extraction were considered essential for this project:
* https://github.com/1on1pt/JPX_Tokyo_Stock_Exchange_Prediction/blob/main/train_files/financials.csv
* https://github.com/1on1pt/JPX_Tokyo_Stock_Exchange_Prediction/blob/main/train_files/stock_prices.csv

These two files were cleaned of unnecessary columns, rows, and null values:
* https://github.com/1on1pt/JPX_Tokyo_Stock_Exchange_Prediction/blob/main/financials_clean.ipynb

![financials_clean_head](https://user-images.githubusercontent.com/94148420/168492280-ee64b2f7-34c7-4f5f-99ab-3d98e8d02349.PNG)

* https://github.com/1on1pt/JPX_Tokyo_Stock_Exchange_Prediction/blob/main/prices_clean.ipynb

![stock_prices_clean_head](https://user-images.githubusercontent.com/94148420/168492410-8b126785-4e41-43fd-b1f5-bc60e043edaf.PNG)


### Description of the Analysis Phase of the Project
There will be six main parts to the analysis phase of the project.
1.	The **NeuralProphet** model will be used to forecast the best 20 stocks out of approximately 2000 stocks from the Tokyo Stock Exchange (TSE) over a period of 56 trading days.  Additionally, the predictions of some of the top stocks will be compared to historical data to see how well the model performed.    
2.	The **Stacked LSTM** model will be used to forecast the best 20 stocks out of approximately 2000 stocks from the TSE over a period of 56 trading days.  Additionally, the predictions of some of the top stocks will be compared to historical data to see how well the model performed.
3.	A comparison will be made between the **NeuralProphet** and **Stacked LSTM** models to see if one predicts stock performance better than the other.  Additionally, a hybrid model utilizing both the NeuralProphet and Stacked LSTM models may be run to see if the hybrid model performs better than either model individually.
4.	**Candlestick** stock data, the daily swing between the high and low performance of the stock versus the closing price, of the 20 best performing stocks and the 20 worst performing stocks will be plotted.  An analysis of the differences in the candlestick data between the best and worst stocks will be done to determine if there is a correlation in the range of stock price fluctuation to the overall growth of the stock.
5.  **Forecast Earnings** versus **Net Sales** for all TSE Securities which have this data included in our dataset will be analyzed.  First, the data will be aggregated in a table within the database before exporting.  Afterward, a scatter plot will be generated to determine if there are any correlations between forecast earnings and net sales.  Earnings forecasts are based on analysts' expectations of company growth and profitability.  Net sales are the sum of a company's gross sales minus its returns, allowances, and discounts.  It is expected that there will be a strong correlation between the two.  
6.	**Material Change in Subsidiaries** versus **stock Price** for all TSE Securities which have this data included in our dataset will be analyzed.  First, the data will be aggregated in a table within the database before exporting.  Afterward, a scatter plot will be generated to determine if there are any correlations between material change in subsidiaries and stock price.  A material change in the affairs of a company is something expected to affect the market value of its securities. These changes can include a change in the nature of the business, a change in senior principal officers, or a change in the share ownership of the company.  It is expected that material changes will affect stock price both positively and negatively, depending on the type of material change occurring.

**Results of the final data analysis will be discussed in the project summary.**

### Resources

![resources_image](https://user-images.githubusercontent.com/94148420/170481277-13a276bf-6a59-4cc8-8a07-e734b48944b1.PNG)


### Presentation
The presentation is hosted on Google Slides and can be accessed [here](https://docs.google.com/presentation/d/1edVy4hYBCBOtUxXZ-iFzxX-yDt1bUzcv-C_nE8Z_uqs/edit?usp=sharing).

**Back to [Quick Links](#quick-links)**

## GitHub
### Main Branch
* Includes README.md
* Code necessary to perform exploratory analysis


* Some code necessary to complete the machine learning portion of the project
    * https://github.com/1on1pt/JPX_Tokyo_Stock_Exchange_Prediction/blob/main/JPX_Prophet_results_testing.ipynb


### README.md Must Include
* Outline of the project
![project_outline](https://user-images.githubusercontent.com/94148420/168448850-38caa56f-0355-432f-895e-5d46030d3050.PNG)



### Individual Branches
* At least one branch for each team member
    * Each team member will add their own branch to the respository

* Each team member has at least four commits from the duration of first segment
    * To be tracked throughout the duration of this project

**Back to [Quick Links](#quick-links)**

## Machine Learning
Three different models will be used to make TSE stock predictions.

##### Stacked LSTM 

Built upon the original long short-term memory model or LSTM model, the Stacked LSTM has multiple hidden LSTM layers and each layer contains multiple memory cells.  An LSTM layer creates a sequence output rather than a single output value.  Hence, for every input time step there is an output time.  This makes it ideal for making time-based stock market predictions. A Stacked LSTM model will be created using the Keras Python deep learning library.   

##### Prophet

Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well (https://github.com/facebook/prophet).

##### NeuralProphet

A Neural Network based Time-Series model, inspired by [Facebook Prophet](https://github.com/facebook/prophet) and [AR-Net](https://github.com/ourownstory/AR-Net), built on PyTorch. NeuralProphet is a hybrid forecasting framework based on PyTorch and trained with standard deep learning methods, making it easy for developers to extend the framework. Local context is introduced with auto-regression and covariate modules, which can be configured as classical linear regression or as Neural Networks. Otherwise, NeuralProphet retains the design philosophy of Prophet and provides the same basic model components (https://github.com/ourownstory/neural_prophet).  


### Description of Preliminary Data Preprocessing
The dataset used was acquired from the Japan Exchange Group, Inc. (JPX), which is a holding company operating the Tokyo Stock Exchange (TSE).  The dataset contains over four years of historical data on about 2000 Japanese stocks.  Preprocessing the data involved evaluating the class type of each column of data, checking for null values, and removing any rows of data where null values were found.  A screenshot below shows a portion of the Python code where null values were first evaluated and then removed.  Additional data preprocessing involved converting the ‘date’ column of data to datetime values so that days could be added and correct dates outputted for predictions.  Lastly, unique values of the security codes of the stocks in the dataset were put into a list.

![part of python code showing preprocessing of data](https://raw.githubusercontent.com/1on1pt/JPX_Tokyo_Stock_Exchange_Prediction/main/Images/stackedlstm_preprocessing.png)

The Prophet and Neural Prophet models required some model-specific preprocessing to allow the model to run. Both models work on a time-series forecast approach and require the Date column to be 'ds' and the historical data to be 'y'. After the data was cleaned and grouped by the Securities Code and Date, the column values were renamed to conform with the model parameters in a for loop:

```python
for i in itemlist:
    temp = df_grouped[df_grouped.SecuritiesCode == i]
    temp = temp.drop(columns=['SecuritiesCode'])
    temp['Date'] = pd.to_datetime(temp['Date'])
    temp = temp.set_index('Date')
    d_df = temp.resample('D').sum()
    d_df = d_df.reset_index().dropna()
    d_df.columns = ['ds','y']
```

### Description of Preliminary Feature Engineering and Preliminary Feature Selection (including decision-making process)
From the options available from the JPX dataset in the machine learning models, it was decided that only three of the twelve columns of data were necessary.  This included the date, security code, and the closing price of the stock.  The best predictor to determine future stock price is the closing price.  If enough time is available, there are plans to potentially use the high and low prices of each stock as well as the volume of stock traded.  After reducing the dataset down to three columns, a minmax scaler was applied to the closing price data in order to reduce any bias.  Next, the normalized closing price data was split into training and testing sets.  Sixty-five percent of the data was used for training and thirty-five percent for testing.  A 65%-35% split of data was chosen, as opposed for a more standard 80%-20% split, because time-series forecasting requires more values in the testing set.  The screenshot below shows the calculation of the root-mean-square-error (RMSE) modeling one stock using a 65%-35% split.   Because the training and testing RMSE values are so close, it indicates that neither underfitting or overfitting is occurring. 

![part of python code root-mean-square-error of training and testing data](https://raw.githubusercontent.com/1on1pt/JPX_Tokyo_Stock_Exchange_Prediction/main/Images/preliminary_engineering.png)

The standard Prophet model has only basic time series features available, but the Neural Prophet model has the ability to add in additional choices, including changepoints, epochs, and learning rates. The Neural Prophet was the trained on the available historical dataset and built new forecasts with the code below:

```python
    m = NeuralProphet(
        n_forecasts=56,
        n_lags=60,
        n_changepoints=50,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        batch_size=56,
        epochs=25,
        learning_rate=1.0,
    )
        
    metrics = m.fit(d_df, freq="D")
    
    future = m.make_future_dataframe(d_df, periods=56, n_historic_predictions=len(df_grouped))
    forecast = m.predict(future)
    forecast['SecuritiesCode'] = i
    forecast_all = pd.concat((forecast_all, forecast))
  
```



### Explanation of Model Choice (including limitations and benefits)

Three different models were chosen to forecast stock prices on the Tokyo Stock Exchange.  The first, Stacked LSTM was chosen because it is a stable and well-documented technique for challenging time-series prediction problems. For our task of predicting stock prices, a Stacked LSTM model creates a hierarchical representation of our time-series data where each layer's output will be used as an input to a subsequent LSTM layer. This hierarchy of hidden layers enables more complex representation of our time-series data, capturing information at different scales.  Hence, we are able to make reasonable future stock price predictions using historical stock price data.  

The limitation of using a Stacked LSTM is the more complex the model, the greater amount of computational time required.  Ultimately, the Stacked LSTM had to be limited to 15 epochs and 100 input parameters in order to complete the prediction of 2000 stocks over two months in a reasonable time period (12 hours).  Additionally, Stacked LSTM models are easily underfitted or overfitted if the training and testing data are not distributed properly.   

The Prophet model was chosen for the ease with which it creates time series forecasts. Generally, the model is applied to a single time series, but it was able to loop over all 2000 codes in just under 2 hours once built into a for loop. The main limitation of the model is that it only considers the time series variables, including seasonality at different aggregations, holidays, and overall trends. Prophet cannot consider other, outside variables that may influence the stock close prices, such as company performance and general market fluctutation. 

The Neural Prophet model was chosen for its ability to expand on the ease of the standard Prophet model to include deep learning and running multiple forecasts quickly. With the model set to run 56 different forecasts with 25 epochs for the 200 highest-performing stocks from the Prophet results, Neural Prophet ran those results in around 15-20 minutes. The model's limitation is similar to Prophet in that it only considers time series variables; though, through its different forecast iterations, Neural Prophet has higher predictive power than Prophet alone.

### Explanation of Changes in Model Choice (if any)

The Prophet model was largely abandonded in favor of the more robust NeuralProphet model. The NeuralProphet model was downsized in scope to avoid the more involved AR Net, regreessor model. As a result, the parameters passed to the NeuralProphet model were reduced to just the number of epochs per training and the learning rate. For performance and time considerations, epochs were set to 50 isntead of the default 'automatic,' which tended to try 150-200 epochs by default. The learning rate was set to 1.0 to help accelerate fitting the modeel but also not over-fitting the model. Overall, the changes to the NeuralProphet model were minimal and mostly to improve performance and output. 

In order to make the Stacked LSTM more efficient and reduce the computational time required, the number of epochs was reduced from 100 to 15.  Consistent testing of the model showed the loss and validation loss, the error evaluated during training a model and during validation, did not significantly change after fifteen epochs was reached.  For stocks with rapid changes in the fluctuation of price, a 150-epoch model did achieve minor improvement in prediction performance.  But given the wish to predict the performance of 2000 stocks, the majority of which had modest to lost changes in performance, the 15-epoch model performed well and cut computational time down dramatically from runs needing days to hours for completion.  

### Description of How Model has been Trained

The NeuralProphet model was trained on all stock closing prices. The training also included trials with adding Japanese holidays to account for the trading closures, but the model ultimately picked up on them better without explicitly naming them in the training data. Additionally, training included looking for seeasonality trends at different aggregates: daily, weekly, and yearly. Ultimately, NeuralProphet sufficiently picked up on those trends independent of manually setting specific seasonality parameters or Fourier orders. 

The model was trained to reduce loss factors, as shown below:
![](https://raw.githubusercontent.com/1on1pt/JPX_Tokyo_Stock_Exchange_Prediction/main/Images/NP_live_loss.png)

NeuralProphet was then able to plot the different training forecasts against the historical data to reduce loss and improve accuracy, as shown below:
![](https://raw.githubusercontent.com/1on1pt/JPX_Tokyo_Stock_Exchange_Prediction/main/Images/NP_training_plot.png)

NeuralProphet also generated seasonal trends and auto-regression trends from the training:

![](https://raw.githubusercontent.com/1on1pt/JPX_Tokyo_Stock_Exchange_Prediction/main/Images/NP_plot_components.png)

The Stacked LSTM model was trained and tested on approximately four years of historical stock data.  Of the stock data provided, only the closing price values of the stocks were used.  Sixty-five percent of the data was used for training and thirty-five percent for testing.  A 65%-35% split of data was chosen because it achieved close training and testing root-mean-square-error (RMSE), which indicates that neither underfitting or overfitting is occurring.  The closing price versus trading day graph below shows an example of how the Stacked LSTM model trains and tests data.  The blue data show the actual closing stock price of TSE #6146 (Disco Corporation), while the orange data shows the training data and green data are the testing data.  Visual inspection shows the Stacked LSTM model does a good job modeling this stock.       

![graphs showing modeling of TSE 6146](https://github.com/1on1pt/JPX_Tokyo_Stock_Exchange_Prediction/blob/f5ccef48abfb1b77a59270275299fbfd06be1924/Images/stacked_lstm_train_test_tse6146.png)

### Description of Current Accuracy Score

During the cross-validation testing, the NeuralProphet model generated the following accuracy score for Stock 6861:

```python
MAE
	MAE              	 (min: 3820.626, max: 216478.183, cur: 3820.626)
RMSE
	RMSE             	 (min: 4994.408, max: 268662.810, cur: 4994.408)
RegLoss
	RegLoss          	 (min:    0.000, max:    0.000, cur:    0.000)
log-SmoothL1Loss
	log-SmoothL1Loss 	 (min:   -5.169, max:    1.420, cur:   -5.169)
```

Given that a time-based series is being used to make predictions, the accuracy of the Stacked LSTM model can be determined by using the root-mean-square-error (RMSE) of the training versus testing data.  Additionally, the mean-absolute-deviation (MAE) of the training versus testing data also provides a way to determine how accurate the model is.  A screenshot of the code used to check the performance of the Stacked LSTM model on Tokyo Stock Exchange #7974 (Nintendo) is shown below.  The RMSE values of the train and test sets are 59,324 and 55,701, respectively, giving a ratio of 0.94.  A ratio of 1.00 would indicate a perfect model; hence, our ratio of 0.94, while not perfect, is a great one.  Furthermore, the MAE of our training and test sets are 58,766 and 55,259, respectively, once again giving a ratio of 0.94. 

![part of python code root-mean-square-error and mean-absolute error of training and testing data](https://github.com/1on1pt/JPX_Tokyo_Stock_Exchange_Prediction/blob/33e34c05ff7ce3d3643fbc8a24e6e381abd0f01c/Images/stacked_lstm_performance_tse7974.png)

**Back to [Quick Links](#quick-links)**

## Database

### Database Stores Static Data for Use During the Project
The static data is the Japanese stock information for the years of 2017 thru 2021. That data was obtained and stored within a .csv file. The .csv files were uploaded into Postgres SQL tables via AWS after the data was cleaned.  Here are the two "clean" .csv files that will be used for this project:
* https://github.com/1on1pt/JPX_Tokyo_Stock_Exchange_Prediction/blob/main/Resources/financials_clean.csv
* https://github.com/1on1pt/JPX_Tokyo_Stock_Exchange_Prediction/blob/main/Resources/prices_clean.csv

After the **Extract** and **Transform** processes were done on the Japanese stock data, the **Load** process was completed to get the data into Postgres SQL for use in the machine learning tools. An RDS database, within AWS, was created, which in turn produced an endpoint for sharing with the team members to allow them access to the database.

### Database Interfaces with the Project in Some Format (database connects to the model)
The database for this project will connect with the machine learning models via AWS RDS.


### Includes at Least Two Tables
Three tables were created for this project:
1. financials_table

![financials_table](https://user-images.githubusercontent.com/94148420/169620559-af69d5d4-1062-4979-9666-187301dba186.PNG)

2. prices_table

![prices_table](https://user-images.githubusercontent.com/94148420/169620582-2aa37b68-7ca6-41e4-a17c-a94690ab5501.PNG)


3. materials_change
* This is the join, see below.

### Includes at Least One Join Using Database Language
The join was created to bring together some data from the prices and the financials tables. The 'securitiescode' column was used to 'JOIN' 'ON' and an 'INNER JOIN' was utilized so that the fields of both tables would be reflected. The materials_change table (view) is below:

![query_join](https://user-images.githubusercontent.com/94148420/169620737-a6ef511c-4df0-46e9-952c-ae4c6607ccf3.PNG)


![join_table](https://user-images.githubusercontent.com/94148420/169620762-2cccfad0-81bb-46ae-a09f-e92b39874509.PNG)



### Includes at Least One Connection String
The database tables from pgAdmin 4 were connected with AWS RDS via a connection string using pyspark.

Here is the connection for **prices**:

![prices_pyspark](https://user-images.githubusercontent.com/94148420/169187396-3a37bd1d-a992-412d-bfc0-8d18d0cf39dc.PNG)


![prices_rds](https://user-images.githubusercontent.com/94148420/169187427-0a7b9c0b-67e7-4782-b382-2351aefa82be.PNG)


And the connection for **financials**:

![financials_pyspark](https://user-images.githubusercontent.com/94148420/169187466-371f0d36-0929-4a58-96b4-d61c9293a4f5.PNG)


![financials_rds](https://user-images.githubusercontent.com/94148420/169187489-e927e8ae-1948-4c73-bf4a-0f409ca7b5e1.PNG)


### ERD
The connection between the **financials_table** and the **prices_table** is the **securitiescode** column.

![ERD_financials_prices_tables](https://user-images.githubusercontent.com/94148420/169694854-7a5a5f7a-8223-4aa0-ac83-376723635073.PNG)

**Back to [Quick Links](#quick-links)**

## Dashboard
### Storyboard on Google Slides
![storyboard](https://github.com/1on1pt/JPX_Tokyo_Stock_Exchange_Prediction/blob/7bd00f302723e24051ea07448f9bb07fa3039c3d/Images/Updated_Storyboard.png)


### Description of the Tools That Will Be Used to Create the Final Dashboard
#### Tableau
Tableau will be used to visualize the dashboards for this project.  Tableau Public will be used locally to create the dashboards, then the visualizations will be shared via the Tableau Server.  There are three primary formats within the Tableau environment:
* **Worksheets -** The building blocks of the visualizations from which dashboards and stories are created.
* **Dashboards -** The collection of worksheets formatted to present data in a way that is easy to read and understand.
* **Stories -** A collection of *dashboards* that includes narration of what is occurring with the data.


### Description of the Interactive Elements
Two interactive dashboards will be created in Tableau:
1. Ability for the user to select a *specific stock* and *date* within the **Stacked LSTM model**

2. Ability for the user to select a *specific stock* and *date* within the **NeuralProphet model**


### Images from Initial Analysis
![Candlestick](https://github.com/1on1pt/JPX_Tokyo_Stock_Exchange_Prediction/blob/7bd00f302723e24051ea07448f9bb07fa3039c3d/Images/Candlestick.png)
![ForecastEarningsVsNetSales](https://github.com/1on1pt/JPX_Tokyo_Stock_Exchange_Prediction/blob/7bd00f302723e24051ea07448f9bb07fa3039c3d/Images/NetSales_ForecastEarnings.png)
![MaterialChangeVsPrice](https://github.com/1on1pt/JPX_Tokyo_Stock_Exchange_Prediction/blob/7bd00f302723e24051ea07448f9bb07fa3039c3d/Images/MaterialChange_Price.png)

### Data (images or report) from the Machine Learning Task
![LSTMPredictions](https://github.com/1on1pt/JPX_Tokyo_Stock_Exchange_Prediction/blob/7bd00f302723e24051ea07448f9bb07fa3039c3d/Images/LSTM_Predictions.png)
![NeuralProphetPredictions](https://github.com/1on1pt/JPX_Tokyo_Stock_Exchange_Prediction/blob/7bd00f302723e24051ea07448f9bb07fa3039c3d/Images/Neural_Prophet_Predictions.png)

### At Least One Interactive Element
![FilterImage](https://github.com/1on1pt/JPX_Tokyo_Stock_Exchange_Prediction/blob/153ccb0b7895f74be6fad06357a2e9bef392a4df/Images/Filter_Screenshot.png)

**Back to [Quick Links](#quick-links)**

## Summary
### Recommendations for Future Analysis
1. Consider different machine learning models
2. Consider various hybrid model combinations
3. Pull *forecast earnings* into machine learning predictors for a more robust model
4. Create dashboards through *Flask* for a different user interactive experience
5. Convert Tokyo Stock Exhchange from *yens* to *US dollars*

### What Would We Have Done Differently
1. Consider a project with smaller datasets
2. Consider NYSE datasets
3. Consider connecting to the database sooner in the project

**Back to [Quick Links](#quick-links)**
