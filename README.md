# JPX_Tokyo_Stock_Exchange_Prediction

![tokyo-stock-exchange](https://user-images.githubusercontent.com/94148420/166913015-6cb22041-0dd8-48f7-9b79-f8fef31ab4b9.jpg)



## Presentation
### Selected Topic
Using machine learning models to predict the performance of stocks from the JPX Tokyo Stock Exchange and rank the stocks from highest to lowest expected returns.  

### Reason for Selected Topic
The data scientists in our group are interested in exploring quantitative trading where decisions to buy or sell stocks are made based on predictions from trained models.  Historically, finance decisions have been made manually by professionals who decide whether a stock or derivative is undervalued or overvalued.  Our goal is to use machine learning to quickly evaluate a large set of financial data in order to make a portfolio of predicted stock outcomes.

### Description of Source of Data
This dataset contains historic data for a variety of Japanese stocks and options obtained from the Japan Exchange Group.  The Japan Exchange Group, Inc. (JPX) is a holding company operating one of the largest stock exchanges in the world, Tokyo Stock Exchange (TSE).  In the dataset are the financial data of around 2,000 stocks traded on the TSE, which includes information on the opening and closing prices of the stocks, daily high and low prices, and the volume of the stock transactions.  

### Questions to Answer with Data
1.	How accurately can machine learning models predict the outcome of stocks using historical data?  
2.	Which machine learning model makes the most accurate predictions of the stock market? 
3.	Are hybrid machine learning models more accurate than one simple model?


## GitHub
### Main Branch
* Includes README.md

### README.md Must Include
* Description of communication protocol
    * Via Slack on the **group-project-channel**
    * Via Slack with **direct messaging**
    * Use of **email**
    * Use of **Zoom** meetings as necessary
    * During **Zoom in-class sessions**

### Individual Branches
* At least one branch for each team member
    * Each team member will add their own branch to the respository

* Each team member has at least four commits from the duration of first segment
    * To be tracked throughout the duration of this project

## Machine Learning
Three different models will be used to make TSE stock predictions.

##### Stacked LSTM 

Built upon the original long short-term memory model or LSTM model, the Stacked LSTM has multiple hidden LSTM layers and each layer contains multiple memory cells.  An LSTM layer creates a sequence output rather than a single output value.  Hence, for every input time step there is an output time.  This makes it ideal for making time-based stock market predictions. A Stacked LSTM model will be created using the Keras Python deep learning library.   

##### Prophet

Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well (https://github.com/facebook/prophet).

##### Neural Prophet

A Neural Network based Time-Series model, inspired by [Facebook Prophet](https://github.com/facebook/prophet) and [AR-Net](https://github.com/ourownstory/AR-Net), built on PyTorch. NeuralProphet is a hybrid forecasting framework based on PyTorch and trained with standard deep learning methods, making it easy for developers to extend the framework. Local context is introduced with auto-regression and covariate modules, which can be configured as classical linear regression or as Neural Networks. Otherwise, NeuralProphet retains the design philosophy of Prophet and provides the same basic model components (https://github.com/ourownstory/neural_prophet).  


## Database
A **provisional database** has been developed using PostgreSQL 11 within the pgAdmin 4 environment.

This database contains five tables:
1. stock_list
   * Primary Key = securities_code (integer)

2. stock_price
   * Primary Key = row_id (character varying)
   * Foreign Key = securities_code (integer)

3. stock_options
   * Primary Key = date_code (character varying)

4. financials
   * Primary Key = disclosure_num (integer)
   * Foriegn Key = date_code (character varying)

5. trades
   * Primary Key = publisheddate (date)
   * Primary Key = section (character varying)

### Entity Relationship Diagrams (ERDs)
The following are the provisional ERDs:
* **stock_list & stock_price**

![ERD_stocklist_stockprice](https://user-images.githubusercontent.com/94148420/167268629-a31c1dae-83c0-49d2-b0dc-522efff9f691.PNG)


* **stock_option & financials**

#### *Showing the Relationship*

![ERD_stockoptions_financials_short](https://user-images.githubusercontent.com/94148420/167268666-08f5a6fc-c079-462c-ad25-7ca8e069e920.PNG)


#### *The Complete Tables*

![ERD_stockoptions_financials_long](https://user-images.githubusercontent.com/94148420/167268663-f20238b7-126a-42ba-a843-e8c3c5af5b4f.PNG)

### Sample Data that Mimics the Expected Final Database Structure or Schema



### Draft Machine Learning Model is Connected to the Provisional Database



## Dashboard
* N/A this week




