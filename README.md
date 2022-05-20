# JPX_Tokyo_Stock_Exchange_Prediction

![tokyo-stock-exchange](https://user-images.githubusercontent.com/94148420/166913015-6cb22041-0dd8-48f7-9b79-f8fef31ab4b9.jpg)

##### University of Wisconsin Extension Campus Data Analytics Bootcamp Final Project

### Team Members

| **Member**            | **Primary Role**         | **Responsibilities**                                |
|-------------------------|:------------------------:|-----------------------------------------------------|
|**[Aslesha Vangareddy](https://github.com/AsleshaV)**    |Dashboard                |Manage the development of the dashboard               |
|**[Jerri Morales](https://github.com/jerrimor)**   |Database               |Manage the developement of the database          |
|**[Carl Stewart](https://github.com/CarlS2rt)**       |Maching Learning Model |Manage the developement of the machine learning model|
|**[Eric Himburg](https://github.com/eric-himburg)**    |Machine Learning Model   |Manage the development of the machine learning model|
|**[Nate Millmann](https://github.com/millmannnate)**   |Machine Learning Model   |Manage the development of the machine learning model |
|**[John Beauchamp](https://github.com/1on1pt)**  |GitHub; Database   |Manage GitHub repository; assist with database development |

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
* https://github.com/1on1pt/JPX_Tokyo_Stock_Exchange_Prediction/blob/Jerri/financials_clean.ipynb

![financials_clean_head](https://user-images.githubusercontent.com/94148420/168492280-ee64b2f7-34c7-4f5f-99ab-3d98e8d02349.PNG)

* https://github.com/1on1pt/JPX_Tokyo_Stock_Exchange_Prediction/blob/Jerri/prices_clean.ipynb

![stock_prices_clean_head](https://user-images.githubusercontent.com/94148420/168492410-8b126785-4e41-43fd-b1f5-bc60e043edaf.PNG)


### Description of the Analysis Phase of the Project



### Presentation
The presentation is hosted on Google Slides and can be accessed [here](https://docs.google.com/presentation/d/1YIA2DkOoDofQbNiOO2Xbnr-BA07IlXciiKUgL-yjzWU/edit?usp=sharing).

## GitHub
### Main Branch
* Includes README.md
* Code necessary to perform exploratory analysis


* Some code necessary to complete the machine learning portion of the project
    * https://github.com/1on1pt/JPX_Tokyo_Stock_Exchange_Prediction/blob/main/JPX_Prophet_results_testing.ipynb


### README.md Must Include
* Description of communication protocol
    * Via Slack on the **group-project-channel**
    * Via Slack with **direct messaging**
    * Use of **email**
    * Use of **Zoom** meetings as necessary
    * During **Zoom in-class sessions**

* Outline of the project
![project_outline](https://user-images.githubusercontent.com/94148420/168448850-38caa56f-0355-432f-895e-5d46030d3050.PNG)



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

##### NeuralProphet

A Neural Network based Time-Series model, inspired by [Facebook Prophet](https://github.com/facebook/prophet) and [AR-Net](https://github.com/ourownstory/AR-Net), built on PyTorch. NeuralProphet is a hybrid forecasting framework based on PyTorch and trained with standard deep learning methods, making it easy for developers to extend the framework. Local context is introduced with auto-regression and covariate modules, which can be configured as classical linear regression or as Neural Networks. Otherwise, NeuralProphet retains the design philosophy of Prophet and provides the same basic model components (https://github.com/ourownstory/neural_prophet).  


### Description of Preliminary Data Preprocessing



### Description of Preliminary Feature Engineering and Preliminary Feature Selection (including decision-making process)



### Explanation of Model Choice (including limitations and benefits)





## Database

### Database Stores Static Data for Use During the Project
The static data is the Japanese stock information for the years of 2017 thru 2021. That data was obtained and stored within a .csv file. The .csv files were uploaded into Postgres SQL tables via AWS after the data was cleaned.  Here are the two "clean" .csv files that will be used for this project:
* https://github.com/1on1pt/JPX_Tokyo_Stock_Exchange_Prediction/blob/main/Resources/financials_clean.csv
* https://github.com/1on1pt/JPX_Tokyo_Stock_Exchange_Prediction/blob/main/Resources/prices_clean.csv

After the **Extract** and **Transform** processes were done on the Japanese stock data, the **Load** process was completed to get the data into Postgres SQL for use in the machine learning tools. An RDS database, within AWS, was created, which in turn produced an endpoint for sharing with the team members to allow them access to the database.

### Database Interfaces with the Project in Some Format (database connects to the model)
The database for this project will connect with the machine learning models via AWS RDS.


### Includes at Least Two Tables
Two tables were created for this project:
* financials_table




### Includes at Least One Join Using Database Language



### Includes at Least One Connection String
The database tables from pgAdmin 4 were connected with AWS RDS via a connection string using pyspark.

Here is the connection for **prices**:

![prices_pyspark](https://user-images.githubusercontent.com/94148420/169187396-3a37bd1d-a992-412d-bfc0-8d18d0cf39dc.PNG)


![prices_rds](https://user-images.githubusercontent.com/94148420/169187427-0a7b9c0b-67e7-4782-b382-2351aefa82be.PNG)


And the connection for **financials**:

![financials_pyspark](https://user-images.githubusercontent.com/94148420/169187466-371f0d36-0929-4a58-96b4-d61c9293a4f5.PNG)


![financials_rds](https://user-images.githubusercontent.com/94148420/169187489-e927e8ae-1948-4c73-bf4a-0f409ca7b5e1.PNG)


## Dashboard
### Storyboard on Google Slides
![storyboard](https://github.com/1on1pt/JPX_Tokyo_Stock_Exchange_Prediction/blob/8fb2baf324bb3ccac0757dc0224d3b8e64565a14/Images/Storyboard%20JPX.png)


### Description of the Tools That Will Be Used to Create the Final Dashboard
#### Tableau
Tableau will be used to visualize the dashboards for this project.  Tableau Public will be used locally to create the dashboards, then the visualizations will be shared via the Tableau Server.  There are three primary formats within the Tableau environment:
* **Worksheets -** The building blocks of the visualizations from which dashboards and stories are created.
* **Dashboards -** The collection of worksheets formatted to present data in a way that is easy to read and understand.
* **Stories -** A collection of *dashboards* that includes narration of what is occurring with the data.


### Description of the Interactive Elements
Two interactive dashboards will be created in Tableau:
1. Ability for the user to select a *specific stock* within the **Stacked LSTM model**
2. Ability for the user to select a *specific stock* within the **NeuralProphet model**

