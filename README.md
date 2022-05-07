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
   * Priimary Key = disclosure_num (integer)
   * Foriegn Key = date_code (character varying)

5. trades
   * Primary Key = publisheddate (date)
   * Primary Key = section (character varying)

### Entity Relationship Diagrams (ERDs)
The following are the provisional ERDs:
* **stock_list & stock_price**

![ERD_stocklist_stockprice](https://user-images.githubusercontent.com/94148420/167268629-a31c1dae-83c0-49d2-b0dc-522efff9f691.PNG)


* **stock_option & financials**

#### <u>Showing the Relationship</u>

![ERD_stockoptions_financials_short](https://user-images.githubusercontent.com/94148420/167268666-08f5a6fc-c079-462c-ad25-7ca8e069e920.PNG)


#### <u>The Complete Tables</u>

![ERD_stockoptions_financials_long](https://user-images.githubusercontent.com/94148420/167268663-f20238b7-126a-42ba-a843-e8c3c5af5b4f.PNG)


## Dashboard
* N/A this week




