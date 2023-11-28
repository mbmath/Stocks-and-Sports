# Stocks-and-Sports
# NEED TO ADD SOURCES, REFERENCES, and LICENSE (add forward fill and MSE loss)

## Before Using

The following packages must be installed:
- PyTorch
- yfinance

Either conda install or pip install

## Objective

This code attempts to find a correlation between the performance of New York based professional sports teams' performance and the perfomance of the stock market (tracked by the S&P 500). 

The data on Sports Teams can be found in the csv files in the repository and the stock data is pulled using the yfinance package

## Our Findings

It should be noted that we found very little, to no correlation between the two variables. When plotting the predicted stock prices against actual stock prices we find that the............

### If you want to use any other correlations, csv files of the same format as those in the folder must be created
The 'test_sports_sum' and 'train_sports_sum' csv files are what we used to train our project. These are the daily wins and losses summed over time. The 'test_sports' and 'train_sports' csv files are the individual daily wins and losses. They can be used in subsitution for the 'sum' files but are not necessary to run the code.

### To change the Ticker Being Analzyed:
Change the lines of code in lines ___ & ___
  
  Note: ^GSPC is the symbol in yfinance for the S&P 500, all other tickers can be directly copied into the text (DO NOT INCLUDE THE '^' if it is a regular stock)
