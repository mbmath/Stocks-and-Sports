# Stocks-and-Sports

## Before Using

The following packages must be installed:
- PyTorch
- yfinance
Either conda install or pip install

## If you want to use any other correlations, csv files of the same format as those in the folder must be created
The 'test_sports_sum' and 'train_sports_sum' csv files are what we used to train our project. These are the daily wins and losses summed over time. The 'test_sports' and 'train_sports' csv files are the individual daily wins and losses. They can be used in subsitution for the 'sum' files but are not necessary to run the code.

## To change the Ticker Being Analzyed:
Change the lines of code in lines ___ & ___
  Note: ^GSPC is the symbol in yfinance for the S&P 500, all other tickers can be directly copied into the text (DO NOT INCLUDE THE '^' if it is a regular stock)
