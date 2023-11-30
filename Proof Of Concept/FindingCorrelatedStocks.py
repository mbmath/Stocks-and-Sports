
import yfinance as yf
import pandas as pd

start_date = '2000-01-01'
end_date = '2023-11-27'

stocks = [
  
    'IBM', 'ORCL', 'CRM', 'HPQ', 'QCOM', 'TXN', 'VMW', 'ADBE', 'EBAY',
    'YHOO', 'EA', 'MU', 'CTSH', 'INFY', 'SAP', 'AMAT', 'ADI', 'NTAP', 'ACN',
    'ADSK', 'FISV', 'INTU', 'ATVI', 'MCHP', 'XRX', 'KLAC', 'LRCX', 'WDC', 'CERN',
    'STX', 'DOX', 'AKAM', 'SWKS', 'WU', 'FLT', 'FFIV', 'NOW', 'ZS', 'ANET',
    'GPN', 'TWTR', 'SNPS', 'PAYC', 'TWLO', 'SNOW', 'DDOG', 'AVGO', 'NXPI', 'LITE',
    'GS', 'AXP', 'BLK', 'MET', 'PRU', 'COF', 'SPGI', 'ICE', 'AIG', 'SCHW',
    'AMP', 'MMC', 'NTRS', 'CINF', 'TROW', 'AJG', 'BHF', 'BEN', 'WLTW', 'RJF',
    'IVZ', 'CME', 'NDAQ', 'AFL', 'DFS', 'KEY', 'MTB', 'CFG', 'PBCT', 'HBAN',
    'FRC', 'FITB', 'RF', 'STT', 'CMA', 'WU', 'SYF', 'ALLY', 'ETFC', 'CFG',
    'F', 'TM', 'HMC', 'GM', 'TSLA', 'RACE', 'TATA', 'NSANY', 'FUJHY', 'MZDAY',
    'DDAIF', 'HYMTF', 'POAHY', 'VOW3', 'BAMXF', 'TTM', 'DLPH', 'NIO', 'WKHS', 'LI',
    'PFE', 'MRK', 'UNH', 'ABBV', 'GILD', 'CELG', 'BIIB', 'AMGN', 'LLY', 'BMY',
    'REGN', 'VRTX', 'ISRG', 'ZBH', 'ANTM', 'CNC', 'HCA', 'VRTX', 'DHR', 'BSX',
    'ALXN', 'TMO', 'IDXX', 'IQV', 'BIO', 'BMRN', 'EXAS', 'EW', 'ALGN', 'COO',
    'PG', 'KO', 'PEP', 'TM', 'UL', 'CL', 'KMB', 'GIS', 'EL', 'CLX',
    'MO', 'PM', 'NKE', 'COLM', 'K', 'SJM', 'KR', 'MKC', 'HSY', 'CPB',
    'TSN', 'MNST', 'ADM', 'HRL', 'CL', 'KHC', 'NWL', 'KMB', 'K', 'COTY',
    'WMT', 'HD', 'M', 'TGT', 'COST', 'WBA', 'CVS', 'LULU', 'NKE',
    'ROST', 'DG', 'BBY', 'COST', 'KSS', 'LVS', 'SBUX', 'YUM', 'CMG', 'DPZ',
    'NKE', 'VFC', 'TIF', 'RL', 'FL', 'ROKU', 'AMZN', 'EBAY', 'ETSY', 'BKNG',
    'DIS', 'NFLX', 'SPOT', 'CMCSA', 'T', 'VZ', 'SNE', 'EA', 'ATVI', 'CMG',
    'LVS', 'MSGN', 'VIA', 'DISCA', 'ROKU', 'FOXA', 'AMCX', 'VIAC', 'SIRI', 'NWSA',
    'NLSN', 'IMAX', 'CHTR', 'LUMN', 'NFLX', 'DLB', 'DISH', 'FWONA', 'VIAC', 'CNK',
    'T', 'VZ', 'TMUS', 'CHL', 'BTI', 'ORAN', 'SKM', 'NTT', 'AMX', 'VOD',
    'CHA', 'BCE', 'RCI', 'TEL', 'TLK', 'HTZ', 'AMT', 'SBAC', 'CCI', 'VIAC',
    'XOM', 'CVX', 'RDS-A', 'TOT', 'COP', 'EOG', 'SLB', 'KMI', 'PXD', 'MRO',
    'VLO', 'OXY', 'APA', 'HAL', 'PSX', 'CXO', 'HES', 'DVN', 'NOV', 'WMB',
    'BA', 'LMT', 'RTX', 'NOC', 'GD', 'LHX', 'UTX', 'TDG', 'HII', 'ERJ',
    'NATI', 'HEI', 'TDY', 'AAXN', 'BWXT', 'HEI', 'CW', 'ESLT', 'CWST', 'GIB',
    'BRK-B', 'MA', 'V', 'GOOGL', 'FB', 'AAPL', 'MSFT', 'AMZN', 'TSLA',
    'BABA', 'AMD', 'INTC', 'GS', 'JPM', 'NVDA', 'PYPL', 'BA', 'WMT', 'VZ',
    'CSCO', 'CVS', 'DIS', 'PEP', 'KO', 'PG', 'IBM', 'WFC', 'T', 'BAC',
]
data = yf.download(stocks, start=start_date, end=end_date)['Adj Close']

correlation_matrix = data.corr()

max_correlation = correlation_matrix.unstack().sort_values(ascending=False).drop_duplicates().head(2)
stock1, stock2 = max_correlation.index

print(f"The two stocks with the highest correlation are: {stock1} and {stock2}")
print(f"Correlation coefficient: {max_correlation.values[0]:.4f}")
