import yfinance as yf
import pandas as pd
import talib
import time

# 公司的股票代码列表
stocks = [
    "0700.HK", "9988.HK", "3690.HK", "1810.HK", "9618.HK", "1024.HK", "9888.HK", "9866.HK", "2015.HK",
    "9868.HK", "9626.HK", "6618.HK", "241.HK", "268.HK", "0981.HK", "0020.HK", "9961.HK", "0992.HK",
    "1347.HK", "489.HK", "2382.HK", "3888.HK", "6060.HK", "6690.HK", "09961.HK", "9999.HK", "000333.SZ",
    "00285.HK", "0522.HK", "0772.HK"
]

# 更新表格数据
def get_updated_data(ticker, existing_data):
    try:
        stock = yf.Ticker(ticker)
        
        # 获取基本信息
        info = stock.info
        market_cap = info.get('marketCap', None)
        volume = info.get('volume', None)
        price = info.get('currentPrice', None)
        historical_high = info.get('fiftyTwoWeekHigh', None)
        
        # 计算距离历史最高价的差距比例
        price_diff = (price - historical_high) / historical_high if historical_high else None
        
        # 获取RSI数据
        hist = stock.history(period="1d", interval="1d")
        rsi = talib.RSI(hist['Close'], timeperiod=14)[-1]  # 计算RSI
        
        # 更新已有数据
        existing_data['当日收盘价'] = price
        existing_data['成交额'] = volume
        existing_data['当日涨幅'] = (price - existing_data['当日收盘价'].shift(1)) / existing_data['当日收盘价'].shift(1) * 100 if not existing_data['当日收盘价'].isna().any() else None
        existing_data['当日收盘后rsi值'] = rsi
        existing_data['市值'] = market_cap
        existing_data['历史最高价差距'] = price_diff

        # 其他财务数据更新
        existing_data['PE'] = info.get('trailingPE', None)
        existing_data['PB'] = info.get('priceToBook', None)
        existing_data['毛利率'] = info.get('grossMargins', None)
        existing_data['净利率'] = info.get('netMargins', None)
        existing_data['ROE'] = info.get('returnOnEquity', None)
        existing_data['经营活动现金流'] = info.get('operatingCashflow', None)
        existing_data['自由现金流'] = info.get('freeCashflow', None)
        existing_data['资产负债率'] = info.get('debtToEquity', None)
        existing_data['利息保障倍数'] = info.get('interestCoverage', None)
        existing_data['收入增长率'] = info.get('revenueGrowth', None)
        existing_data['利润增长率'] = info.get('profitMargins', None)

        return existing_data
    
    except Exception as e:
        print(f"Error retrieving data for {ticker}: {e}")
        return None

# 更新所有公司的数据
def update_all_data(df):
    updated_data = []
    for idx, row in df.iterrows():
        ticker = row['公司名称']  # 根据公司名称获取对应的股票代码
        print(f"Updating {ticker}...")
        updated_row = get_updated_data(ticker, row.copy())  # Get updated data and keep original row
        if updated_row is not None:
            updated_data.append(updated_row)
        time.sleep(2)  # Add delay to avoid rate limiting
    return pd.DataFrame(updated_data)

# 加载已有数据并更新
df = pd.read_excel('恒生科技数据更新.xlsx')  # 替换为您文件的路径
updated_df = update_all_data(df)

# 保存更新后的数据
updated_df.to_csv('htimain.csv', index=False)

# 显示更新后的前几行
print(updated_df.head())
