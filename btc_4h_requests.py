import ccxt
import pandas as pd
import time

# 初始化Binance交易所对象
binance = ccxt.binance()

# 定义交易对和时间间隔
symbol = 'BTC/USDT'
timeframe = '4h'

# 获取当前时间戳
end_time = int(time.time() * 1000)

# 获取历史K线数据
def fetch_data(start_time):
    all_ohlcv = []
    while True:
        ohlcv = binance.fetch_ohlcv(symbol, timeframe, since=start_time, limit=1000)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        start_time = ohlcv[-1][0] + 1  # 下一次请求的起始时间
        time.sleep(1)  # 避免请求过于频繁
    data = []
    for ohlc in all_ohlcv:
        data.append({
            'timestamp': pd.to_datetime(ohlc[0], unit='ms'),
            'open': ohlc[1],
            'high': ohlc[2],
            'low': ohlc[3],
            'close': ohlc[4],
            'volume': ohlc[5]
        })
    df = pd.DataFrame(data)
    return df

# 读取现有数据
try:
    df_existing = pd.read_csv('btc_usdt_4h.csv', parse_dates=['timestamp'])
    last_timestamp = df_existing['timestamp'].max()
    start_time = int(last_timestamp.timestamp() * 1000)
except (FileNotFoundError, ValueError):
    # 如果文件不存在或无法解析时间戳，则从最早时间开始
    start_time = 1502942400000  # 2017-08-17 00:00:00 UTC的时间戳

# 获取新数据
df_new = fetch_data(start_time)

# 合并新旧数据
df_combined = pd.concat([df_existing, df_new]).drop_duplicates(subset='timestamp', keep='last')

# 保存数据到CSV文件
df_combined.to_csv('btc_usdt_4h.csv', index=False)

print(f"数据已更新至 {df_combined['timestamp'].max()}")
