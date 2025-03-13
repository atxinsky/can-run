import requests
import pandas as pd
import datetime
import time

# Binance API参数
base_url = "https://api.binance.com/api/v3/klines"
symbol = "BTCUSDT"  # BTC/USDT交易对
interval = "4h"     # 时间间隔为4小时
end_time = int(datetime.datetime.now().timestamp() * 1000)
limit = 1000  # 每次请求最多1000条数据

# 列名
columns = ["Open Time", "Open", "High", "Low", "Close", "Volume",
           "Close Time", "Quote Asset Volume", "Number of Trades",
           "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore"]

def get_last_close_time(filename):
    """从CSV文件中获取最后一条记录的`Close Time`"""
    try:
        df = pd.read_csv(filename)
        last_close_str = df.iloc[-1]["Close Time"]
        # 转换为毫秒时间戳
        last_close_time = int(datetime.datetime.strptime(last_close_str, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
        return last_close_time
    except Exception as e:
        print(f"读取文件失败，可能是文件不存在：{e}")
        return None

# 函数：获取指定时间段的K线数据
def get_historical_data(start, end):
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start,
        "endTime": end,
        "limit": limit
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # 检查请求是否成功
        data = response.json()
        return data
    except Exception as e:
        print(f"请求失败：{e}")
        return []

# 主函数
def main():
    filename = "BTC_4h_2017-2025.csv"
    
    # 获取上次运行的最后时间
    last_time = get_last_close_time(filename)
    if last_time is not None:
        start_time = last_time + 1  # 下一个时间戳开始拉取
    else:
        start_time = int(datetime.datetime(2017, 10, 1).timestamp() * 1000)  # 默认从2017-10-01开始
    
    print(f"开始下载数据，起始时间：{datetime.datetime.fromtimestamp(last_time/1000) if last_time else '初始化'}")
    
    all_data = []
    current_start_time = start_time

    # 分段下载
    while current_start_time < end_time:
        data = get_historical_data(current_start_time, end_time)
        if not data:
            print("未获取到新数据，退出循环")
            break
        
        # 提取最新的Close Time
        last_close_time = data[-1][6]
        all_data.extend(data)
        
        # 更新起始时间
        current_start_time = last_close_time + 1
        
        print(f"已拉取新数据 {len(data)} 条，当前Close Time：{datetime.datetime.fromtimestamp(last_close_time/1000)}")
        time.sleep(1)  # 防止频繁请求被封IP
    
    if all_data:
        # 转换为DataFrame
        df = pd.DataFrame(all_data, columns=columns)
        
        # 转换时间格式
        df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
        df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms")
        
        # 保存数据
        # 如果文件存在，则追加数据
        if last_time is not None:
            existing_df = pd.read_csv(filename)
            updated_df = pd.concat([existing_df, df], ignore_index=True)
        else:
            updated_df = df
        
        updated_df.to_csv(filename, index=False)
        print(f"新数据已追加至 {filename}，共 {len(all_data)} 条记录")
    else:
        print("没有新数据需要更新")

if __name__ == "__main__":
    main()