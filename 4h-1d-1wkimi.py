import requests
import pandas as pd
import datetime
import time
import os
import random
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Binance API参数
base_url = "https://api.binance.com/api/v3/klines"
symbol = "BTCUSDT"  # BTC/USDT交易对
intervals = ["4h", "1d", "1w"]  # 4小时、日线、周线三种时间级别
end_time = int(datetime.datetime.now().timestamp() * 1000)
limit = 1000  # 每次请求最多1000条数据

# 列名
columns = ["Open Time", "Open", "High", "Low", "Close", "Volume",
           "Close Time", "Quote Asset Volume", "Number of Trades",
           "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore"]

# 创建输出目录
output_dir = "btc_data"
os.makedirs(output_dir, exist_ok=True)

# 创建一个带有重试机制的会话
def create_session():
    session = requests.Session()
    retry = Retry(
        total=5,  # 最多重试5次
        backoff_factor=1,  # 重试之间的延迟时间会按指数增长
        status_forcelist=[429, 500, 502, 503, 504],  # 这些HTTP状态码会触发重试
        allowed_methods=["GET"]  # 只对GET请求进行重试
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

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

# 函数：获取指定时间段的K线数据，带有重试机制
def get_historical_data(start, end, interval_type, session=None):
    if session is None:
        session = create_session()
        
    params = {
        "symbol": symbol,
        "interval": interval_type,
        "startTime": start,
        "endTime": end,
        "limit": limit
    }
    
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = session.get(base_url, params=params, timeout=30)
            response.raise_for_status()  # 检查请求是否成功
            data = response.json()
            return data
        except Exception as e:
            retry_count += 1
            wait_time = retry_count * random.uniform(1.0, 3.0)
            print(f"请求失败 (尝试 {retry_count}/{max_retries}): {e}")
            print(f"等待 {wait_time:.1f} 秒后重试...")
            time.sleep(wait_time)
    
    print(f"已达到最大重试次数，跳过当前请求")
    return []

# 函数：下载指定时间间隔的所有历史数据
def download_all_historical_data(interval_type):
    print(f"\n开始下载 {interval_type} 数据...")
    
    # 文件名格式：BTC_4h.csv, BTC_1d.csv, BTC_1w.csv
    filename = os.path.join(output_dir, f"BTC_{interval_type}.csv")
    
    # 检查文件是否存在，获取上次最后更新时间
    last_time = get_last_close_time(filename)
    if last_time is not None:
        start_time = last_time + 1  # 从下一个时间戳开始拉取
        print(f"续接上一次下载，起始时间：{datetime.datetime.fromtimestamp(start_time/1000).strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        # 默认起始时间设置为比特币上线以来足够早的时间
        start_time = int(datetime.datetime(2017, 1, 1).timestamp() * 1000)  # 改为2017年，更合理的起始时间
        print(f"初始化下载，起始时间：2017-01-01")
    
    all_data = []
    current_start_time = start_time
    session = create_session()
    
    # 根据时间间隔设置不同的请求等待时间
    if interval_type == "4h":
        wait_time = random.uniform(2.0, 3.0)  # 4小时数据量大，等待时间长一些
    else:
        wait_time = random.uniform(1.0, 2.0)

    # 分段下载
    while current_start_time < end_time:
        data = get_historical_data(current_start_time, end_time, interval_type, session)
        if not data:
            print(f"未获取到 {interval_type} 数据，可能已经到达最新数据或API限制")
            break
        
        # 提取最新的Close Time
        last_close_time = data[-1][6]
        all_data.extend(data)
        
        # 更新起始时间
        current_start_time = last_close_time + 1
        
        # 打印下载进度
        datetime_str = datetime.datetime.fromtimestamp(last_close_time/1000).strftime('%Y-%m-%d %H:%M:%S')
        current_count = len(all_data)
        print(f"已拉取 {interval_type} 数据 {len(data)} 条 (总计: {current_count})，当前Close Time：{datetime_str}")
        
        # 防止频繁请求被封IP，每次请求后等待一段时间
        print(f"等待 {wait_time:.1f} 秒后继续下载...")
        time.sleep(wait_time)
        
        # 每5000条数据保存一次，防止数据丢失
        if current_count % 5000 == 0:
            print(f"已下载 {current_count} 条 {interval_type} 数据，临时保存...")
            temp_df = pd.DataFrame(all_data, columns=columns)
            temp_df["Open Time"] = pd.to_datetime(temp_df["Open Time"], unit="ms")
            temp_df["Close Time"] = pd.to_datetime(temp_df["Close Time"], unit="ms")
            temp_filename = os.path.join(output_dir, f"BTC_{interval_type}_temp.csv")
            temp_df.to_csv(temp_filename, index=False)
    
    if all_data:
        # 转换为DataFrame
        df = pd.DataFrame(all_data, columns=columns)
        
        # 转换时间格式
        df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
        df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms")
        
        # 将数值列转换为数值类型
        numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
        
        # 保存数据
        # 如果文件存在，则追加数据
        if os.path.exists(filename) and last_time is not None:
            try:
                existing_df = pd.read_csv(filename)
                merged_df = pd.concat([existing_df, df], ignore_index=True)
                
                # 去除可能的重复数据
                merged_df.drop_duplicates(subset=["Open Time"], keep="last", inplace=True)
                
                # 按时间排序
                merged_df.sort_values("Open Time", inplace=True)
                
                # 保存合并后的数据
                merged_df.to_csv(filename, index=False)
                print(f"{interval_type} 数据已更新，总计 {len(merged_df)} 条记录")
                
                # 清理临时文件
                temp_filename = os.path.join(output_dir, f"BTC_{interval_type}_temp.csv")
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                
                df = merged_df  # 使用合并后的数据创建兼容版本
            except Exception as e:
                print(f"合并数据出错: {e}")
                print("尝试保存新下载的数据...")
                df.to_csv(filename, index=False)
        else:
            # 新文件，直接保存
            df.to_csv(filename, index=False)
            print(f"{interval_type} 数据已保存，总计 {len(df)} 条记录")
        
        # 创建兼容版本（用于回测系统）
        create_compatible_csv(df, interval_type)
        
        # 删除临时文件
        temp_filename = os.path.join(output_dir, f"BTC_{interval_type}_temp.csv")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        
        return len(all_data)
    else:
        print(f"{interval_type} 没有新数据需要更新")
        return 0

def create_compatible_csv(df, interval_type):
    """创建兼容回测系统的CSV文件格式"""
    try:
        # 复制数据框
        df_compatible = df.copy()
        
        # 转换列名为小写
        df_compatible.columns = [c.lower() for c in df_compatible.columns]
        
        # 重命名列
        df_compatible = df_compatible.rename(columns={
            "open time": "date", 
            "close time": "close_time"
        })
        
        # 只保留必要的列
        df_compatible = df_compatible[["date", "open", "high", "low", "close", "volume"]]
        
        # 保存兼容版本
        compatible_filename = os.path.join(output_dir, f"BTC_{interval_type}_compatible.csv")
        df_compatible.to_csv(compatible_filename, index=False)
        print(f"已创建兼容版本: {compatible_filename}")
    except Exception as e:
        print(f"创建兼容版本出错: {e}")

# 恢复下载功能
def resume_download(interval_type):
    """从临时文件恢复下载"""
    temp_filename = os.path.join(output_dir, f"BTC_{interval_type}_temp.csv")
    if os.path.exists(temp_filename):
        try:
            temp_df = pd.read_csv(temp_filename)
            last_close_time = get_last_close_time(temp_filename)
            if last_close_time:
                print(f"发现 {interval_type} 的临时数据，从 {datetime.datetime.fromtimestamp(last_close_time/1000).strftime('%Y-%m-%d %H:%M:%S')} 继续下载")
                return last_close_time + 1
        except Exception as e:
            print(f"读取临时文件失败: {e}")
    return None

# 主函数
def main():
    print("开始下载比特币多时间级别历史数据...")
    start_time = time.time()
    
    total_records = 0
    
    # 按顺序下载所有时间级别的数据
    for interval in intervals:
        # 检查是否有临时文件可以恢复
        resume_point = resume_download(interval)
        if resume_point:
            # TODO: 实现从临时文件恢复下载的逻辑
            pass
        
        records = download_all_historical_data(interval)
        total_records += records
        
        # 时间间隔添加随机延迟，避免API限制
        if interval != intervals[-1]:  # 不是最后一个间隔
            delay = random.uniform(5.0, 10.0)
            print(f"等待 {delay:.1f} 秒后下载下一个时间级别...")
            time.sleep(delay)
    
    # 计算总耗时
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    
    print(f"\n数据下载完成，总共下载 {total_records} 条记录")
    print(f"总耗时: {hours}小时 {minutes}分钟 {seconds:.2f}秒")
    
    print("\n文件保存在以下位置：")
    for interval in intervals:
        print(f"• {os.path.join(output_dir, f'BTC_{interval}.csv')} (原始格式)")
        print(f"• {os.path.join(output_dir, f'BTC_{interval}_compatible.csv')} (回测兼容格式)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断，程序已停止")
    except Exception as e:
        print(f"\n程序执行出错: {e}")
    finally:
        print("\n程序已结束")
