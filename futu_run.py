import pandas as pd
from futu import OpenSecTradeContext, TrdMarket, SecurityFirm, RET_OK
from datetime import datetime, timedelta

# 设置连接参数
host = '127.0.0.1'  # Futu OpenD的主机地址
port = 11111        # Futu OpenD的端口
security_firm = SecurityFirm.FUTUSECURITIES  # 证券公司

# 创建交易上下文
trd_ctx = OpenSecTradeContext(host=host, port=port, security_firm=security_firm)

# 定义时间范围
start_date = datetime(2023, 7, 1)
end_date = datetime(2025, 2, 15)

# 初始化交易记录列表
all_trades = []

# 分批获取交易记录
while start_date < end_date:
    # 计算当前批次的结束日期
    batch_end_date = min(start_date + timedelta(days=90), end_date)
    
    # 将日期转换为字符串，格式为“YYYY-MM-DD HH:MM:SS”
    start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
    end_str = batch_end_date.strftime('%Y-%m-%d %H:%M:%S')
    
    # 查询当前时间段的历史成交记录
    ret, data = trd_ctx.history_deal_list_query(start=start_str, end=end_str)
    if ret == RET_OK:
        all_trades.append(data)
    else:
        print(f'查询失败: {data}')
    
    # 更新开始日期为下一个时间段的开始
    start_date = batch_end_date

# 合并所有交易记录
if all_trades:
    df = pd.concat(all_trades, ignore_index=True)
    # 保存到Excel文件
    df.to_excel('富途交易流水.xlsx', index=False)
    print("交易记录已成功导出到 '富途交易流水.xlsx'")
else:
    print("未获取到任何交易记录")

# 关闭交易上下文
trd_ctx.close()
