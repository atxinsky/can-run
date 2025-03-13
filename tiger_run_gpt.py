import requests
import pandas as pd
from tigeropen.common.util.signature_utils import read_private_key
from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.trade.trade_client import TradeClient
from tigeropen.common.consts import SecurityType, Market
from datetime import datetime, timedelta

# 配置客户端
def get_client_config():
    """
    https://quant.itigerup.com/#developer 开发者信息获取
    """
    # 港股牌照需用 props_path 参数指定token路径，如 '/Users/xxx/xxx/', 如不指定则取当前路径
    # 必须使用关键字参数指定 props_path
    client_config = TigerOpenClientConfig(props_path='/Users/tretra/tiger_openapi_config.properties')
    return client_config

client_config = get_client_config()
client_config.license = 'TBNZ'
client_config.env = 'PROD'

# 初始化交易客户端
trade_client = TradeClient(client_config)

# 获取已成交订单列表
def get_all_filled_orders():
    all_orders = []
    start_time = datetime.strptime('2023-07-01', '%Y-%m-%d')
    end_time = datetime.strptime('2025-2-1', '%Y-%m-%d')
    
    while start_time < end_time:
        temp_end_time = min(start_time + timedelta(days=90), end_time)
        orders = trade_client.get_filled_orders(
            sec_type=SecurityType.STK,
            market=Market.ALL,
            start_time=start_time.strftime('%Y-%m-%d'),
            end_time=temp_end_time.strftime('%Y-%m-%d'),
            limit=1000
        )
        all_orders.extend(orders)
        start_time = temp_end_time
    
    return all_orders

# 获取订单数据
orders = get_all_filled_orders()

# 将订单数据转换为DataFrame
orders_data = []
for order in orders:
    order_info = {
        '订单ID': order.id,
        '证券代码': order.contract.symbol,
        '买卖方向': order.action,
        '数量': order.quantity,
        '成交价格': order.avg_fill_price,
        '成交金额': order.filled * order.avg_fill_price,
        '状态': order.status,
        '下单时间': datetime.fromtimestamp(order.order_time / 1000),
        '成交时间': datetime.fromtimestamp(order.trade_time / 1000)
    }
    orders_data.append(order_info)

df = pd.DataFrame(orders_data)

# 保存到Excel文件
df.to_excel('老虎记录.xlsx', index=False)

print("交易记录已保存到 '老虎记录.xlsx'")
