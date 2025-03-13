import yfinance as yf
import pandas as pd
import time
import random
from notion_client import Client
import json
import numpy as np

# 公司列表，格式：名称, ticker, 英文名称（可供参考）
companies = [
    ("小米集团 - W", "1810.HK", "Xiaomi Corporation"),
    ("京东集团 - SW", "9618.HK", "JD.com, Inc."),
    ("阿里巴巴 - W", "9988.HK", "Alibaba Group Holding Limited"),
    ("腾讯控股", "0700.HK", "Tencent Holdings Limited"),
    ("中芯国际", "0981.HK", "Semiconductor Manufacturing International Corporation"),
    ("美团 - W", "3690.HK", "Meituan"),
    ("快手 - W", "1024.HK", "Kuaishou Technology"),
    ("理想汽车 - W", "2015.HK", "Li Auto Inc."),
    ("网易 - S", "9999.HK", "NetEase Inc."),
    ("小鹏汽车 - W", "9868.HK", "Xpeng Inc."),
    ("携程集团 - S", "9961.HK", "Ctrip.com International Limited"),
    ("联想集团", "0992.HK", "Lenovo Group Limited"),
    ("海尔智家", "6690.HK", "Haier智家股份有限公司"),
    ("百度集团 - SW", "9888.HK", "Baidu, Inc."),
    ("舜宇光学科技", "2382.HK", "Sunny Optical Technology Group Company Limited"),
    ("商汤 - W", "0020.HK", "SenseTime Group Inc."),
    ("哔哩哔哩 - W", "9626.HK", "Bilibili Inc."),
    ("美的集团", "0300.HK", "Midea Group"),
    ("金山软件", "3888.HK", "Jinshan Software Company Limited"),
    ("京东健康", "6618.HK", "JD Health International Inc."),
    ("比亚迪电子", "0285.HK", "Build Your Dreams Electronic Device Company Limited"),
    ("同程旅行", "0780.HK", "Tongcheng Travel International Holdings Limited"),
    ("ASMPT", "0522.HK", "ASMPT Company Limited"),
    ("金蝶国际", "0268.HK", "Kingdee International Software Group Company Limited"),
    ("阿里健康", "0241.HK", "Alibaba Health Care Holdings Limited"),
    ("华虹半导体", "1347.HK", "Hua Hong Semiconductor Limited"),
    ("阅文集团", "0772.HK", "Yuewen Group"),
    ("蔚来 - SW", "9866.HK", "NIO Inc."),
    ("众安在线", "6060.HK", "ZhongAn Online P&C Insurance Company Limited"),
    ("东方甄选", "1797.HK", "Eastern Select Group Company Limited")
]

# 修改需要获取的指标及其对应 Yahoo Finance 的键
metrics_keys = {
    "Market Cap": "marketCap",
    "Enterprise Value": "enterpriseValue",
    "Trailing P/E": "trailingPE",
    "Forward P/E": "forwardPE",
    # "PEG Ratio (5yr expected)": "pegRatio",  # 已删除
    "Price/Sales (ttm)": "priceToSalesTrailing12Months",
    "Price/Book (mrq)": "priceToBook",
    "Enterprise Value/Revenue": "enterpriseToRevenue",
    "Enterprise Value/EBITDA": "enterpriseToEbitda",
    "Profit Margin": "profitMargins",
    "Return on Assets (ttm)": "returnOnAssets",
    "Return on Equity (ttm)": "returnOnEquity",
    "Revenue (ttm)": "totalRevenue",
    "Revenue Per Share (ttm)": "revenuePerShare",  # 新增
    "Quarterly Revenue Growth (yoy)": "revenueGrowth",  # 新增
    "Gross Profit (ttm)": "grossProfits",  # 新增
    "EBITDA": "ebitda",  # 新增
    "Net Income Avi to Common (ttm)": "netIncomeToCommon",
    "Quarterly Earnings Growth (yoy)": "earningsGrowth",  # 新增
    "Diluted EPS (ttm)": "trailingEps",
    "Total Cash (mrq)": "totalCash",
    "Total Debt/Equity (mrq)": "debtToEquity",
    "Forward Annual Dividend Rate 4": "dividendRate"  # 新增
    # "Levered Free Cash Flow (ttm)": "freeCashflow"  # 已删除
}

# 定义哪些指标需要格式化为百分比（通常为小数，需要乘以100后加 %）
percent_metrics = {
    "Profit Margin",
    "Return on Assets (ttm)",
    "Return on Equity (ttm)",
    "Quarterly Revenue Growth (yoy)",  # 新增
    "Quarterly Earnings Growth (yoy)"  # 新增
}

# 计算RSI指标的函数
def calculate_rsi(prices, period=14):
    """计算相对强弱指标(RSI)"""
    try:
        # 确保有足够的数据
        if len(prices) < period + 1:
            return None
            
        # 计算价格变化
        deltas = np.diff(prices)
        
        # 分离上涨和下跌
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # 计算平均上涨和下跌
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        # 避免除以零
        if avg_loss == 0:
            return 100
            
        # 计算相对强度
        rs = avg_gain / avg_loss
        
        # 计算RSI
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception as e:
        print(f"计算RSI时出错: {e}")
        return None

def format_large_number(x):
    """将大数值转换为B或M表示，保留2位小数"""
    try:
        if x is None:
            return ""
        if x >= 1e9:
            return f"{x/1e9:.2f}B"
        elif x >= 1e6:
            return f"{x/1e6:.2f}M"
        else:
            return f"{x}"
    except Exception:
        return str(x)

def format_percent(x):
    """将小数转换为百分比字符串，保留2位小数"""
    try:
        if x is None:
            return ""
        return f"{x*100:.2f}%"
    except Exception:
        return str(x)

# 用于存放所有公司数据
data = []

for name, ticker, eng_name in companies:
    print(f"正在获取 {name} ({ticker}) 的数据...")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # 获取历史数据以计算当日收盘价和涨跌幅
        hist = stock.history(period="2d")  # 获取最近两天的数据用于计算涨跌幅
        
        # 获取历史最高价
        max_price_hist = stock.history(period="max")
        historical_high = max_price_hist['High'].max() if not max_price_hist.empty else None
        
        # 计算当日收盘价和涨跌幅
        if not hist.empty and len(hist) >= 1:
            current_price = hist['Close'].iloc[-1]
            
            # 计算涨跌幅
            price_change_pct = None
            if len(hist) >= 2:
                prev_close = hist['Close'].iloc[-2]
                if prev_close > 0:
                    price_change_pct = (current_price - prev_close) / prev_close
                    
            # 计算与历史最高价的差距百分比
            high_price_gap_pct = None
            if historical_high is not None and historical_high > 0 and current_price is not None:
                high_price_gap_pct = (historical_high - current_price) / historical_high
                
            # 计算RSI指标
            rsi_value = None
            rsi_hist = stock.history(period="1mo")  # 获取一个月的数据用于计算RSI
            if not rsi_hist.empty and len(rsi_hist) > 14:  # 确保有足够的数据
                rsi_value = calculate_rsi(rsi_hist['Close'].values)
        else:
            current_price = None
            price_change_pct = None
            high_price_gap_pct = None
            rsi_value = None
        
        company_data = {
            "Company": name,
            "Ticker": ticker,
            "English Name": eng_name,
            "Current Price": f"{current_price:.1f}" if current_price is not None else "",
            "Price Change %": format_percent(price_change_pct) if price_change_pct is not None else "",
            "Historical High": f"{historical_high:.1f}" if historical_high is not None else "",
            "Gap to High %": format_percent(high_price_gap_pct) if high_price_gap_pct is not None else "",  # 新增
            "RSI (14)": f"{rsi_value:.2f}" if rsi_value is not None else ""  # 新增
        }
        
        # 遍历所有指标
        for metric, key in metrics_keys.items():
            value = info.get(key, None)
            # 针对百分比指标，转换为百分比字符串
            if metric in percent_metrics and value is not None:
                value = format_percent(value)
            # 针对大数值指标（如市值、营收等），做格式化显示
            elif metric in ["Market Cap", "Enterprise Value", "Revenue (ttm)", "Net Income Avi to Common (ttm)", 
                           "Total Cash (mrq)", "Gross Profit (ttm)", "EBITDA"]:  # 更新大数值指标列表
                value = format_large_number(value)
            else:
                # 其他指标，直接保留原值（若为数字可保留2位小数）
                if isinstance(value, (int, float)):
                    value = f"{value:.2f}"
                else:
                    value = value if value is not None else ""
            company_data[metric] = value
        data.append(company_data)
    except Exception as e:
        print(f"获取 {name} ({ticker}) 数据时出错：{e}")
    # 随机延时2到5秒，防止请求频繁
    time.sleep(random.uniform(2, 5))

# 转换为 DataFrame 并保存为 CSV 文件（UTF-8 带 BOM 编码便于 Excel 打开）
df = pd.DataFrame(data)
df.to_csv("htimain.csv", index=False, encoding="utf-8-sig")

print("数据已更新并保存至 htimain.csv")

# 上传数据到Notion
def upload_to_notion(df, notion_api_key, database_id):
    """将数据上传到Notion数据库"""
    print("开始上传数据到Notion...")
    
    # 初始化Notion客户端
    notion = Client(auth=notion_api_key)
    
    # 定义期望的列顺序
    EXPECTED_COLUMNS = [
        "Company", "Ticker", "English Name", "Current Price", "Price Change %", 
        "Historical High", "Gap to High %", "RSI (14)", "Market Cap", "Enterprise Value", 
        "Trailing P/E", "Forward P/E", "Price/Sales (ttm)", "Price/Book (mrq)", 
        "Enterprise Value/Revenue", "Enterprise Value/EBITDA", "Profit Margin", 
        "Return on Assets (ttm)", "Return on Equity (ttm)", "Revenue (ttm)", 
        "Revenue Per Share (ttm)", "Quarterly Revenue Growth (yoy)", "Gross Profit (ttm)", 
        "EBITDA", "Net Income Avi to Common (ttm)", "Quarterly Earnings Growth (yoy)", 
        "Diluted EPS (ttm)", "Total Cash (mrq)", "Total Debt/Equity (mrq)", 
        "Forward Annual Dividend Rate 4"
    ]
    
    # 创建带序号的列名映射
    NUMBERED_COLUMNS = {}
    for i, column in enumerate(EXPECTED_COLUMNS):
        if column == "Company":  # Company作为标题属性，不需要添加序号
            NUMBERED_COLUMNS[column] = column
        else:
            NUMBERED_COLUMNS[column] = f"{i:02d}. {column}"
    
    # 获取CSV文件的列顺序
    csv_columns = list(df.columns)
    print(f"CSV文件的列顺序: {csv_columns}")
    
    # 清空现有数据库内容
    try:
        # 获取现有页面
        response = notion.databases.query(database_id=database_id)
        existing_pages = response.get("results", [])
        
        # 删除现有页面
        for page in existing_pages:
            notion.pages.update(page_id=page["id"], archived=True)
        
        print(f"已清空Notion数据库中的{len(existing_pages)}条记录")
    except Exception as e:
        print(f"清空数据库时出错: {e}")
    
    # 获取数据库结构
    try:
        db_info = notion.databases.retrieve(database_id=database_id)
        existing_properties = db_info.get("properties", {})
        
        # 找出标题属性
        title_property = None
        for prop_name, prop_info in existing_properties.items():
            if prop_info.get("type") == "title":
                title_property = prop_name
                print(f"找到标题属性: {title_property}")
                break
        
        # 如果没有找到标题属性，使用"Company"
        if not title_property:
            title_property = "Company"
            print(f"未找到标题属性，将使用默认值: {title_property}")
        
        # 准备新的属性结构
        properties = {}
        
        # 只保留标题属性
        for prop_name, prop_info in existing_properties.items():
            if prop_info.get("type") == "title":
                properties[prop_name] = prop_info
        
        # 创建带序号的属性
        column_to_numbered = {}  # 原始列名到带序号列名的映射
        
        # 按照期望的顺序添加属性
        for column in EXPECTED_COLUMNS:
            # 如果是Company列且已有标题属性，则使用现有标题属性
            if column == "Company" and title_property:
                column_to_numbered[column] = title_property
                continue
                
            # 使用带序号的列名
            numbered_column = NUMBERED_COLUMNS.get(column, column)
            column_to_numbered[column] = numbered_column
            
            # 如果不是标题属性，则创建
            if column != "Company" or title_property is None:
                if any(keyword in column for keyword in ["Price", "P/E", "P/B", "EPS", "RSI", "Gap", "%", "Market Cap", "Enterprise Value", "Revenue", "EBITDA", "Margin", "Return", "Growth", "Debt", "Dividend", "Cash"]):
                    # 数字属性
                    properties[numbered_column] = {"number": {}}
                elif "Date" in column:
                    # 日期属性
                    properties[numbered_column] = {"date": {}}
                elif column == "Status" or column == "Ticker" or "English Name" in column:
                    # 选择属性
                    properties[numbered_column] = {"select": {}}
                else:
                    # 默认为富文本
                    properties[numbered_column] = {"rich_text": {}}
        
        # 更新数据库结构
        notion.databases.update(
            database_id=database_id,
            properties=properties
        )
        print("已更新数据库结构，添加了带序号的属性")
        
        # 重新获取数据库结构
        db_info = notion.databases.retrieve(database_id=database_id)
        existing_properties = db_info.get("properties", {})
    except Exception as e:
        print(f"获取或更新数据库结构时出错: {e}")
        title_property = "Company"  # 默认使用Company作为标题
        column_to_numbered = NUMBERED_COLUMNS  # 使用默认映射
    
    # 上传新数据
    success_count = 0
    for index, row in df.iterrows():
        try:
            # 准备属性数据
            properties = {}
            
            # 设置标题属性 - 修复这里的问题
            company_value = str(row.get("Company", "未知公司"))
            # 正确设置标题属性
            properties[title_property] = {
                "title": [{"text": {"content": company_value}}]
            }
            
            # 严格按照期望的列顺序处理其他列
            for column in EXPECTED_COLUMNS:
                if column == "Company":
                    continue  # 标题列已处理，无论标题属性名称是什么
                
                # 如果CSV中没有这一列，跳过
                if column not in csv_columns:
                    continue
                
                # 跳过空值
                value = row.get(column)
                if pd.isna(value) or value == "":
                    continue
                
                # 获取带序号的列名
                numbered_column = column_to_numbered.get(column)
                if not numbered_column or (numbered_column not in existing_properties and numbered_column != title_property):
                    print(f"警告: 列 '{column}' (映射为 '{numbered_column}') 在数据库中不存在，已跳过")
                    continue
                
                # 获取属性类型
                prop_type = existing_properties.get(numbered_column, {}).get("type", "rich_text")
                
                # 根据属性类型设置值
                if prop_type == "number":
                    # 数字类型
                    try:
                        # 尝试将值转换为数字
                        if isinstance(value, str):
                            # 移除百分号、B、M等单位
                            clean_value = str(value).replace("%", "").replace("B", "").replace("M", "").replace(",", "")
                            num_value = float(clean_value)
                        else:
                            num_value = float(value)
                        properties[numbered_column] = {"number": num_value}
                    except Exception as e:
                        # 如果转换失败，使用富文本
                        print(f"警告: 无法将 '{column}' 的值 '{value}' 转换为数字: {e}")
                        properties[numbered_column] = {"rich_text": [{"text": {"content": str(value)}}]}
                elif prop_type == "select":
                    # 选择类型
                    properties[numbered_column] = {"select": {"name": str(value)}}
                elif prop_type == "date":
                    # 日期类型
                    try:
                        properties[numbered_column] = {"date": {"start": str(value)}}
                    except:
                        properties[numbered_column] = {"rich_text": [{"text": {"content": str(value)}}]}
                else:
                    # 默认使用富文本
                    properties[numbered_column] = {"rich_text": [{"text": {"content": str(value)}}]}
            
            # 创建页面
            response = notion.pages.create(
                parent={"database_id": database_id},
                properties=properties
            )
            success_count += 1
            
            # 每10条记录显示一次进度
            if success_count % 10 == 0 or success_count == len(df):
                print(f"已上传 {success_count}/{len(df)} 条记录")
        except Exception as e:
            print(f"上传 {row.get('Company', '未知公司')} 数据时出错: {e}")
            # 添加更详细的错误信息
            print(f"错误详情: {str(e)}")
            # 打印当前属性结构，帮助调试
            if index == 0:  # 只打印第一条记录的详细信息
                print(f"属性结构: {json.dumps(properties, ensure_ascii=False)}")
    
    print(f"成功上传 {success_count}/{len(df)} 条记录到Notion")
    return success_count > 0

# Notion API凭据
NOTION_API_KEY = "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"
NOTION_DATABASE_ID = "1b2cc7dd340480e7a93bdff5e868610a"

# 添加错误处理和验证
try:
    # 尝试获取数据库信息以验证ID
    notion = Client(auth=NOTION_API_KEY)
    try:
        db_info = notion.databases.retrieve(database_id=NOTION_DATABASE_ID)
        print(f"成功连接到Notion数据库: {db_info.get('title', [{}])[0].get('plain_text', '未命名数据库')}")
    except Exception as e:
        print(f"无法连接到指定的Notion数据库，请检查数据库ID是否正确: {e}")
        print("提示: Notion数据库ID通常是一个长字符串，形如'123e4567-e89b-12d3-a456-426614174000'")
        print("您可以从Notion数据库页面的URL中获取ID")
        print("是否仍要尝试上传数据? (y/n)")
        choice = input().strip().lower()
        if choice != 'y':
            print("已取消上传到Notion")
            exit()

    # 上传数据到Notion
    upload_to_notion(df, NOTION_API_KEY, NOTION_DATABASE_ID)
    print("数据已成功保存到本地CSV文件并上传到Notion数据库")
    
except Exception as e:
    print(f"连接Notion API时出错: {e}")
    print("数据已保存到本地CSV文件，但未能上传到Notion")
