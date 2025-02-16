import pandas as pd
import numpy as np
from datetime import datetime

def process_trade_data(file_path):
    try:
        # 读取Excel文件
        print("正在读取Excel文件...")
        df = pd.read_excel(file_path)
        
        # 打印原始列名，便于确认
        print("\n原始数据列名：")
        print(df.columns.tolist())
        
        # 重命名列以匹配处理需求
        column_mapping = {
            '名称': '股票名称',
            '下单时间': '交易时间',
            '方向': '交易方向',
            '成交量': '成交数量',
            '成交金额': '成交金额',
            '成交均价': '成交单价'
        }
        # 重命名列
        df = df.rename(columns=column_mapping)
        
        # 设置手续费（如果原数据中没有，默认为成交金额的万分之三）
        if '手续费' not in df.columns:
            df['手续费'] = df['成交金额'] * 0.0003
        
        # 创建结果列表
        results = []
        
        # 按股票名称分组处理数据
        grouped = df.groupby('股票名称')
        for stock_name, group in grouped:
            # 初始化股票数据字典
            stock_data = {
                '股票名称': stock_name,
                '买入时间': None,
                '卖出时间': None,
                '成交均价':0,
                '成交量': 0,
                '买入总价': 0,
                '卖出总价': 0,
                '盈亏金额': 0,
                '盈亏比例': 0,
                '持仓时间': 0,
                '手续费': group['手续费'].sum()
            }
            
            # 处理买入记录
            buy_records = group[group['交易方向'] == '买入']
            if not buy_records.empty:
                stock_data['买入时间'] = buy_records['交易时间'].min()
                stock_data['买入总价'] = buy_records['成交金额'].sum()
                stock_data['成交量'] = buy_records['成交数量'].sum()
                if stock_data['成交量'] != 0:
                    stock_data['成交均价'] = stock_data['买入总价'] / stock_data['成交量']
            
            # 处理卖出记录
            sell_records = group[group['交易方向'] == '卖出']
            if not sell_records.empty:
                stock_data['卖出时间'] = sell_records['交易时间'].max()
                stock_data['卖出总价'] = sell_records['成交金额'].sum()
            
            # 计算盈亏金额和比例
            stock_data['盈亏金额'] = stock_data['卖出总价'] - stock_data['买入总价']
            if stock_data['买入总价'] != 0:
                stock_data['盈亏比例'] = (stock_data['盈亏金额'] / stock_data['买入总价']) * 100
            
            # 计算持仓时间
            if stock_data['买入时间'] and stock_data['卖出时间']:
                buy_date = pd.to_datetime(stock_data['买入时间'])
                sell_date = pd.to_datetime(stock_data['卖出时间'])
                stock_data['持仓时间'] = (sell_date - buy_date).days
            
            results.append(stock_data)
        
        # 创建结果DataFrame
        result_df = pd.DataFrame(results)
        
        # 设置列的顺序
        columns = ['股票名称', '买入时间', '卖出时间', '成交均价', '成交量', '买入总价', '卖出总价', '盈亏金额', '盈亏比例', '持仓时间', '手续费']
        result_df = result_df[columns]
        
        # 数值格式化
        for col in ['成交均价', '盈亏金额', '盈亏比例', '手续费']:
            result_df[col] = result_df[col].round(2)
        
        # 保存结果
        output_file = '交易汇总结果.xlsx'
        result_df.to_excel(output_file, index=False)
        print(f"\n数据处理完成，结果已保存至：{output_file}")
        
        # 显示数据预览
        print("\n数据预览：")
        print(result_df.head())
        return result_df
    
    except Exception as e:
        print(f"处理过程中出现错误：{str(e)}")
        return None

# 执行数据处理
if __name__ == "__main__":
    file_path = '老虎交易总流水.xlsx'
    result = process_trade_data(file_path)