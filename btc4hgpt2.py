import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

class MACDStrategy(bt.Strategy):
    # 参数设定
    params = (
        ("period_me1", 12),  # MACD快线
        ("period_me2", 26),  # MACD慢线
        ("period_signal", 9),  # 信号线
        ("stop_loss", 0.02),  # 止损2%
        ("take_profit", 0.22),  # 止盈18%
        ("leverage", 8),  # 杠杆倍数
        ("trade_percent", 0.1),  # 每次交易使用总资金的40%
        ("commission", 0.0005),  # 手续费0.05%
    )
    
    def __init__(self):
        # 定义MACD指标，使用Backtrader标准的参数名称
        self.macd = bt.indicators.MACD(self.data.close, period_me1=self.params.period_me1, period_me2=self.params.period_me2, period_signal=self.params.period_signal)
        self.crossup = bt.indicators.CrossUp(self.macd.macd, self.macd.signal)  # 金叉
        self.crossdown = bt.indicators.CrossDown(self.macd.macd, self.macd.signal)  # 死叉
        self.order = None

    def next(self):
        if self.order:  # 检查是否有未完成的订单
            return
        
        # 判断是否满足多单开仓条件
        if self.macd.macd[0] > 0 and self.crossup[0]:
            cash = self.broker.get_cash()
            size = cash * self.params.trade_percent / self.data.close[0]  # 每次开仓资金的40%
            margin = size / self.params.leverage  # 根据杠杆计算保证金

            # 设置杠杆，并执行买单
            self.buy(size=size, margin=margin)  # 执行买单，并使用杠杆

        # 判断是否满足平仓条件
        if self.position:
            # 止损条件
            if self.data.close[0] < self.position.price * (1 - self.params.stop_loss):
                self.close()  # 止损平仓

            # 止盈条件
            elif self.data.close[0] > self.position.price * (1 + self.params.take_profit):
                self.close()  # 止盈平仓

    def stop(self):
        # 输出策略的最终性能
        print(f"策略完成，总盈利：{self.broker.get_value() - 10000:.2f} USD")

# 定义回测的主程序
def run_backtest(start_date, end_date):
    cerebro = bt.Cerebro()

    # 加载历史数据
    data = bt.feeds.GenericCSVData(
        dataname='BTC_4h_2017-2025.csv',
        fromdate=datetime.strptime(start_date, "%Y-%m-%d"),
        todate=datetime.strptime(end_date, "%Y-%m-%d"),
        dtformat=('%Y/%m/%d %H:%M'),  # 修改时间格式，适配你的数据
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=-1
    )

    cerebro.adddata(data)
    
    # 设置初始现金
    cerebro.broker.set_cash(10000)  # 初始本金
    cerebro.broker.setcommission(commission=0.0005)  # 手续费

    # 设置回测时间范围
    cerebro.addstrategy(MACDStrategy)
    
    # 添加性能分析
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual_return')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')

    results = cerebro.run()
    strategy = results[0]

    # 输出性能指标
    print(f"初始资金: {cerebro.broker.startingcash:.2f}")
    print(f"最终资金: {cerebro.broker.get_value():.2f}")
    print(f"年化收益: {strategy.analyzers.annual_return.get_analysis()}")
    print(f"夏普比率: {strategy.analyzers.sharpe.get_analysis()}")
    print(f"最大回撤: {strategy.analyzers.drawdown.get_analysis()}")
    
    # 获取交易分析结果，输出胜率和盈利亏损次数
    trade_analysis = strategy.analyzers.trade_analyzer.get_analysis()
    total_wins = trade_analysis['won']['total']
    total_losses = trade_analysis['lost']['total']
    win_rate = (total_wins / (total_wins + total_losses)) * 100 if total_wins + total_losses > 0 else 0
    
    print(f"胜率: {win_rate:.2f}%")
    print(f"策略盈利次数: {total_wins}")
    print(f"策略亏损次数: {total_losses}")

    # 可视化结果
    cerebro.plot(style='candlestick')

# 主程序入口
if __name__ == "__main__":
    # 使用 argparse 获取用户输入的开始和结束日期
    parser = argparse.ArgumentParser(description="Backtest Bitcoin MACD Strategy")
    parser.add_argument('--start_date', type=str, default="2017-01-01", help='回测开始日期 (格式: yyyy-mm-dd)')
    parser.add_argument('--end_date', type=str, default="2025-01-01", help='回测结束日期 (格式: yyyy-mm-dd)')
    args = parser.parse_args()

    # 启动回测
    run_backtest(args.start_date, args.end_date)
