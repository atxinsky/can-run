import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns

# Set styling for plots
plt.style.use('fivethirtyeight')
sns.set_style('darkgrid')

class MACDZeroCrossPullbackStrategy:
    def __init__(self, csv_file):
        """
        Initialize the strategy with the CSV file path.
        
        Parameters:
        -----------
        csv_file : str
            Path to the CSV file containing BTC price data
        """
        self.data = self.load_data(csv_file)
        
    def load_data(self, csv_file):
        """
        Load and prepare data from CSV file.
        """
        # Load data
        df = pd.read_csv(csv_file)
        
        # 检查并转换日期列
        date_columns = ['timestamp', 'date', 'time', 'datetime']
        date_col = None
        
        # 查找可能的日期列
        for col in date_columns:
            if col in df.columns:
                date_col = col
                break
        
        # 如果找到日期列，则转换为日期时间索引
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        else:
            # 如果没有日期列，创建一个基于行号的日期索引
            print("警告: 未找到日期列，使用行号作为索引")
            df.index = pd.date_range(start='2017-01-01', periods=len(df), freq='4H')
        
        # Ensure we have the required columns for the strategy
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Check if columns exist (case-insensitive) and rename if necessary
        for col in required_columns:
            # Find any column that matches the required column name (case-insensitive)
            matches = [c for c in df.columns if c.lower() == col.lower()]
            if matches:
                df.rename(columns={matches[0]: col}, inplace=True)
            else:
                raise ValueError(f"Required column '{col}' not found in the data")
        
        # Ensure the data is sorted by time
        df.sort_index(inplace=True)
        
        return df
    
    def calculate_indicators(self, fast_length=12, slow_length=26, signal_length=9):
        """
        Calculate MACD and other indicators needed for the strategy.
        
        Parameters:
        -----------
        fast_length : int
            Fast EMA period for MACD
        slow_length : int
            Slow EMA period for MACD
        signal_length : int
            Signal line period for MACD
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with calculated indicators
        """
        df = self.data.copy()
        
        # Calculate MACD
        df['ema_fast'] = df['close'].ewm(span=fast_length, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=slow_length, adjust=False).mean()
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['signal'] = df['macd'].ewm(span=signal_length, adjust=False).mean()
        df['histogram'] = df['macd'] - df['signal']
        
        # Calculate additional indicators for signal confirmation
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['sma_20'] = df['close'].rolling(window=20).mean()
        std_dev = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['sma_20'] + (std_dev * 2)
        df['bb_lower'] = df['sma_20'] - (std_dev * 2)
        
        # Drop NaN values from calculations
        df.dropna(inplace=True)
        
        return df
    
    def identify_signals(self, df):
        """
        Identify entry and exit signals based on MACD strategy.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with calculated indicators
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with entry and exit signals
        """
        # 创建一个副本
        signals = df.copy()
        
        # 初始化信号列
        signals['golden_cross'] = False
        signals['death_cross'] = False
        signals['pullback'] = False
        signals['entry_signal'] = False
        signals['stop_loss'] = False
        signals['exit_signal'] = False
        
        # 识别金叉（MACD线在零轴上方穿越信号线）
        signals['golden_cross'] = (
            (signals['macd'] > 0) & 
            (signals['signal'] > 0) & 
            (signals['macd'] > signals['signal']) & 
            (signals['macd'].shift(1) <= signals['signal'].shift(1))
        )
        
        # 识别死叉（MACD线穿越信号线向下）
        signals['death_cross'] = (
            (signals['macd'] < signals['signal']) & 
            (signals['macd'].shift(1) >= signals['signal'].shift(1))
        )
        
        # 寻找金叉后的回踩但不死叉的情况
        
        # 跟踪金叉后的状态
        post_golden_cross = False
        pullback_active = False
        entry_points = []
        
        for i in range(1, len(signals)):
            current_idx = signals.index[i]
            
            # 检查是否是金叉
            if signals.loc[current_idx, 'golden_cross']:
                post_golden_cross = True
                pullback_active = False
                
            # 检查是否是死叉
            if signals.loc[current_idx, 'death_cross']:
                post_golden_cross = False
                pullback_active = False
                
            # 如果在金叉之后，寻找回踩
            if post_golden_cross:
                prev_idx = signals.index[i-1]
                
                # 回踩开始于MACD线开始接近信号线
                if (not pullback_active and 
                    signals.loc[current_idx, 'macd'] < signals.loc[prev_idx, 'macd'] and
                    (signals.loc[current_idx, 'macd'] - signals.loc[current_idx, 'signal']) < 
                    (signals.loc[prev_idx, 'macd'] - signals.loc[prev_idx, 'signal'])):
                    pullback_active = True
                    
                # 如果在回踩中，MACD线开始再次远离信号线
                # 但没有穿越到信号线下方，这是我们的入场信号
                if (pullback_active and 
                    signals.loc[current_idx, 'macd'] > signals.loc[prev_idx, 'macd'] and
                    (signals.loc[current_idx, 'macd'] - signals.loc[current_idx, 'signal']) > 
                    (signals.loc[prev_idx, 'macd'] - signals.loc[prev_idx, 'signal']) and
                    signals.loc[current_idx, 'macd'] > signals.loc[current_idx, 'signal'] and
                    signals.loc[current_idx, 'macd'] > 0 and
                    signals.loc[current_idx, 'signal'] > 0):
                    
                    signals.loc[current_idx, 'pullback'] = True
                    signals.loc[current_idx, 'entry_signal'] = True
                    entry_points.append(i)
                    pullback_active = False
        
        # 为每个入场点计算止损
        # 设置止损为回踩的最低点
        for entry in entry_points:
            entry_idx = signals.index[entry]
            
            # 寻找之前的金叉
            golden_cross_mask = signals['golden_cross'].iloc[:entry]
            if golden_cross_mask.any():
                last_golden_cross_idx = golden_cross_mask.iloc[::-1].idxmax()
                last_golden_cross_loc = signals.index.get_loc(last_golden_cross_idx)
                
                # 基于回踩期间的最低点计算止损
                pullback_low = signals['low'].iloc[last_golden_cross_loc:entry].min()
                # 为止损价格添加1%的缓冲
                stop_loss_price = pullback_low * 0.99
                
                # 标记止损价格
                signals.loc[entry_idx, 'stop_loss_price'] = stop_loss_price
        
        return signals
    
    def backtest(self, initial_capital=10000, risk_per_trade=0.03):
        """
        Backtest the strategy with historical data.
        
        Parameters:
        -----------
        initial_capital : float
            Initial capital for the backtest
        risk_per_trade : float
            Percentage of capital to risk per trade
            
        Returns:
        --------
        pd.DataFrame, dict
            Dataframe with backtest results and performance metrics
        """
        # 计算指标
        df = self.calculate_indicators()
        
        # 识别信号
        signals = self.identify_signals(df)
        
        # 初始化投资组合列
        signals['position'] = 0.0  # 改为浮点型
        signals['entry_price'] = np.nan
        signals['stop_loss_price'] = np.nan
        signals['target_price_1'] = np.nan
        signals['target_price_2'] = np.nan
        signals['exit_price'] = np.nan
        signals['returns'] = 0.0
        signals['equity'] = initial_capital
        
        # 跟踪活跃仓位和表现
        in_position = False
        entry_price = 0
        stop_loss_price = 0
        target_price_1 = 0
        target_price_2 = 0
        trade_count = 0
        winning_trades = 0
        total_return = 0
        
        # 目标利润水平
        target_1_pct = 0.10  # 10% 第一目标
        target_2_pct = 0.20  # 20% 第二目标
        
        # 存储交易详情的列表
        trades = []
        
        # 模拟交易
        for i in range(1, len(signals)):
            current_idx = signals.index[i]
            current_price = signals['close'].iloc[i]
            
            # 退出逻辑 - 检查是否在仓位中
            if in_position:
                # 检查是否触及止损
                if signals['low'].iloc[i] <= stop_loss_price:
                    # 在止损价退出仓位
                    exit_price = stop_loss_price  # 假设我们在止损价退出
                    signals.loc[current_idx, 'position'] = 0
                    signals.loc[current_idx, 'exit_price'] = exit_price
                    signals.loc[current_idx, 'exit_signal'] = True
                    
                    # 计算收益
                    trade_return = (exit_price / entry_price) - 1
                    total_return += trade_return
                    signals.loc[current_idx, 'returns'] = trade_return
                    
                    # 记录交易详情
                    trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': current_idx,
                        'exit_price': exit_price,
                        'return': trade_return,
                        'exit_type': 'Stop Loss'
                    })
                    
                    # 重置仓位跟踪
                    in_position = False
                    
                # 检查是否达到第一个目标
                elif signals['high'].iloc[i] >= target_price_1 and signals['high'].iloc[i] < target_price_2:
                    # 在第一个目标部分退出（50%仓位）
                    exit_price = target_price_1
                    signals.loc[current_idx, 'position'] = 0.5
                    signals.loc[current_idx, 'exit_price'] = exit_price
                    
                    # 更新止损到保本点
                    stop_loss_price = entry_price
                    
                    # 计算部分收益
                    trade_return = ((exit_price / entry_price) - 1) * 0.5
                    total_return += trade_return
                    signals.loc[current_idx, 'returns'] = trade_return
                    
                    # 记录部分交易
                    trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': current_idx,
                        'exit_price': exit_price,
                        'return': (exit_price / entry_price) - 1,
                        'exit_type': 'Target 1 (Partial)'
                    })
                
                # 检查是否达到第二个目标
                elif signals['high'].iloc[i] >= target_price_2:
                    # 在第二个目标退出剩余仓位
                    exit_price = target_price_2
                    signals.loc[current_idx, 'position'] = 0
                    signals.loc[current_idx, 'exit_price'] = exit_price
                    signals.loc[current_idx, 'exit_signal'] = True
                    
                    # 计算剩余仓位收益
                    position_size = 0.5 if signals['position'].iloc[i-1] == 0.5 else 1.0
                    trade_return = ((exit_price / entry_price) - 1) * position_size
                    total_return += trade_return
                    signals.loc[current_idx, 'returns'] = trade_return
                    
                    # 记录交易详情
                    trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': current_idx,
                        'exit_price': exit_price,
                        'return': (exit_price / entry_price) - 1,
                        'exit_type': 'Target 2'
                    })
                    
                    # 计为盈利交易
                    winning_trades += 1
                    
                    # 重置仓位跟踪
                    in_position = False
                
                # 检查死叉退出信号
                elif (signals['macd'].iloc[i] <= signals['signal'].iloc[i] and 
                      signals['macd'].iloc[i-1] > signals['signal'].iloc[i-1]) or \
                     (signals['macd'].iloc[i] <= 0):
                    
                    # 因死叉或MACD低于零而退出
                    exit_price = signals['close'].iloc[i]
                    signals.loc[current_idx, 'position'] = 0
                    signals.loc[current_idx, 'exit_price'] = exit_price
                    signals.loc[current_idx, 'exit_signal'] = True
                    
                    # 计算仓位收益
                    position_size = 0.5 if signals['position'].iloc[i-1] == 0.5 else 1.0
                    trade_return = ((exit_price / entry_price) - 1) * position_size
                    total_return += trade_return
                    signals.loc[current_idx, 'returns'] = trade_return
                    
                    # 记录交易详情
                    trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': current_idx,
                        'exit_price': exit_price,
                        'return': (exit_price / entry_price) - 1,
                        'exit_type': 'MACD Signal'
                    })
                    
                    # 如果盈利则计为盈利交易
                    if trade_return > 0:
                        winning_trades += 1
                        
                    # 重置仓位跟踪
                    in_position = False
            
            # 入场逻辑 - 如果不在仓位中且有入场信号
            elif signals['entry_signal'].iloc[i]:
                # 计算仓位参数
                entry_price = signals['close'].iloc[i]
                entry_date = current_idx
                
                # 计算止损（从信号识别或回退到百分比）
                if np.isnan(signals['stop_loss_price'].iloc[i]):
                    stop_loss_price = entry_price * 0.95  # 默认5%止损
                else:
                    stop_loss_price = signals['stop_loss_price'].iloc[i]
                
                # 计算目标价格
                target_price_1 = entry_price * (1 + target_1_pct)
                target_price_2 = entry_price * (1 + target_2_pct)
                
                # 记录入场详情
                signals.loc[current_idx, 'position'] = 1
                signals.loc[current_idx, 'entry_price'] = entry_price
                signals.loc[current_idx, 'stop_loss_price'] = stop_loss_price
                signals.loc[current_idx, 'target_price_1'] = target_price_1
                signals.loc[current_idx, 'target_price_2'] = target_price_2
                
                # 更新跟踪
                in_position = True
                trade_count += 1
        
        # ... 其余代码保持不变 ...

        # 更新资金曲线
        signals['equity'] = initial_capital * (1 + signals['returns'].cumsum())
        
        # 计算回撤
        signals['peak'] = signals['equity'].cummax()
        signals['drawdown'] = (signals['equity'] - signals['peak']) / signals['peak']
        
        # 创建交易DataFrame
        self.trades = pd.DataFrame(trades)
        
        # 计算性能指标
        self.performance = self.calculate_performance(signals, trade_count, winning_trades, initial_capital)
        
        # 保存信号数据
        self.signals = signals
        
        return signals, self.performance
    
    def calculate_performance(self, signals, trade_count, winning_trades, initial_capital):
        """计算回测性能指标"""
        final_equity = signals['equity'].iloc[-1]
        total_return = (final_equity / initial_capital) - 1
        
        # 计算年化收益率 - 修复日期计算
        try:
            # 尝试计算日期差
            days = (signals.index[-1] - signals.index[0]).days
        except:
            # 如果索引不是日期类型，则使用交易日数量
            days = len(signals)
        
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # 计算最大回撤
        max_drawdown = signals['drawdown'].min()
        
        # 计算夏普比率 (假设无风险利率为0)
        daily_returns = signals['equity'].pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if len(daily_returns) > 0 else 0
        
        # 计算胜率
        win_rate = winning_trades / trade_count if trade_count > 0 else 0
        
        # 如果有交易记录，计算更多指标
        if hasattr(self, 'trades') and not self.trades.empty:
            avg_trade = self.trades['return'].mean()
            best_trade = self.trades['return'].max()
            worst_trade = self.trades['return'].min()
            
            # 计算盈亏比
            winning_trades_df = self.trades[self.trades['return'] > 0]
            losing_trades_df = self.trades[self.trades['return'] < 0]
            
            avg_win = winning_trades_df['return'].mean() if not winning_trades_df.empty else 0
            avg_loss = abs(losing_trades_df['return'].mean()) if not losing_trades_df.empty else 0
            profit_factor = avg_win / avg_loss if avg_loss != 0 else float('inf')
            
            return {
                'Initial Capital': initial_capital,
                'Final Equity': final_equity,
                'Total Return': total_return,
                'Annual Return': annual_return,
                'Max Drawdown': max_drawdown,
                'Sharpe Ratio': sharpe_ratio,
                'Total Trades': trade_count,
                'Win Rate': win_rate,
                'Avg Trade Return': avg_trade,
                'Best Trade': best_trade,
                'Worst Trade': worst_trade,
                'Profit Factor': profit_factor
            }
        else:
            return {
                'Initial Capital': initial_capital,
                'Final Equity': final_equity,
                'Total Return': total_return,
                'Annual Return': annual_return,
                'Max Drawdown': max_drawdown,
                'Sharpe Ratio': sharpe_ratio,
                'Total Trades': trade_count,
                'Win Rate': win_rate
            }
    
    def plot_results(self, save_path=None):
        """绘制回测结果图表"""
        if not hasattr(self, 'signals'):
            raise ValueError("Run backtest() first")
            
        signals = self.signals
        
        # 创建图表
        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
        
        # 价格和信号图
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(signals.index, signals['close'], label='BTC Price')
        
        # 标记入场点和出场点
        entries = signals[signals['entry_signal']]
        exits = signals[signals['exit_signal']]
        
        # 绘制入场点
        ax1.scatter(entries.index, entries['close'], marker='^', color='g', s=100, label='Entry Signal')
        
        # 绘制出场点
        ax1.scatter(exits.index, exits['close'], marker='v', color='r', s=100, label='Exit Signal')
        
        # 绘制止损线
        if 'stop_loss_price' in signals.columns:
            for idx, row in signals[signals['entry_signal']].iterrows():
                if not np.isnan(row['stop_loss_price']):
                    ax1.hlines(row['stop_loss_price'], idx, 
                              signals.index[signals.index.get_loc(idx) + 20 if signals.index.get_loc(idx) + 20 < len(signals) else -1], 
                              colors='r', linestyles='dashed', alpha=0.7)
        
        ax1.set_ylabel('Price (USD)')
        ax1.set_title('BTC MACD Zero-Line Cross Pullback Strategy Backtest')
        ax1.legend()
        ax1.grid(True)
        
        # MACD图
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.plot(signals.index, signals['macd'], label='MACD')
        ax2.plot(signals.index, signals['signal'], label='Signal Line')
        ax2.bar(signals.index, signals['histogram'], color='grey', label='Histogram', alpha=0.4)
        ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # 高亮金叉
        golden_crosses = signals[signals['golden_cross']]
        ax2.scatter(golden_crosses.index, golden_crosses['macd'], marker='o', color='gold', s=80, label='Golden Cross')
        
        # 高亮回踩入场信号
        pullback_entries = signals[signals['pullback']]
        ax2.scatter(pullback_entries.index, pullback_entries['macd'], marker='*', color='green', s=120, label='Pullback Entry')
        
        ax2.set_ylabel('MACD Values')
        ax2.legend()
        ax2.grid(True)
        
        # RSI图
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.plot(signals.index, signals['rsi'], label='RSI', color='purple')
        ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        ax3.axhline(y=50, color='k', linestyle='--', alpha=0.3)
        ax3.set_ylabel('RSI')
        ax3.grid(True)
        ax3.legend()
        
        # 资金曲线图
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        ax4.plot(signals.index, signals['equity'], label='Portfolio Value', color='blue')
        ax4.set_ylabel('Equity (USD)')
        ax4.set_xlabel('Date')
        ax4.grid(True)
        ax4.legend()
        
        # 格式化x轴日期
        date_format = mdates.DateFormatter('%Y-%m-%d')
        ax4.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
        
        # 额外图表
        self.plot_drawdown()
        self.plot_monthly_returns()
        
    def plot_drawdown(self):
        """绘制回撤图表"""
        if not hasattr(self, 'signals'):
            raise ValueError("Run backtest() first")
            
        plt.figure(figsize=(14, 6))
        plt.plot(self.signals['drawdown'], color='red', alpha=0.7)
        plt.fill_between(self.signals.index, self.signals['drawdown'], 0, color='red', alpha=0.3)
        plt.title('Portfolio Drawdown')
        plt.ylabel('Drawdown %')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def plot_monthly_returns(self):
        """绘制月度收益热图"""
        if not hasattr(self, 'signals'):
            raise ValueError("Run backtest() first")
            
        # 确保索引是日期时间类型
        if not isinstance(self.signals.index, pd.DatetimeIndex):
            print("警告: 索引不是日期时间类型，无法生成月度收益热图")
            return
            
        # 提取日收益率
        daily_returns = self.signals['equity'].pct_change().dropna()
        
        # 创建月度收益
        try:
            monthly_returns = daily_returns.groupby([
                daily_returns.index.year,
                daily_returns.index.month
            ]).apply(lambda x: (1 + x).prod() - 1)
            
            # 转换为DataFrame，年份作为行，月份作为列
            monthly_returns_matrix = []
            years = sorted(set(monthly_returns.index.get_level_values(0)))
            
            for year in years:
                year_data = [None] * 12
                for month in range(1, 13):
                    try:
                        idx = (year, month)
                        if idx in monthly_returns.index:
                            year_data[month-1] = monthly_returns[idx]
                    except:
                        continue
                monthly_returns_matrix.append(year_data)
            
            # 创建DataFrame
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_returns_df = pd.DataFrame(monthly_returns_matrix, index=years, columns=month_names)
            
            # 绘制热图
            plt.figure(figsize=(14, 8))
            ax = sns.heatmap(monthly_returns_df, annot=True, cmap='RdYlGn', fmt='.1%',
                             linewidths=1, center=0, vmin=-0.2, vmax=0.2, cbar_kws={'label': 'Monthly Return'})
            plt.title('Monthly Returns Heatmap')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"生成月度收益热图时出错: {e}")
            return

    def print_summary(self):
        """打印性能摘要"""
        if not hasattr(self, 'performance'):
            raise ValueError("Run backtest() first")
            
        print("\n" + "="*50)
        print("PERFORMANCE SUMMARY")
        print("="*50)
        
        print(f"Initial Capital: ${self.performance['Initial Capital']:,.2f}")
        print(f"Final Equity: ${self.performance['Final Equity']:,.2f}")
        print(f"Total Return: {self.performance['Total Return']:.2%}")
        print(f"Annual Return: {self.performance['Annual Return']:.2%}")
        print(f"Max Drawdown: {self.performance['Max Drawdown']:.2%}")
        print(f"Sharpe Ratio: {self.performance['Sharpe Ratio']:.2f}")
        
        print("\nTrade Statistics:")
        print(f"Total Trades: {self.performance['Total Trades']}")
        print(f"Win Rate: {self.performance['Win Rate']:.2%}")
        
        if 'Avg Trade Return' in self.performance:
            print(f"Average Trade Return: {self.performance['Avg Trade Return']:.2%}")
            print(f"Best Trade: {self.performance['Best Trade']:.2%}")
            print(f"Worst Trade: {self.performance['Worst Trade']:.2%}")
            print(f"Profit Factor: {self.performance['Profit Factor']:.2f}")
        
        print("="*50)
        
        # 按退出类型打印交易分布
        if hasattr(self, 'trades') and not self.trades.empty:
            exit_types = self.trades['exit_type'].value_counts()
            print("\nTrade Distribution by Exit Type:")
            for exit_type, count in exit_types.items():
                print(f"  {exit_type}: {count} trades")


# 添加主函数来运行回测
if __name__ == "__main__":
    # 创建策略实例
    csv_file = "/Users/tretra/BTC_4h_2017-2025.csv"  # 确保文件路径正确
    strategy = MACDZeroCrossPullbackStrategy(csv_file)
    
    # 运行回测
    print("开始回测...")
    signals, performance = strategy.backtest(initial_capital=10000)
    
    # 打印性能摘要
    strategy.print_summary()
    
    # 绘制结果
    print("绘制回测结果...")
    strategy.plot_results()
