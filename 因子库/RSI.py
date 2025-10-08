import pandas as pd
import numpy as np

def add_factor(df: pd.DataFrame, param=None, **kwargs) -> pd.DataFrame:
    """
    计算并将新的因子列添加到股票行情数据中，并返回包含计算因子的DataFrame及其聚合方式。

    工作流程：
    1. 根据提供的参数计算股票的RSI因子值。
    2. 将因子值添加到原始行情数据DataFrame中。

    :param df: pd.DataFrame，包含单只股票的K线数据，必须包括市场数据（如收盘价等）。
    :param param: 因子计算所需的参数，格式为 [period, upper_threshold, lower_threshold]
        - period: RSI计算周期，默认14
        - upper_threshold: RSI超买阈值，默认70
        - lower_threshold: RSI超卖阈值，默认30
    :param kwargs: 其他关键字参数，包括：
        - col_name: 新计算的因子列名。
    :return:
        - pd.DataFrame: 包含新计算的因子列，与输入的df具有相同的索引。

     注意事项：
     - RSI值在0-100之间，超过70通常被认为是超买，低于30被认为是超卖
     - 因子值为1表示持有信号（RSI < 超卖阈值），0表示不持有信号（RSI > 超买阈值）
    """
    # 从额外参数中获取因子名称
    col_name = kwargs['col_name']
    
    # 设置默认参数
    if param is None:
        period, upper_threshold, lower_threshold = 14, 70, 30
    else:
        period = param[0] if len(param) > 0 else 14
        upper_threshold = param[1] if len(param) > 1 else 70
        lower_threshold = param[2] if len(param) > 2 else 30

    # 计算价格变化
    df['price_change'] = df['收盘价_复权'].diff()
    
    # 分离上涨和下跌
    df['gain'] = df['price_change'].where(df['price_change'] > 0, 0)
    df['loss'] = -df['price_change'].where(df['price_change'] < 0, 0)
    
    # 计算平均上涨和下跌
    df['avg_gain'] = df['gain'].rolling(window=period, min_periods=period).mean()
    df['avg_loss'] = df['loss'].rolling(window=period, min_periods=period).mean()
    
    # 计算RSI
    rs = df['avg_gain'] / df['avg_loss']
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 生成择时信号
    # 1: 持有信号（RSI < 超卖阈值，买入机会）
    # 0: 不持有信号（RSI > 超买阈值，卖出信号）
    conditions = [
        df['rsi'] < lower_threshold,  # 超卖，持有信号
        df['rsi'] > upper_threshold   # 超买，不持有信号
    ]
    choices = [1, 0]  # 持有信号=1，不持有信号=0
    df[col_name] = np.select(conditions, choices, default=0)  # 默认持有信号=0
    
    # 清理临时列
    df.drop(['price_change', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rsi'], axis=1, inplace=True)
    
    # 返回只包含因子列的DataFrame
    return df[[col_name]]
