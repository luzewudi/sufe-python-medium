import pandas as pd
import numpy as np

def add_factor(df: pd.DataFrame, param=None, **kwargs) -> pd.DataFrame:
    """
    计算并将新的因子列添加到股票行情数据中，并返回包含计算因子的DataFrame及其聚合方式。

    工作流程：
    1. 根据提供的参数计算股票的布林带因子值。
    2. 将因子值添加到原始行情数据DataFrame中。

    :param df: pd.DataFrame，包含单只股票的K线数据，必须包括市场数据（如收盘价等）。
    :param param: 因子计算所需的参数，格式为 [period, std_multiplier]
        - period: 布林带计算周期，默认20
        - std_multiplier: 标准差倍数，默认2
    :param kwargs: 其他关键字参数，包括：
        - col_name: 新计算的因子列名。
    :return:
        - pd.DataFrame: 包含新计算的因子列，与输入的df具有相同的索引。

     注意事项：
     - 价格触及下轨为持有信号，触及上轨为不持有信号
     - 因子值为1表示持有信号，0表示不持有信号
    """
    # 从额外参数中获取因子名称
    col_name = kwargs['col_name']
    
    # 设置默认参数
    if param is None:
        period, std_multiplier = 20, 2
    else:
        period = param[0] if len(param) > 0 else 20
        std_multiplier = param[1] if len(param) > 1 else 2

    # 计算中轨（移动平均线）
    df['middle_band'] = df['收盘价_复权'].rolling(window=period, min_periods=period).mean()
    
    # 计算标准差
    df['std'] = df['收盘价_复权'].rolling(window=period, min_periods=period).std()
    
    # 计算上轨和下轨
    df['upper_band'] = df['middle_band'] + (df['std'] * std_multiplier)
    df['lower_band'] = df['middle_band'] - (df['std'] * std_multiplier)
    
    # 生成择时信号
    # 1: 持有信号（价格触及或跌破下轨）
    # 0: 不持有信号（价格触及或突破上轨）
    
    conditions = [
        df['收盘价_复权'] <= df['lower_band'],  # 触及下轨，持有信号
        df['收盘价_复权'] >= df['upper_band']   # 触及上轨，不持有信号
    ]
    choices = [1, 0]  # 持有信号=1，不持有信号=0
    df[col_name] = np.select(conditions, choices, default=1)  # 默认持有信号=1
    
    # 清理临时列
    df.drop(['middle_band', 'std', 'upper_band', 'lower_band'], axis=1, inplace=True)
    
    # 返回只包含因子列的DataFrame
    return df[[col_name]]
