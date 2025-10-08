import pandas as pd
import numpy as np

def add_factor(df: pd.DataFrame, param=None, **kwargs) -> pd.DataFrame:
    """
    计算并将新的因子列添加到股票行情数据中，并返回包含计算因子的DataFrame及其聚合方式。

    工作流程：
    1. 根据提供的参数计算股票的VWMA因子值。
    2. 将因子值添加到原始行情数据DataFrame中。

    :param df: pd.DataFrame，包含单只股票的K线数据，必须包括市场数据（如收盘价、成交量等）。
    :param param: 因子计算所需的参数，格式为 [period, price_threshold]
        - period: VWMA计算周期，默认20
        - price_threshold: 价格比较阈值，默认0.02（2%）
    :param kwargs: 其他关键字参数，包括：
        - col_name: 新计算的因子列名。
    :return:
        - pd.DataFrame: 包含新计算的因子列，与输入的df具有相同的索引。

    注意事项：
    - VWMA = Σ(价格 × 成交量) / Σ(成交量)
    - 当收盘价高于VWMA时为持有信号，低于VWMA时为不持有信号
    - 因子值为1表示持有信号，0表示不持有信号
    """
    # 从额外参数中获取因子名称
    col_name = kwargs['col_name']
    
    # 设置默认参数
    if param is None:
        period, price_threshold = 20, 0.02
    else:
        period = param[0] if len(param) > 0 else 20
        price_threshold = param[1] if len(param) > 1 else 0.02

    # 计算VWMA（成交量加权移动平均线）
    # VWMA = Σ(价格 × 成交量) / Σ(成交量)
    df['price_volume'] = df['收盘价_复权'] * df['成交量']
    df['vwma'] = df['price_volume'].rolling(window=period, min_periods=period).sum() / df['成交量'].rolling(window=period, min_periods=period).sum()
    
    # 生成择时信号
    # 1: 持有信号（收盘价高于VWMA）
    # 0: 不持有信号（收盘价低于VWMA）
    
    # 计算价格相对于VWMA的偏离度
    df['price_deviation'] = (df['收盘价_复权'] - df['vwma']) / df['vwma']
    
    # 生成信号：价格高于VWMA且偏离度超过阈值时为持有信号
    conditions = [
        df['price_deviation'] > price_threshold  # 价格显著高于VWMA，持有信号
    ]
    choices = [1]  # 持有信号=1
    df[col_name] = np.select(conditions, choices, default=0)  # 默认不持有信号=0
    
    # 清理临时列
    df.drop(['price_volume', 'vwma', 'price_deviation'], axis=1, inplace=True)
    
    # 返回只包含因子列的DataFrame
    return df[[col_name]]
