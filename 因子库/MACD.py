import pandas as pd
import numpy as np

def add_factor(df: pd.DataFrame, param=None, **kwargs) -> pd.DataFrame:
    """
    计算并将新的因子列添加到股票行情数据中，并返回包含计算因子的DataFrame及其聚合方式。

    工作流程：
    1. 根据提供的参数计算股票的MACD因子值。
    2. 将因子值添加到原始行情数据DataFrame中。

    :param df: pd.DataFrame，包含单只股票的K线数据，必须包括市场数据（如收盘价等）。
    :param param: 因子计算所需的参数，格式为 [fast_period, slow_period, signal_period]
        - fast_period: 快速EMA周期，默认12
        - slow_period: 慢速EMA周期，默认26
        - signal_period: 信号线EMA周期，默认9
    :param kwargs: 其他关键字参数，包括：
        - col_name: 新计算的因子列名。
        - fin_data: 财务数据字典，格式为 {'财务数据': fin_df, '原始财务数据': raw_fin_df}，其中fin_df为处理后的财务数据，raw_fin_df为原始数据，后者可用于某些因子的自定义计算。
        - 其他参数：根据具体需求传入的其他因子参数。
    :return:
        - pd.DataFrame: 包含新计算的因子列，与输入的df具有相同的索引。

     注意事项：
     - MACD金叉（DIF上穿DEA）为持有信号，死叉（DIF下穿DEA）为不持有信号
     - 因子值为1表示持有信号，0表示不持有信号
    """
    # 从额外参数中获取因子名称
    col_name = kwargs['col_name']
    
    # 设置默认参数
    if param is None:
        fast_period, slow_period, signal_period = 12, 26, 9
    else:
        fast_period = param[0] if len(param) > 0 else 12
        slow_period = param[1] if len(param) > 1 else 26
        signal_period = param[2] if len(param) > 2 else 9

    # 计算快速EMA和慢速EMA
    df['ema_fast'] = df['收盘价_复权'].ewm(span=fast_period, min_periods=fast_period).mean()
    df['ema_slow'] = df['收盘价_复权'].ewm(span=slow_period, min_periods=slow_period).mean()
    
    # 计算DIF（快线）
    df['dif'] = df['ema_fast'] - df['ema_slow']
    
    # 计算DEA（信号线）
    df['dea'] = df['dif'].ewm(span=signal_period, min_periods=signal_period).mean()
    
    # 计算MACD柱状图
    df['macd'] = (df['dif'] - df['dea']) * 2
    
    # 生成择时信号
    # 1: 持有信号（DIF上穿DEA，即金叉）
    # 0: 不持有信号（DIF下穿DEA，即死叉）
    
    # 计算DIF和DEA的交叉情况
    df['dif_prev'] = df['dif'].shift(1)
    df['dea_prev'] = df['dea'].shift(1)
    
    # 金叉：当前DIF > DEA 且 前一日DIF <= 前一日DEA
    golden_cross = (df['dif'] > df['dea']) & (df['dif_prev'] <= df['dea_prev'])
    
    # 死叉：当前DIF < DEA 且 前一日DIF >= 前一日DEA
    death_cross = (df['dif'] < df['dea']) & (df['dif_prev'] >= df['dea_prev'])
    
    # 生成信号
    conditions = [
        golden_cross,  # 金叉，持有信号
        death_cross    # 死叉，不持有信号
    ]
    choices = [1, 0]  # 持有信号=1，不持有信号=0
    df[col_name] = np.select(conditions, choices, default=1)  # 默认持有信号=1
    
    # 清理临时列
    df.drop(['ema_fast', 'ema_slow', 'dif', 'dea', 'macd', 'dif_prev', 'dea_prev'], axis=1, inplace=True)
    
    # 返回只包含因子列的DataFrame
    return df[[col_name]]
