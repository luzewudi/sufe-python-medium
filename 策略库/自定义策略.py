"""
邢不行｜策略分享会
选股策略框架𝓟𝓻𝓸

版权所有 ©️ 邢不行
微信: xbx1717

本代码仅供个人学习使用，未经授权不得复制、修改或用于商业用途。

Author: 邢不行
"""
import pandas as pd


def filter_stock(df, strategy) -> pd.DataFrame:
    """
    过滤函数，在选股前过滤
    :param df: 整理好的数据，包含因子信息，并做过周期转换
    :param strategy: 策略配置
    :return: 返回过滤后的数据
    """
    # 删除月末为st状态的周期数
    cond1 = ~df['股票名称'].str.contains('ST', regex=False)
    # 删除月末为s状态的周期数
    cond2 = ~df['股票名称'].str.contains('S', regex=False)
    # 删除月末有退市风险的周期数
    cond3 = ~df['股票名称'].str.contains('*', regex=False)
    cond4 = ~df['股票名称'].str.contains('退', regex=False)
    # 删除交易天数过少的周期数
    # cond5 = df['交易天数'] / df['市场交易天数'] >= 0.8

    cond6 = df['下日_是否交易'] == 1
    cond7 = df['下日_开盘涨停'] != 1
    cond8 = df['下日_是否ST'] != 1
    cond9 = df['下日_是否退市'] != 1
    # cond10 = df['上市至今交易天数'] > days_listed
    
    # 检查沪深300成分股列是否存在
    if '沪深300成分股' in df.columns:
        cond10 = df['沪深300成分股'] == 'Y'
    else:
        # 如果列不存在，默认所有股票都符合条件
        cond10 = pd.Series([True] * len(df), index=df.index)

    common_filter = cond1 & cond2 & cond3 & cond4 & cond6 & cond7 & cond8 & cond9 & cond10
    df = df[common_filter]
    
    # 基于情绪因子进行筛选：选择每个交易日情绪得分最高的100只股票

    df = df.sort_values(['交易日期', '情绪因子'], ascending=[True, False])
    df = df.groupby('交易日期').head(100)

    
    return df


def calc_timing_factor(df, strategy) -> pd.DataFrame:
    """
    计算择时信号
    :param df: 整理好的数据，包含因子信息，并做过周期转换
    :param strategy: 策略配置
    :return: 返回包含择时信号的数据

    ### df 列说明
    包含基础列：  ['交易日期', '股票代码', '股票名称', '上市至今交易天数', '复权因子', '开盘价', '最高价',
                '最低价', '收盘价', '成交额', '是否交易', '流通市值', '总市值', '下日_开盘涨停', '下日_是否ST', '下日_是否交易',
                '下日_是否退市']
    以及config中配置好的，因子计算的结果列。

    ### strategy 数据说明
    - strategy.name: 策略名称
    - strategy.timing_list: 择时因子列表
    - strategy.filter_list: 过滤因子列表
    - strategy.factor_columns: 择时+过滤因子的列名
    """
    # 直接从df中获取3个择时因子的信号
    timing_signals = df[['MACD', 'RSI', 'VWMA']]
    
    # 计算每个交易日择时信号为1的数量
    signal_count = timing_signals.sum(axis=1)
    
    # 择时逻辑：有任意2个择时信号为1时，最终择时信号为1，否则为0
    df['择时信号'] = (signal_count >= 2).astype(int)
    
    return df
