"""
邢不行™️选股框架
Python股票量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""
import pandas as pd
import numpy as np

def add_factor(df: pd.DataFrame, param=None, **kwargs) -> pd.DataFrame:
    """
    计算并将新的因子列添加到股票行情数据中，并返回包含计算因子的DataFrame及其聚合方式。

    工作流程：
    1. 根据提供的参数计算股票的因子值。
    2. 将因子值添加到原始行情数据DataFrame中。

    :param df: pd.DataFrame，包含单只股票的K线数据，必须包括市场数据（如收盘价等）。
    :param param: 因子计算所需的参数，格式和含义根据因子类型的不同而有所不同。
    :param kwargs: 其他关键字参数，包括：
        - col_name: 新计算的因子列名。
    :return:
        - pd.DataFrame: 包含新计算的因子列，与输入的df具有相同的索引。

    """
    # 从额外参数中获取因子名称
    col_name = kwargs['col_name']
    n, m = param[0], param[1]  # n和m分别为两条均线的周期

    # 计算n日均线和m日均线
    df[f'ma_{n}'] = df['收盘价_复权'].rolling(n, min_periods=n).mean()
    df[f'ma_{m}'] = df['收盘价_复权'].rolling(m, min_periods=m).mean()

    # 当n日均线大于m日均线时，因子值为1，否则为0
    # 对于均线数据不足的情况（NaN），结果也会是NaN
    df[col_name] = np.where(df[f'ma_{n}'] > df[f'ma_{m}'], 1, 0)

    # 返回只包含因子列的DataFrame
    return df[[col_name]]
