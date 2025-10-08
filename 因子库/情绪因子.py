import pandas as pd
import os
from datetime import datetime

def add_factor(df: pd.DataFrame, param=None, **kwargs) -> pd.DataFrame:
    """
    计算并将新的情绪因子列添加到股票行情数据中，并返回包含计算因子的DataFrame及其聚合方式。

    工作流程：
    1. 根据股票代码从processed_data/stock-emotion-data目录加载对应的情绪数据
    2. 将情绪数据与原始行情数据按交易日期合并
    3. 将情绪因子添加到原始行情数据DataFrame中

    :param df: pd.DataFrame，包含单只股票的K线数据，必须包括市场数据（如收盘价等）。
    :param param: 因子计算所需的参数，格式和含义根据因子类型的不同而有所不同。
    :param kwargs: 其他关键字参数，包括：
        - col_name: 新计算的因子列名。
    :return:
        - pd.DataFrame: 包含新计算的情绪因子列，与输入的df具有相同的索引。

    注意事项：
    - 如果股票代码对应的情绪数据文件不存在，则跳过该股票
    - 情绪数据与行情数据按交易日期合并，以df为主
    - 没有情绪数据的日期对应的情绪因子值为空
    """

    # ======================== 参数处理 ===========================
    # 从kwargs中提取因子列的名称，这里使用'col_name'来标识因子列名称
    col_name = kwargs['col_name']
    
    # 获取第一行的股票代码
    stock_code = df['股票代码'].iloc[0]
    
    # ======================== 加载情绪数据 ===========================
    # 构建情绪数据文件路径
    emotion_file_path = f"D:\python大作业\processed_data\stock-emotion-data/{stock_code}.csv"
    
    # 检查文件是否存在
    if not os.path.exists(emotion_file_path):
        print(f"警告：股票 {stock_code} 的情绪数据文件不存在，跳过")
        df[col_name] = None
        return df[[col_name]]
    
    try:
        # 读取情绪数据
        emotion_df = pd.read_csv(emotion_file_path, skiprows=1, encoding='utf-8')
        
        # 确保日期列是datetime类型
        emotion_df['交易日期'] = pd.to_datetime(emotion_df['交易日期'])
        df['交易日期'] = pd.to_datetime(df['交易日期'])
        
        # 直接合并情绪得分列到原始数据
        # 以df为主，左连接
        merged_df = df.merge(
            emotion_df[['交易日期', '情绪得分']], 
            on='交易日期', 
            how='left'
        )
        
        # 直接添加情绪得分列到df
        df[col_name] = merged_df['情绪得分']
            
    except Exception as e:
        print(f"错误：加载股票 {stock_code} 的情绪数据时出错：{str(e)}")
        df[col_name] = None
        return df[[col_name]]

    return df[[col_name]]
