# -*- coding: utf-8 -*-
"""
择时因子分析工具
对择时因子（只有0和1）进行因子分析，计算未来1日、5日、20日的平均回报
包括IC、IR、分箱图等常用指标分析

择时因子分组说明：
- 因子值为0：第1组（卖出信号）
- 因子值为1：第2组（买入信号）
- 不进行分箱处理，直接使用因子值作为分组

"""

import datetime
import os
import gc
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import tools.utils.pfunctions as pf
import tools.utils.tfunctions as tf

# region =====需要配置的内容=====
# 择时因子的名称，按照运行缓存中的因子名输入
timing_factor_name = "MACD"  # 修改为你的择时因子名称

# 输入其他需要的基础因子，可用于后续例如复合因子的计算
other_factor_list = [
    # 'factor_成交额缩量因子_(10,60)', 'factor_市值_None'
]

# 数据中心路径配置
data_center_path = Path("D:\python大作业\processed_data")  # 修改为你的数据路径
runtime_data_path = Path("D:/python大作业/data")  # 修改为你的运行时数据路径

# 分析配置
bins = 2  # 择时因子只有2组：0和1
limit = 100  # 每周期最少需要多少个股票
period_offset = "1_0"  # 分析周期，择时因子通常用1日
fee_rate = 0.9988  # 手续费率

# 未来收益计算周期
future_periods = [1, 5, 20]  # 计算未来1日、5日、20日的收益

# 行业名称映射
ind_name_change = {
    "采掘": "煤炭",
    "化工": "基础化工", 
    "电气设备": "电力设备",
    "休闲服务": "社会服务",
    "纺织服装": "纺织服饰",
    "商业贸易": "商贸零售",
}


class TimingFactorAnalysisConfig:
    """择时因子分析配置类"""
    
    def __init__(self, factor_name: str, data_process_func, other_factor_list: List[str] = None):
        # 基础配置
        self.factor_name = factor_name
        self.fa_name = factor_name if factor_name.startswith("factor_") else f"factor_{factor_name}"
        self.func = data_process_func
        self.other_factor_list = other_factor_list or []
        
        # 路径配置
        self.data_center_path = data_center_path
        self.stock_data_path = self.data_center_path / "stock-trading-data-pro"
        self.index_data_path = self.data_center_path / "stock-main-index-data"
        self.fin_data_path = self.data_center_path / "stock-fin-data"
        
        # 分析配置
        self.bins = bins
        self.limit = limit
        self.period_offset = period_offset
        self.fee_rate = fee_rate
        self.ind_name_change = ind_name_change
        self.future_periods = future_periods
        
        # 财务数据列
        self.fin_cols = []
        
        # 需要保留的列
        self.keep_cols = [
            "交易日期",
            "股票代码", 
            "股票名称",
            "下日_是否交易",
            "下日_开盘涨停",
            "下日_是否ST",
            "下日_是否退市",
            "上市至今交易天数",
            self.fa_name,
            "新版申万一级行业名称",
        ]
        
        # 确保路径存在
        self._ensure_paths()
    
    def _ensure_paths(self):
        """确保必要的路径存在"""
        if not self.data_center_path.exists():
            raise FileNotFoundError(f"数据中心路径不存在: {self.data_center_path}")
        if not self.stock_data_path.exists():
            raise FileNotFoundError(f"股票数据路径不存在: {self.stock_data_path}")
        if not self.fin_data_path.exists():
            raise FileNotFoundError(f"财务数据路径不存在: {self.fin_data_path}")
        if not self.index_data_path.exists():
            raise FileNotFoundError(f"指数数据路径不存在: {self.index_data_path}")
    
    def get_runtime_folder(self) -> Path:
        """获取运行时缓存文件夹"""
        return runtime_data_path / "运行缓存" / "测试"
    
    def get_result_folder(self) -> Path:
        """获取结果文件夹"""
        return runtime_data_path / "分析结果" / "择时因子分析"
    
    def get_analysis_folder(self) -> Path:
        """获取分析结果文件夹"""
        return runtime_data_path / "分析结果"


def data_process(df):
    """
    数据处理函数，主要是过滤、计算复合因子等
    :param df: 输入数据
    :return: 处理后的数据
    """
    # 择时因子通常不需要额外的数据处理
    # 如果需要过滤条件，可以在这里添加
    
    # 示例：过滤条件
    # df = df[df['收盘价'] > 5]  # 过滤低价股
    # df = df[df['总市值'] > 50]  # 过滤小市值股票
    
    return df


def load_timing_factor_data(cfg, factor_list, boost=True):
    """
    加载择时因子数据，参考get_data方法
    :param cfg: 配置对象
    :param factor_list: 因子列表
    :param boost: 是否加速
    :return: 合并后的数据
    """
    print("📊 开始加载择时因子数据...")
    
    # 获取未来涨跌幅数据
    print("  🔄 获取未来涨跌幅数据...")
    rs_df = tf.get_ret_and_style(cfg, boost)
    
    # 读取all_factors_kline.pkl数据
    print("  📁 读取all_factors_kline.pkl...")
    kline_data_path = cfg.get_runtime_folder() / 'all_factors_kline.pkl'
    if not kline_data_path.exists():
        raise FileNotFoundError(f"K线数据文件不存在: {kline_data_path}")
    
    factor_df = pd.read_pickle(kline_data_path)
    print(f"  ✅ K线数据加载完成，形状: {factor_df.shape}")
    
    # 读取择时因子数据
    for factor_name in factor_list:
        print(f"  🔍 读取因子: {factor_name}")
        factor_path = cfg.get_runtime_folder() / f'{factor_name}.pkl'
        if not factor_path.exists():
            raise FileNotFoundError(f"因子文件不存在: {factor_path}")
        
        factor = pd.read_pickle(factor_path)
        if factor.empty:
            raise ValueError(f"{factor_name} 因子数据为空，请检查数据")
        if len(factor_df) != len(factor):
            raise ValueError(f"{factor_name} 因子长度不匹配，需要重新回测，更新数据")
        
        factor_df[factor_name] = factor
        print(f"  ✅ {factor_name} 因子加载完成，形状: {factor.shape}")
    
    # 优化rs_df，只保留合并需要的列，减少内存使用
    print("  🔧 优化rs_df数据...")
    rs_keep_cols = ['交易日期', '股票代码', '下周期涨跌幅', '下周期每天涨跌幅']
    # 添加所有风格因子列
    style_cols = [col for col in rs_df.columns if col.startswith('风格因子_')]
    rs_keep_cols.extend(style_cols)
    rs_df = rs_df[rs_keep_cols].copy()
    
    # 分次合并数据，避免内存爆炸
    print("  🔗 分次合并数据...")
    batch_size = 50000  # 每批处理5万行数据
    total_rows = len(rs_df)
    
    if total_rows <= batch_size:
        # 数据量小，直接合并
        factor_df = pd.merge(factor_df, rs_df, on=['交易日期', '股票代码'], how='right')
    else:
        # 数据量大，分批合并
        print(f"  📊 数据量较大({total_rows:,}行)，采用分批合并策略...")
        merged_chunks = []
        
        for i in range(0, total_rows, batch_size):
            end_idx = min(i + batch_size, total_rows)
            batch_rs_df = rs_df.iloc[i:end_idx].copy()
            
            print(f"  🔄 处理批次 {i//batch_size + 1}/{(total_rows-1)//batch_size + 1} (行 {i:,}-{end_idx-1:,})")
            
            # 合并当前批次
            batch_factor_df = pd.merge(factor_df, batch_rs_df, on=['交易日期', '股票代码'], how='right')
            merged_chunks.append(batch_factor_df)
            
            # 清理内存
            del batch_rs_df, batch_factor_df
            gc.collect()
        
        # 合并所有批次
        print("  🔗 合并所有批次...")
        factor_df = pd.concat(merged_chunks, ignore_index=True)
        del merged_chunks
        gc.collect()
    
    # 数据清洗
    print("  🧹 数据清洗...")
    factor_df = tf.data_preprocess(factor_df, cfg)
    if factor_df.empty:
        return pd.DataFrame()
    
    # 删除不需要的列
    print("  🗑️ 删除不需要的列...")
    drop_cols = ['上市至今交易天数', '复权因子', '开盘价', '最高价', '最低价', '收盘价', '成交额', '是否交易',
                 '下日_开盘涨停', '下日_是否ST', '下日_是否交易', '下日_是否退市']
    factor_df.drop(columns=drop_cols, inplace=True)
    
    del rs_df, drop_cols
    gc.collect()
    
    print(f"  ✅ 数据加载完成，最终形状: {factor_df.shape}")
    return factor_df


def calculate_future_returns(df, future_periods):
    """
    计算未来收益
    :param df: 数据
    :param future_periods: 未来周期列表
    :return: 添加了未来收益列的数据
    """
    print("📈 计算未来收益...")
    
    df = df.copy()
    
    for period in future_periods:
        print(f"  🔄 计算未来{period}日收益...")
        
        # 计算未来N日总收益率（复利）
        df[f'未来{period}日总收益率'] = df.groupby('股票代码')['下周期涨跌幅'].apply(
            lambda x: (1 + x).rolling(window=period, min_periods=1).apply(
                lambda y: y.prod() - 1, raw=False
            ).shift(-period+1)
        ).reset_index(0, drop=True)
        
        # 计算未来N日平均收益率
        df[f'未来{period}日平均收益率'] = df.groupby('股票代码')['下周期涨跌幅'].rolling(
            window=period, min_periods=1
        ).mean().reset_index(0, drop=True).shift(-period+1)
    
    print("  ✅ 未来收益计算完成")
    return df


def save_stock_data_by_code(df, factor_name, analysis_folder):
    """
    按股票代码将数据保存为单独的CSV文件
    :param df: 包含所有股票数据的DataFrame
    :param factor_name: 因子名称
    :param analysis_folder: 分析结果文件夹路径
    """
    print("💾 按股票代码保存数据...")
    
    # 创建因子专用文件夹
    factor_folder = analysis_folder / "择时因子分析" / factor_name
    factor_folder.mkdir(parents=True, exist_ok=True)
    
    # 获取所有股票代码
    stock_codes = df['股票代码'].unique()
    print(f"  📊 共发现 {len(stock_codes)} 只股票")
    
    saved_count = 0
    for stock_code in stock_codes:
        try:
            # 筛选单只股票的数据
            stock_data = df[df['股票代码'] == stock_code].copy()
            
            # 按交易日期排序
            stock_data = stock_data.sort_values('交易日期')
            
            # 保存为CSV文件
            csv_file = factor_folder / f"{stock_code}.csv"
            stock_data.to_csv(csv_file, index=False, encoding='utf-8-sig')
            
            saved_count += 1
            if saved_count % 100 == 0:
                print(f"  ✅ 已保存 {saved_count}/{len(stock_codes)} 只股票")
                
        except Exception as e:
            print(f"  ❌ 保存股票 {stock_code} 时出错: {e}")
            continue
    
    print(f"  ✅ 数据保存完成，共保存 {saved_count} 只股票的数据")
    print(f"  📁 保存路径: {factor_folder}")
    return factor_folder


def calculate_timing_ic(df, factor_name, future_periods):
    """
    计算择时因子的IC
    :param df: 数据
    :param factor_name: 因子名称
    :param future_periods: 未来周期列表
    :return: IC结果字典
    """
    print("📊 计算择时因子IC...")
    
    ic_results = {}
    
    for period in future_periods:
        print(f"  🔄 计算未来{period}日IC...")
        
        # 计算IC序列
        ic_data = df.groupby('交易日期').apply(
            lambda x: x[factor_name].corr(x[f'未来{period}日总收益率'], method='spearman')
        ).reset_index()
        ic_data.columns = ['交易日期', 'RankIC']
        
        # 计算IC统计信息
        ic_mean = ic_data['RankIC'].mean()
        ic_std = ic_data['RankIC'].std()
        ic_ir = ic_mean / ic_std if ic_std != 0 else 0
        ic_win_rate = (ic_data['RankIC'] > 0).mean()
        
        # 计算累计IC
        ic_data['累计RankIC'] = ic_data['RankIC'].cumsum()
        
        ic_results[f'未来{period}日'] = {
            'ic_data': ic_data,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ic_ir': ic_ir,
            'ic_win_rate': ic_win_rate
        }
        
        print(f"    ✅ 未来{period}日IC: 均值={ic_mean:.4f}, IR={ic_ir:.4f}, 胜率={ic_win_rate:.4f}")
    
    return ic_results


def calculate_timing_performance(df, factor_name, future_periods):
    """
    计算择时因子表现
    :param df: 数据
    :param factor_name: 因子名称
    :param future_periods: 未来周期列表
    :return: 表现结果字典
    """
    print("📈 计算择时因子表现...")
    
    performance_results = {}
    
    for period in future_periods:
        print(f"  🔄 计算未来{period}日表现...")
        
        # 按因子值分组
        factor_0 = df[df[factor_name] == 0]
        factor_1 = df[df[factor_name] == 1]
        
        # 计算各组的平均收益（使用总收益率）
        ret_0 = factor_0[f'未来{period}日总收益率'].mean()
        ret_1 = factor_1[f'未来{period}日总收益率'].mean()
        
        # 计算多空收益
        long_short_ret = ret_1 - ret_0
        
        # 计算胜率
        win_rate_0 = (factor_0[f'未来{period}日总收益率'] > 0).mean()
        win_rate_1 = (factor_1[f'未来{period}日总收益率'] > 0).mean()
        
        # 计算夏普比率
        sharpe_0 = factor_0[f'未来{period}日总收益率'].mean() / factor_0[f'未来{period}日总收益率'].std() if factor_0[f'未来{period}日总收益率'].std() != 0 else 0
        sharpe_1 = factor_1[f'未来{period}日总收益率'].mean() / factor_1[f'未来{period}日总收益率'].std() if factor_1[f'未来{period}日总收益率'].std() != 0 else 0
        
        performance_results[f'未来{period}日'] = {
            'factor_0_ret': ret_0,
            'factor_1_ret': ret_1,
            'long_short_ret': long_short_ret,
            'win_rate_0': win_rate_0,
            'win_rate_1': win_rate_1,
            'sharpe_0': sharpe_0,
            'sharpe_1': sharpe_1
        }
        
        print(f"    ✅ 未来{period}日总收益: 因子0={ret_0:.4f}, 因子1={ret_1:.4f}, 多空={long_short_ret:.4f}")
        
        # 如果是20日，额外显示平均收益
        if period == 20:
            factor_0_avg = df[df[factor_name] == 0][f'未来20日平均收益率'].mean()
            factor_1_avg = df[df[factor_name] == 1][f'未来20日平均收益率'].mean()
            long_short_avg = factor_1_avg - factor_0_avg
            print(f"    ✅ 未来20日平均收益: 因子0={factor_0_avg:.4f}, 因子1={factor_1_avg:.4f}, 多空={long_short_avg:.4f}")
    
    return performance_results


def create_timing_analysis_plots(df, factor_name, ic_results, performance_results, future_periods):
    """
    创建择时因子分析图表
    :param df: 数据
    :param factor_name: 因子名称
    :param ic_results: IC结果
    :param performance_results: 表现结果
    :param future_periods: 未来周期列表
    :return: 图表列表
    """
    print("📊 创建择时因子分析图表...")
    
    fig_list = []
    
    # 1. IC曲线图
    for period in future_periods:
        period_name = f'未来{period}日'
        ic_data = ic_results[period_name]['ic_data']
        
        fig = pf.draw_ic_plotly(
            x=ic_data["交易日期"], 
            y1=ic_data["RankIC"], 
            y2=ic_data["累计RankIC"], 
            title=f"择时因子{period_name}RankIC图",
            info=f"IC均值: {ic_results[period_name]['ic_mean']:.4f}, IR: {ic_results[period_name]['ic_ir']:.4f}"
        )
        fig_list.append(fig)
    
    # 2. 因子分布图
    factor_dist = df[factor_name].value_counts().sort_index()
    fig = pf.draw_bar_plotly(
        x=factor_dist.index.astype(str), 
        y=factor_dist.values, 
        title="择时因子分布图"
    )
    fig_list.append(fig)
    
    # 3. 分组总收益对比图
    for period in future_periods:
        period_name = f'未来{period}日'
        perf = performance_results[period_name]
        
        # 创建分组总收益对比数据
        group_data = pd.DataFrame({
            '分组': ['因子=0', '因子=1', '多空'],
            '总收益率': [perf['factor_0_ret'], perf['factor_1_ret'], perf['long_short_ret']]
        })
        
        fig = pf.draw_bar_plotly(
            x=group_data['分组'], 
            y=group_data['总收益率'], 
            title=f"择时因子{period_name}分组总收益对比"
        )
        fig_list.append(fig)
    
    # 4. 20日平均收益对比图（仅20日）
    if 20 in future_periods:
        period_name = '未来20日'
        perf = performance_results[period_name]
        
        # 计算20日平均收益
        factor_0_avg = df[df[factor_name] == 0][f'未来20日平均收益率'].mean()
        factor_1_avg = df[df[factor_name] == 1][f'未来20日平均收益率'].mean()
        long_short_avg = factor_1_avg - factor_0_avg
        
        avg_data = pd.DataFrame({
            '分组': ['因子=0', '因子=1', '多空'],
            '平均收益率': [factor_0_avg, factor_1_avg, long_short_avg]
        })
        
        fig = pf.draw_bar_plotly(
            x=avg_data['分组'], 
            y=avg_data['平均收益率'], 
            title="择时因子未来20日平均收益对比"
        )
        fig_list.append(fig)
    
    # 5. 胜率对比图
    for period in future_periods:
        period_name = f'未来{period}日'
        perf = performance_results[period_name]
        
        win_rate_data = pd.DataFrame({
            '分组': ['因子=0', '因子=1'],
            '胜率': [perf['win_rate_0'], perf['win_rate_1']]
        })
        
        fig = pf.draw_bar_plotly(
            x=win_rate_data['分组'], 
            y=win_rate_data['胜率'], 
            title=f"择时因子{period_name}胜率对比",
            y_range=[0, 1]
        )
        fig_list.append(fig)
    
    # 6. 夏普比率对比图
    for period in future_periods:
        period_name = f'未来{period}日'
        perf = performance_results[period_name]
        
        sharpe_data = pd.DataFrame({
            '分组': ['因子=0', '因子=1'],
            '夏普比率': [perf['sharpe_0'], perf['sharpe_1']]
        })
        
        fig = pf.draw_bar_plotly(
            x=sharpe_data['分组'], 
            y=sharpe_data['夏普比率'], 
            title=f"择时因子{period_name}夏普比率对比"
        )
        fig_list.append(fig)
    
    print(f"  ✅ 创建了 {len(fig_list)} 个图表")
    return fig_list


def timing_factor_analysis(name, func, cfg, _other_factor_list, boost):
    """
    择时因子分析主函数
    """
    start_time = datetime.datetime.now()
    print(f"🚀 开始择时因子分析: {name}")
    
    # 构建因子列表
    factor_list = []
    if cfg.fa_name not in factor_list:
        factor_list.append(cfg.fa_name)
    if _other_factor_list is not None:
        for _other_factor in _other_factor_list:
            _other_factor = _other_factor if _other_factor.startswith("factor_") else f"factor_{_other_factor}"
            if _other_factor not in factor_list:
                factor_list.append(_other_factor)
    
    # 加载数据
    factor_df = load_timing_factor_data(cfg, factor_list, boost)
    if factor_df.empty:
        print("❌ 数据为空，分析终止")
        return
    
    # 应用数据处理函数
    factor_df = func(factor_df)
    
    # 计算未来收益
    factor_df = calculate_future_returns(factor_df, cfg.future_periods)
    
    # 按股票代码保存数据
    factor_folder = save_stock_data_by_code(factor_df, cfg.fa_name, cfg.get_analysis_folder())
    
    # 计算IC
    ic_results = calculate_timing_ic(factor_df, cfg.fa_name, cfg.future_periods)
    
    # 计算表现
    performance_results = calculate_timing_performance(factor_df, cfg.fa_name, cfg.future_periods)
    
    # 创建图表
    fig_list = create_timing_analysis_plots(factor_df, cfg.fa_name, ic_results, performance_results, cfg.future_periods)
    
    # 生成分析报告
    start_date = factor_df["交易日期"].min().strftime("%Y/%m/%d")
    end_date = factor_df["交易日期"].max().strftime("%Y/%m/%d")
    
    # 计算综合得分
    total_score = 0
    for period in cfg.future_periods:
        period_name = f'未来{period}日'
        ic_ir = ic_results[period_name]['ic_ir']
        long_short_ret = performance_results[period_name]['long_short_ret']
        total_score += ic_ir * 0.5 + long_short_ret * 100
    
    title = f"{cfg.fa_name} 择时因子分析报告\n分析区间：{start_date} - {end_date}\n分析周期：{cfg.period_offset}\n综合得分：{total_score:.2f}"
    
    # 保存结果
    save_path = tf.get_folder_path(cfg.get_analysis_folder(), "择时因子分析")
    pf.merge_html(save_path, fig_list=fig_list, strategy_file=f"{cfg.fa_name}择时因子分析报告", bbs_id="31614", title=title)
    
    print(f"✅ 择时因子分析完成，耗时：{datetime.datetime.now() - start_time}")
    print(f"📊 分析结果已保存到: {save_path}")
    print(f"📁 股票数据已按代码保存到: {factor_folder}")


if __name__ == "__main__":
    print("🚀 开始运行择时因子分析程序...")
    
    # 创建配置对象
    conf = TimingFactorAnalysisConfig(timing_factor_name, data_process, other_factor_list)
    
    # 确保必要的文件夹存在
    conf.get_runtime_folder().mkdir(parents=True, exist_ok=True)
    conf.get_result_folder().mkdir(parents=True, exist_ok=True)
    conf.get_analysis_folder().mkdir(parents=True, exist_ok=True)
    
    # 运行择时因子分析
    timing_factor_analysis(timing_factor_name, data_process, conf, other_factor_list, boost=True)
