import gc

import pandas as pd
import numpy as np
import hashlib
import os
from pathlib import Path
from tqdm import tqdm
import datetime
from concurrent.futures import ProcessPoolExecutor
from core.fin_essentials import merge_with_finance_data
from core.utils.path_kit import get_folder_path
import math


# region 通用函数
def get_data_path_md5(data_path):
    # 将文件夹的大小、更改时间信息作为参数，生成md5值
    info_txt = ''
    if data_path.is_dir():
        files = list(sorted([str(f) for f in data_path.iterdir() if f.is_file()]))
        for file in files:
            info_txt += f'{os.path.getsize(file)}-{os.path.getmtime(file)}'
    if data_path.is_file():
        info_txt += f'{os.path.getsize(data_path)}-{os.path.getmtime(data_path)}'
    md5_txt = hashlib.md5(info_txt.encode('utf-8')).hexdigest()
    return md5_txt


def read_txt(path):
    with open(path, 'r') as f:
        txt = f.read()
    return txt


def write_txt(path, txt):
    with open(path, 'w') as f:
        f.write(txt)


def filter_stock(df):
    """
    过滤函数，ST/退市/交易天数不足等情况
    :param df:
    :return:
    """
    # =删除不能交易的周期数
    # 删除月末为st状态的周期数
    df = df[df['股票名称'].str.contains('ST') == False]
    # 删除月末为s状态的周期数
    df = df[df['股票名称'].str.contains('S') == False]
    # 删除月末有退市风险的周期数
    df = df[df['股票名称'].str.contains('\*') == False]
    df = df[df['股票名称'].str.contains('退') == False]

    df = df[df['下日_是否交易'] == 1]
    df = df[df['下日_开盘涨停'] == False]
    df = df[df['下日_是否ST'] == False]
    df = df[df['下日_是否退市'] == False]
    df = df[df['上市至今交易天数'] > 250]

    return df


def float_num_process(num, return_type=float, keep=2, max=5):
    """
    针对绝对值小于1的数字进行特殊处理，保留非0的N位（N默认为2，即keep参数）
    输入  0.231  输出  0.23
    输入  0.0231  输出  0.023
    输入  0.00231  输出  0.0023
    如果前面max个都是0，直接返回0.0
    :param num: 输入的数据
    :param return_type: 返回的数据类型，默认是float
    :param keep: 需要保留的非零位数
    :param max: 最长保留多少位
    :return:
        返回一个float或str
    """

    # 如果输入的数据是0，直接返回0.0
    if num == 0.:
        return 0.0

    # 绝对值大于1的数直接保留对应的位数输出
    if abs(num) > 1:
        return round(num, keep)
    # 获取小数点后面有多少个0
    zero_count = -int(math.log10(abs(num)))
    # 实际需要保留的位数
    keep = min(zero_count + keep, max)

    # 如果指定return_type是float，则返回float类型的数据
    if return_type == float:
        return round(num, keep)
    # 如果指定return_type是str，则返回str类型的数据
    else:
        return str(round(num, keep))


# endregion

# region 单因子分析要用到的函数（不分双因子的也在）

def process_stock(stock_folder, per_df, cfg, stock):
    stock_path = stock_folder / stock
    df = pd.read_csv(stock_path, encoding='gbk', parse_dates=['交易日期'], skiprows=1)
    
    # 特殊处理 1_0 情况
    if cfg.period_offset == '1_0':
        # 对于 1_0，不需要合并 period_offset_df，直接使用原始数据
        if df.empty:
            return pd.DataFrame()
    else:
        # 正常情况，合并 period_offset 数据
        max_date = df['交易日期'].max()
        min_date = df['交易日期'].min()
        df = pd.merge(df, per_df[per_df['交易日期'].between(min_date, max_date)], on='交易日期', how='right')
        if df.empty:
            return pd.DataFrame()
    for col in ['股票代码', '交易日期', '收盘价', '总市值', '新版申万一级行业名称']:
        df[col] = df[col].ffill()

    df['涨跌幅'] = df['收盘价'] / df['前收盘价'] - 1
    df['涨跌幅'] = df['涨跌幅'].fillna(value=0)
    df['开盘价'] = df['开盘价'].fillna(value=df['收盘价'])
    # 计算复权因子
    df['复权因子'] = (df['涨跌幅'] + 1).cumprod()
    df['收盘价_复权'] = df['复权因子'] * (df.iloc[0]['收盘价'] / df['复权因子'].iloc[0])
    df['开盘价_复权'] = df['开盘价'] / df['收盘价'] * df['收盘价_复权']

    # 计算风格因子
    fin_cols = ['R_np@xbx_ttm', 'B_total_equity_atoopc@xbx', 'R_revenue@xbx_ttm', 'R_np@xbx_ttm同比',
                'R_revenue@xbx_ttm同比', 'R_np@xbx_单季同比', 'R_revenue@xbx_单季同比', 'B_total_liab@xbx',
                'B_actual_received_capital@xbx', 'B_preferred_shares@xbx', 'B_total_assets@xbx',
                'B_total_liab_and_owner_equity@xbx', 'R_op@xbx_ttm']
    df = merge_with_finance_data(stock[:-4], df, fin_cols)[0]

    name = '风格因子_'
    # ===估值因子
    df[name + 'EP'] = df['R_np@xbx_ttm'] / df['总市值']  # 市盈率倒数
    df[name + 'BP'] = df['B_total_equity_atoopc@xbx'] / df['总市值']  # 市净率倒数
    df[name + 'SP'] = df['R_revenue@xbx_ttm'] / df['总市值']  # 市销率倒数

    # ===动量因子
    df[name + 'Ret_252'] = df['收盘价_复权'].shift(21) / df['收盘价_复权'].shift(252) - 1

    # ===反转因子
    df[name + 'Ret_21'] = df['收盘价_复权'] / df['收盘价_复权'].shift(21) - 1

    # ===成长因子
    df[name + '净利润ttm同比'] = df['R_np@xbx_ttm同比']
    df[name + '营业收入ttm同比'] = df['R_revenue@xbx_ttm同比']
    df[name + '净利润单季同比'] = df['R_np@xbx_单季同比']
    df[name + '营业收入单季同比'] = df['R_revenue@xbx_单季同比']

    # ===杠杆因子
    df[name + 'MLEV'] = (df['总市值'] + df['B_total_liab@xbx']) / df['总市值']
    df[name + 'BLEV'] = (df[['B_actual_received_capital@xbx', 'B_preferred_shares@xbx']].sum(axis=1, skipna=True)) / df[
        '总市值']
    df[name + 'DTOA'] = df['B_total_liab@xbx'] / df['B_total_assets@xbx']

    # ===波动因子
    df[name + 'Std21'] = df['涨跌幅'].rolling(21).std()
    df[name + 'Std252'] = df['涨跌幅'].rolling(252).std()

    # ===盈利因子
    df[name + 'ROE'] = df['R_np@xbx_ttm'] / df['B_total_equity_atoopc@xbx']  # ROE 净资产收益率
    df[name + 'ROA'] = df['R_np@xbx_ttm'] / df['B_total_liab_and_owner_equity@xbx']  # ROA 资产收益率
    df[name + '净利润率'] = df['R_np@xbx_ttm'] / df['R_revenue@xbx_ttm']  # 净利润率：净利润 / 营业收入
    df[name + 'GP'] = df['R_op@xbx_ttm'] / df['B_total_assets@xbx']

    # ===规模因子
    df[name + '总市值'] = np.log(df['总市值'])

    # 做一些简单的周期转换
    if cfg.period_offset == '1_0':
        # 对于 1_0，不需要分组聚合，直接使用原始数据
        period_df = df.copy()
        
        # 计算 1_0 的特殊收益：当天开盘买，第二天开盘卖
        # 使用开盘价复权/前一日开盘价复权
        period_df['下周期涨跌幅'] = (period_df['开盘价_复权'] / period_df['开盘价_复权'].shift(1) - 1).shift(-1)
        # 对于 1_0，下周期每天涨跌幅就是当天的涨跌幅
        period_df['下周期每天涨跌幅'] = period_df['涨跌幅'].shift(-1)
    else:
        # 正常情况的分组聚合
        agg_dict = {'交易日期': 'last', '股票代码': 'last', '开盘价_复权': 'first',
                    '收盘价_复权': 'last', '新版申万一级行业名称': 'last'}
        style_cols = [col for col in df.columns if col.startswith(name)]
        for col in style_cols:
            agg_dict[col] = 'last'
        period_df = df.groupby(cfg.period_offset).agg(agg_dict)

        # 计算下周期的收益
        period_df['下周期涨跌幅'] = (period_df['收盘价_复权'] / period_df['开盘价_复权'] - 1).shift(-1)
        # 计算下周期每天的收益
        period_df['下周期每天涨跌幅'] = df.groupby(cfg.period_offset)['涨跌幅'].apply(lambda x: list(x)).shift(-1)

    period_df.dropna(subset=['下周期涨跌幅', '下周期每天涨跌幅'], how='any', inplace=True)
    return period_df


def get_data(cfg, _factor_list, boost):
    # 获取未来涨跌幅数据
    rs_df = get_ret_and_style(cfg)

    # 读取因子数据
    factor_df = pd.read_pickle(cfg.get_runtime_folder() / 'all_factors_kline.pkl')

    for factor_name in _factor_list:
        factor = pd.read_pickle(cfg.get_runtime_folder() / f'{factor_name}.pkl')
        if factor.empty:
            raise ValueError(f"{factor} 因子数据为空，请检查数据")
        if len(factor_df) != len(factor):
            raise ValueError(f"{factor} 因子长度不匹配，需要重新回测，更新数据")
        factor_df[factor_name] = factor

    # 合并数据，如果有重复列以factor_df为准
    factor_df = pd.merge(factor_df, rs_df, on=['交易日期', '股票代码'], how='right', suffixes=('', '_rs'))
    # 删除rs_df的重复列（以_rs结尾的列）
    rs_cols = [col for col in factor_df.columns if col.endswith('_rs')]
    factor_df = factor_df.drop(columns=rs_cols)
    # 数据清洗
    factor_df = data_preprocess(factor_df, cfg)
    if factor_df.empty:
        return pd.DataFrame()
    drop_cols = ['上市至今交易天数', '复权因子', '开盘价', '最高价', '最低价', '收盘价', '成交额', '是否交易',
                 '下日_开盘涨停', '下日_是否ST', '下日_是否交易', '下日_是否退市']
    factor_df.drop(columns=drop_cols, inplace=True)
    del rs_df, drop_cols
    gc.collect()
    return factor_df


def cal_style_factor(df, style_name, base_factors):
    print(f'开始计算【{style_name}】风格因子...')
    name = '风格因子_'
    factor_cols = []
    for factor in base_factors:
        if not factor.startswith(name):
            factor = name + factor
        df[factor] = df.groupby('交易日期')[factor].rank(ascending=True, method='min')
        factor_cols.append(factor)

    df[name + style_name] = df[factor_cols].sum(axis=1)
    df.drop(columns=factor_cols, inplace=True)
    return df


def get_ret_and_style(cfg, boost=False):
    cal_future_rate = False
    # 看一下md5文件是否存在
    md5_file = get_folder_path(cfg.get_result_folder().parent.parent, '运行缓存', 'md5信息') / 'future_rate_md5.txt'
    stock_folder = cfg.stock_data_path
    future_rate_path = get_folder_path(cfg.get_result_folder().parent.parent, '运行缓存') / '未来收益及风格因子.pkl'
    if (not md5_file.exists()) or not (future_rate_path.exists()):
        cal_future_rate = True
        new_md5_txt = get_data_path_md5(stock_folder)
    else:
        old_md5_txt = read_txt(md5_file)
        new_md5_txt = get_data_path_md5(stock_folder)
        if old_md5_txt != new_md5_txt:
            cal_future_rate = True
    if cal_future_rate:
        print('数据发生变更，需要重新计算未来收益 & 风格因子')
        start_time = datetime.datetime.now()  # 记录开始时间
        if 'M' in cfg.period_offset:
            raise ValueError('因子分析不支持M系列的offset，因为每月的交易日数不固定')
        if cfg.period_offset == 'W53_0':
            raise ValueError('因子分析不支持W53_0的offset')
        
        # 特殊处理 1_0 情况
        if cfg.period_offset == '1_0':
            print('检测到 1_0 周期，使用特殊处理逻辑（当天开盘买，第二天开盘卖）')
            # 对于 1_0，不需要读取 period_offset.csv，直接处理原始数据
            period_offset_df = None
        else:
            # 读取period_offset数据
            try:
                period_offset_df = pd.read_csv(Path(cfg.data_center_path) / 'period_offset.csv', encoding='gbk', skiprows=1,
                                               parse_dates=['交易日期'], usecols=['交易日期', cfg.period_offset])
            except KeyError:
                raise ValueError(f'period_offset.csv 文件中没有找到列 "{cfg.period_offset}"，请检查文件内容')
        dfs = []
        stock_list = [s for s in os.listdir(stock_folder) if ('.csv' in s) and ('bj' not in s)]
        if boost:
            with ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
                futures = []
                for code in stock_list:
                    futures.append(executor.submit(process_stock, stock_folder, period_offset_df, cfg, code))
                for future in tqdm(futures, desc='📦 处理数据', total=len(futures)):
                    df = future.result()
                    dfs.append(df)
        else:
            for stock in tqdm(stock_list):
                dfs.append(process_stock(stock_folder, period_offset_df, cfg, stock))

        df = pd.concat(dfs, ignore_index=True)
        del dfs
        gc.collect()

        # 对风格因子做截面处理
        # ===估值
        df = cal_style_factor(df, '估值', ['EP', 'BP', 'SP'])

        # ===动量
        df = cal_style_factor(df, '动量', ['Ret_252'])

        # ===反转
        df = cal_style_factor(df, '反转', ['Ret_21'])

        # ===成长
        df = cal_style_factor(df, '成长', ['净利润ttm同比', '营业收入ttm同比', '净利润单季同比', '营业收入单季同比'])

        # ===杠杆
        df = cal_style_factor(df, '杠杆', ['MLEV', 'BLEV', 'DTOA'])

        # ===波动
        df = cal_style_factor(df, '波动', ['Std21', 'Std252'])

        # ===盈利
        df = cal_style_factor(df, '盈利', ['ROE', 'ROA', '净利润率', 'GP'])

        # ===规模
        df = cal_style_factor(df, '规模', ['总市值'])

        df.to_pickle(future_rate_path)
        write_txt(md5_file, new_md5_txt)
        print(f'计算耗时：{datetime.datetime.now() - start_time}')
    else:
        print('数据无变更，无需重新计算未来收益')
        start_time = datetime.datetime.now()  # 记录开始时间
        write_txt(md5_file, new_md5_txt)
        df = pd.read_pickle(future_rate_path)
        print(f'读取耗时：{datetime.datetime.now() - start_time}')
    return df


def data_preprocess(df, cfg):
    # 删除数据不全的日期
    df.dropna(subset=['股票代码'], inplace=True)
    # 过滤掉无法交易的股票
    df = filter_stock(df)
    # 额外的计算函数
    df = cfg.func(df)
    # 删除掉字段为空的列
    # 删除必要字段为空的部分
    df = df.dropna(subset=cfg.keep_cols, how='any')
    # 将因子信息转换成float类型
    if hasattr(cfg, 'fa_name'):
        df[cfg.fa_name] = df[cfg.fa_name].astype(float)
    else:
        df[cfg.main] = df[cfg.main].astype(float)
        df[cfg.sub] = df[cfg.sub].astype(float)

    # =保留每个周期的股票数量大于limit的日期
    df['当周期股票数'] = df.groupby('交易日期')['交易日期'].transform('count')
    df = df[df['当周期股票数'] > cfg.limit].reset_index(drop=True)
    if df.empty:
        return df

    # 如果是单因子分析
    if hasattr(cfg, 'fa_name'):
        # 检查是否为择时因子（只有0和1两个值）
        unique_values = df[cfg.fa_name].dropna().unique()
        is_timing_factor = len(unique_values) <= 2 and all(val in [0, 1] for val in unique_values)
        
        if is_timing_factor:
            # 择时因子直接使用因子值作为分组，0为第1组，1为第2组
            print(f"  🎯 检测到择时因子 {cfg.fa_name}，直接使用因子值分组...")
            df['groups'] = df[cfg.fa_name].replace({0: 1, 1: 2})
        else:
            # 普通因子按照排名进行分箱
            print(f"  📊 普通因子 {cfg.fa_name}，按排名分箱...")
            df['因子_排名'] = df.groupby(['交易日期'])[cfg.fa_name].rank(ascending=True, method='first')
            df['groups'] = df.groupby(['交易日期'])['因子_排名'].transform(
                lambda x: pd.qcut(x, q=cfg.bins, labels=range(1, cfg.bins + 1), duplicates='drop'))
    # 如果是双因子分析
    else:
        df = double_factor_grouping(df, cfg)
    return df


def get_ic(df, cfg):
    print('正在进行因子IC分析...')
    start_time = datetime.datetime.now()  # 记录开始时间

    # 计算IC并处理数据
    ic = df.groupby('交易日期').apply(lambda x: x[cfg.fa_name].corr(x['下周期涨跌幅'], method='spearman')).to_frame()
    ic = ic.rename(columns={0: 'RankIC'}).reset_index()
    ic['累计RankIC'] = ic['RankIC'].cumsum()

    # ===计算IC的统计值，并进行约等
    # =IC均值
    ic_mean = float_num_process(ic['RankIC'].mean())
    # =IC标准差
    ic_std = float_num_process(ic['RankIC'].std())
    # =icir
    icir = float_num_process(ic_mean / ic_std)
    # =IC胜率
    # 如果累计IC为正，则计算IC为正的比例
    if ic['累计RankIC'].iloc[-1] > 0:
        ic_ratio = str(float_num_process((ic['RankIC'] > 0).sum() / len(ic) * 100)) + '%'
    # 如果累计IC为负，则计算IC为负的比例
    else:
        ic_ratio = str(float_num_process((ic['RankIC'] < 0).sum() / len(ic) * 100)) + '%'

    # 将上述指标合成一个字符串，加入到IC图中
    ic_info = f'IC均值：{ic_mean}，IC标准差：{ic_std}，icir：{icir}，IC胜率：{ic_ratio}'

    # 计算每月的IC热力图
    ic_month = ic.resample('ME', on='交易日期').agg({'RankIC': 'mean'})
    ic_month.reset_index(inplace=True)
    # 提取出年份和月份
    ic_month['年份'] = ic_month['交易日期'].dt.year.astype('str')
    ic_month['月份'] = ic_month['交易日期'].dt.month
    # 将年份月份设置为index，在将月份unstack为列
    ic_month = ic_month.set_index(['年份', '月份'])['RankIC']
    ic_month = ic_month.unstack('月份')
    ic_month.columns = ic_month.columns.astype(str)
    # 计算各月平均的IC
    ic_month.loc['各月平均', :] = ic_month.mean(axis=0)
    # 按年份大小排名
    ic_month = ic_month.sort_index(ascending=False)

    print(f'因子IC分析完成，耗时：{datetime.datetime.now() - start_time}')
    return ic, ic_info, ic_month


def get_group_net_value(df, cfg):
    print('正在进行因子分组分析...')
    start_time = datetime.datetime.now()  # 记录开始时间

    # 由于会对原始的数据进行修正，所以需要把数据copy一份
    df['持仓收益'] = df['下周期涨跌幅'] * cfg.fee_rate
    # 按照分组计算资金曲线
    groups = df.groupby(['groups'], observed=False)
    res_list = []
    time_df = pd.DataFrame(sorted(df['交易日期'].unique()), columns=['交易日期'])
    for t, g in groups:
        ret = pd.DataFrame(g.groupby('交易日期')['持仓收益'].mean()).reset_index()
        ret['净值'] = (ret['持仓收益'] + 1).cumprod()
        ret = pd.merge(ret, time_df, on='交易日期', how='right')
        ret['净值'] = ret['净值'].ffill()
        ret['groups'] = t[0]
        res_list.append(ret[['交易日期', '净值', 'groups']])
    res_df = pd.concat(res_list, ignore_index=True)
    res_df = pd.DataFrame(res_df.groupby(['交易日期', 'groups'])['净值'].mean()).reset_index()
    group_nv = res_df.pivot(values='净值', index='交易日期', columns='groups')
    group_nv.reset_index(inplace=True)

    for i in range(1, cfg.bins + 1):
        group_nv.rename(columns={i: f'第{i}组'}, inplace=True)

    # 计算多空净值走势
    # 获取第一组的涨跌幅数据
    first_group_ret = group_nv['第1组'].pct_change()
    first_group_ret.fillna(value=group_nv['第1组'].iloc[0] - 1, inplace=True)
    # 获取最后一组的涨跌幅数据
    last_group_ret = group_nv[f'第{cfg.bins}组'].pct_change()
    last_group_ret.fillna(value=group_nv[f'第{cfg.bins}组'].iloc[0] - 1, inplace=True)
    # 判断到底是多第一组空最后一组，还是多最后一组空第一组
    if group_nv['第1组'].iloc[-1] > group_nv[f'第{cfg.bins}组'].iloc[-1]:
        ls_ret = (first_group_ret - last_group_ret) / 2
    else:
        ls_ret = (last_group_ret - first_group_ret) / 2
    # 计算多空净值曲线
    group_nv['多空净值'] = (ls_ret + 1).cumprod()

    # 计算绘制分箱所需要的数据
    group_value = group_nv[-1:].T[1:].reset_index()
    group_value.columns = ['分组', '净值']

    # 计算持仓走势图
    df['周期数量'] = df['下周期每天涨跌幅'].apply(len)
    hold_nums = int(df['周期数量'].mode().iloc[0])
    df['下周期每天涨跌幅'] = df['下周期每天涨跌幅'].apply(
        lambda x: x[: hold_nums] if len(x) > hold_nums else (x + [0] * (hold_nums - len(x))))
    df['下周期每天净值'] = df['下周期每天涨跌幅'].apply(lambda x: (np.array(x) + 1).cumprod())
    df['下周期净值'] = df['下周期每天净值'].apply(lambda x: x[-1] * cfg.fee_rate)

    # 计算各分组在持仓内的每天收益
    group_hold_value = pd.DataFrame(df.groupby('groups', observed=False)['下周期每天净值'].mean()).T
    # 所有分组的第一天都是从1开始的
    for col in group_hold_value.columns:
        group_hold_value[col] = group_hold_value[col].apply(lambda x: [1] + list(x))
    # 将未来收益从list展开成逐行的数据
    group_hold_value = group_hold_value.explode(list(group_hold_value.columns)).reset_index(drop=True).reset_index()
    # 重命名列
    group_cols = ['时间'] + [f'第{i}组' for i in range(1, cfg.bins + 1)]
    group_hold_value.columns = group_cols

    print(f'因子分组分析完成，耗时：{datetime.datetime.now() - start_time}')

    # 返回数据：分组资金曲线、分组持仓走势
    return group_nv, group_value, group_hold_value


def get_style_corr(df, cfg):
    print('正在进行因子风格暴露分析...')
    start_date = datetime.datetime.now()  # 记录开始时间

    # 取出风格列，格式：以 风格因子_ 开头
    style_cols = [col for col in df.columns if '风格因子_' in col]

    # 如果df中没有风格因子列，返回空df
    if len(style_cols) == 0:
        return pd.DataFrame()

    # 计算因子与风格的相关系数
    res = df.groupby('交易日期').apply(
        lambda x: x[[cfg.fa_name] + style_cols].corr(method='spearman').iloc[0, 1:].to_frame())
    style_corr = res.reset_index().groupby('level_1')[cfg.fa_name].mean().reset_index()
    # 整理数据
    style_corr = style_corr.rename(columns={'level_1': '风格', cfg.fa_name: '相关系数'})
    style_corr['风格'] = style_corr['风格'].map(lambda x: x.split('_')[1])

    print(f'因子风格分析完成，耗时：{datetime.datetime.now() - start_date}')

    return style_corr


def get_class_ic_and_pct(df, cfg, is_industry=True):
    print('正在进行因子行业分析...' if is_industry else '正在进行因子市值分组分析...')
    start_date = datetime.datetime.now()  # 记录开始时间

    # 如果是行业分组
    if is_industry:
        class_col = '新版申万一级行业名称'
        class_name = '行业'
        import warnings
        warnings.filterwarnings('ignore')
        # 替换行业名称
        df['新版申万一级行业名称'] = df['新版申万一级行业名称'].replace(cfg.ind_name_change)
    else:  # 按照市值进行分组
        class_col = '市值分组'
        class_name = '市值分组'
        # 先对市值数据进行排名以及分组
        df['市值分组'] = df.groupby(['交易日期'])['风格因子_规模'].transform(
            lambda x: pd.qcut(x, q=cfg.bins, labels=range(1, cfg.bins + 1), duplicates='drop'))

    def get_data(temp):
        """
        计算分行业IC、占比
        :param temp: 每个行业的数据
        :return:
            返回IC序列的均值、第一组占比、最后一组占比
        """
        # 计算每个行业的IC序列
        ic = temp.groupby('交易日期').apply(lambda x: x[cfg.fa_name].corr(x['下周期涨跌幅'], method='spearman'))
        # 整理IC数据
        ic = ic.to_frame().reset_index().rename(columns={0: 'RankIC'})

        # 计算每个行业的第一组的占比和最后一组的占比
        part_min_group = temp.groupby('交易日期').apply(lambda x: (x['groups'] == min_group).sum())
        part_max_group = temp.groupby('交易日期').apply(lambda x: (x['groups'] == max_group).sum())
        part_min_group = part_min_group / all_min_group
        part_max_group = part_max_group / all_max_group
        # 整理占比数据
        part_min_group = part_min_group.to_frame().reset_index().rename(
            columns={0: f'因子第一组选股在各{class_name}的占比'})
        part_max_group = part_max_group.to_frame().reset_index().rename(
            columns={0: f'因子最后一组选股在各{class_name}的占比'})

        # 将各个数据合并一下
        data = pd.merge(ic, part_min_group, on='交易日期', how='inner')
        data = pd.merge(data, part_max_group, on='交易日期', how='inner')
        data.set_index('交易日期', inplace=True)  # 设置下索引

        return data

    # 获取以因子分组第一组和最后一组的数量
    min_group, max_group = df['groups'].min(), df['groups'].max()
    all_min_group = df.groupby('交易日期').apply(lambda x: (x['groups'] == min_group).sum())
    all_max_group = df.groupby('交易日期').apply(lambda x: (x['groups'] == max_group).sum())
    # 以行业分组计算IC及占比，并处理数据
    class_data = df.groupby(class_col, observed=False).apply(get_data).reset_index()

    # 对每个行业求IC均值、行业占比第一组均值、行业占比最后一组均值
    class_data = class_data.groupby(class_col, observed=False).apply(
        lambda x: [x['RankIC'].mean(), x[f'因子第一组选股在各{class_name}的占比'].mean(),
                   x[f'因子最后一组选股在各{class_name}的占比'].mean()])
    class_data = class_data.to_frame().reset_index()  # 整理数据
    # 取出IC数据、行业占比_第一组数据、行业占比_最后一组数据
    class_data['RankIC'] = class_data[0].map(lambda x: x[0])
    class_data[f'因子第一组选股在各{class_name}的占比'] = class_data[0].map(lambda x: x[1])
    class_data[f'因子最后一组选股在各{class_name}的占比'] = class_data[0].map(lambda x: x[2])
    # 处理数据
    class_data.drop(0, axis=1, inplace=True)
    # 以IC排序
    class_data.sort_values('RankIC', ascending=False, inplace=True)

    print(f'因子{class_col}分析完成，耗时：{datetime.datetime.now() - start_date}')
    return class_data


def get_factor_score(ic, group_value):
    max_net = max(group_value['净值'].iloc[0], group_value['净值'].iloc[-2])
    icir = ic['RankIC'].mean() / ic['RankIC'].std()
    rank_corr = np.corrcoef(list(group_value['净值'][:-1]), list(range(1, len(group_value))))[0, 1]
    score = max_net * abs(icir) * abs(rank_corr)
    return score


# endregion

# region 双因子分析要用到的函数
def double_factor_grouping(df, cfg):
    print(f'正在对双因子 {cfg.main} 和 {cfg.sub} 分组...')
    start_date = datetime.datetime.now()  # 记录开始时间

    # 根据主因子计算主因子的排名method='min'与风格因子保持相同取法
    df['排名_主因子'] = df.groupby(['交易日期'], observed=False)[cfg.main].rank(ascending=True, method='first')
    # 根据次因子计算次因子的排名
    df['排名_次因子'] = df.groupby(['交易日期'], observed=False)[cfg.sub].rank(ascending=True, method='first')
    # 根据主因子的排名进行分组
    df['groups_主因子'] = df.groupby(['交易日期'], observed=False)['排名_主因子'].transform(
        lambda x: pd.qcut(x, q=cfg.bins, labels=range(1, cfg.bins + 1), duplicates='drop'))
    # 根据次因子的排名进行分组
    df['groups_次因子'] = df.groupby(['交易日期'], observed=False)['排名_次因子'].transform(
        lambda x: pd.qcut(x, q=cfg.bins, labels=range(1, cfg.bins + 1), duplicates='drop'))
    # 在主因子分组基础上，再根据次因子的排名进行分组
    df['groups_主因子分箱_次因子'] = df.groupby(['交易日期', 'groups_主因子'], observed=False)['排名_次因子'].transform(
        lambda x: pd.qcut(x, q=cfg.bins, labels=range(1, cfg.bins + 1), duplicates='drop'))
    # 在次因子分组基础上，再根据主因子的排名进行分组
    df['groups_次因子分箱_主因子'] = df.groupby(['交易日期', 'groups_次因子'], observed=False)['排名_主因子'].transform(
        lambda x: pd.qcut(x, q=cfg.bins, labels=range(1, cfg.bins + 1), duplicates='drop'))

    # 这里不需要判断某个周期的股票数量大于bins，因为之前在处理limit时已经处理过这个问题
    print(f'双因子 {cfg.main} 和 {cfg.sub} 分组完成，耗时：{datetime.datetime.now() - start_date}')
    return df


def get_group_nv_double(df, cfg):
    """
    针对双因子分组数据进行分析，给出双因子分组的组合平均收益、过滤平均收益数据
    :param df: 输入的数据
    :param cfg: 配置
    :return:
        返回双因子组合分组平均收益、双因子组合分组平均占比、双因子过滤分组平均收益数据
    """

    print('计算双因子平均收益...')
    start_date = datetime.datetime.now()  # 记录开始时间

    # 由于会对原始的数据进行修正，所以需要把数据copy一份
    temp = df.copy()

    # 计算下周期每天的净值，并扣除手续费得到下周期的实际净值
    temp['下周期每天净值'] = temp['下周期每天涨跌幅'].apply(lambda x: (np.array(x) + 1).cumprod())
    temp['下周期平均收益'] = temp['下周期每天净值'].apply(lambda x: np.power((x[-1] * cfg.fee_rate), 1 / len(x)) - 1)

    # 计算双因子组合分组在持仓内的平均收益
    mix_nv = temp.groupby(['groups_主因子', 'groups_次因子'], observed=False)['下周期平均收益'].mean().reset_index()

    # 计算双因子组合分组在持仓内的股票占比
    mix_prop = temp.groupby(['交易日期', 'groups_主因子', 'groups_次因子'], observed=False).agg(
        {'股票名称': 'count', '当周期股票数': 'last'}).reset_index()
    mix_prop['当周期平均占比'] = mix_prop['股票名称'] / mix_prop['当周期股票数']
    mix_prop['当周期平均占比'] = mix_prop['当周期平均占比'].fillna(0)
    mix_prop = mix_prop.groupby(['groups_主因子', 'groups_次因子'], observed=False)[
        '当周期平均占比'].mean().reset_index()

    # 计算双因子过滤分组在持仓内的平均收益 主->次
    filter_nv_ms = temp.groupby(['groups_主因子', 'groups_主因子分箱_次因子'], observed=False)[
        '下周期平均收益'].mean().reset_index()

    # 计算双因子过滤分组在持仓内的平均收益 次->主
    filter_nv_sm = temp.groupby(['groups_次因子', 'groups_次因子分箱_主因子'], observed=False)[
        '下周期平均收益'].mean().reset_index()

    # 下周期平均收益转换单位千分之,当周期平均占比转换单位百分之
    mix_nv['下周期平均收益'] = mix_nv['下周期平均收益'].apply(lambda x: x * 1000)
    mix_prop['当周期平均占比'] = mix_prop['当周期平均占比'].apply(lambda x: x * 100)
    filter_nv_ms['下周期平均收益'] = filter_nv_ms['下周期平均收益'].apply(lambda x: x * 1000)
    filter_nv_sm['下周期平均收益'] = filter_nv_sm['下周期平均收益'].apply(lambda x: x * 1000)

    # 将groups_次因子、groups_主因子设置为index，在将groups_主因子为列
    mix_nv['groups_主因子'] = mix_nv['groups_主因子'].apply(lambda x: '主因子' + str(x))
    mix_nv['groups_次因子'] = mix_nv['groups_次因子'].apply(lambda x: '次因子' + str(x))
    mix_nv = mix_nv.set_index(['groups_次因子', 'groups_主因子'])['下周期平均收益']
    mix_nv = mix_nv.unstack('groups_主因子')
    # 添加平均收益
    mix_nv.loc['主因子平均收益'] = mix_nv.mean()
    mix_nv['次因子平均收益'] = mix_nv.mean(axis=1)

    mix_prop['groups_主因子'] = mix_prop['groups_主因子'].apply(lambda x: '主因子' + str(x))
    mix_prop['groups_次因子'] = mix_prop['groups_次因子'].apply(lambda x: '次因子' + str(x))
    mix_prop = mix_prop.set_index(['groups_次因子', 'groups_主因子'])['当周期平均占比']
    mix_prop = mix_prop.unstack('groups_主因子')

    # 计算双因子过滤组合主因子分箱平均收益，主因子分组的基础上，次因子再分组
    filter_nv_main_mean = filter_nv_ms.groupby(['groups_主因子'], observed=False).agg(
        {'groups_主因子分箱_次因子': 'first', '下周期平均收益': 'mean'}).reset_index()
    filter_nv_main_mean['groups_主因子分箱_次因子'] = 0
    filter_nv_ms = pd.concat([filter_nv_ms, filter_nv_main_mean], ignore_index=True)
    filter_nv_ms['groups_主因子'] = filter_nv_ms['groups_主因子'].astype(int)
    filter_nv_ms['groups_主因子分箱_次因子'] = filter_nv_ms['groups_主因子分箱_次因子'].astype(int)
    filter_nv_ms.sort_values(by=['groups_主因子', 'groups_主因子分箱_次因子'], inplace=True, ignore_index=True)

    filter_nv_ms = filter_nv_ms.set_index(['groups_主因子分箱_次因子', 'groups_主因子'])['下周期平均收益']
    filter_nv_ms = filter_nv_ms.unstack('groups_主因子')
    # 根据bins的数量来重命名
    rename_dict = {i: f'主因子{i}' for i in range(1, cfg.bins + 1)}
    filter_nv_ms.rename(columns=rename_dict, inplace=True)
    rename_dict = {i: f'次因子{i}' for i in range(1, cfg.bins + 1)}
    rename_dict[0] = '主因子平均收益'
    filter_nv_ms.rename(index=rename_dict, inplace=True)
    filter_nv_ms.loc['主因子平均收益'] = filter_nv_ms.mean()
    filter_nv_ms['次因子平均收益'] = filter_nv_ms.mean(axis=1)

    # 计算双因子过滤组合主因子分箱平均收益，次因子分组的基础上，主因子再分组
    filter_nv_sub_mean = filter_nv_sm.groupby(['groups_次因子'], observed=False).agg(
        {'groups_次因子分箱_主因子': 'first', '下周期平均收益': 'mean'}).reset_index()
    filter_nv_sub_mean['groups_次因子分箱_主因子'] = 0
    filter_nv_sm = pd.concat([filter_nv_sm, filter_nv_sub_mean], ignore_index=True)
    filter_nv_sm['groups_次因子'] = filter_nv_sm['groups_次因子'].astype(int)
    filter_nv_sm['groups_次因子分箱_主因子'] = filter_nv_sm['groups_次因子分箱_主因子'].astype(int)
    filter_nv_sm.sort_values(by=['groups_次因子', 'groups_次因子分箱_主因子'], inplace=True, ignore_index=True)
    filter_nv_sm = filter_nv_sm.set_index(['groups_次因子分箱_主因子', 'groups_次因子'])['下周期平均收益']
    filter_nv_sm = filter_nv_sm.unstack('groups_次因子')
    # 根据bins的数量来重命名
    rename_dict = {i: f'次因子{i}' for i in range(1, cfg.bins + 1)}
    filter_nv_sm.rename(columns=rename_dict, inplace=True)
    rename_dict = {i: f'主因子{i}' for i in range(1, cfg.bins + 1)}
    rename_dict[0] = '次因子平均收益'
    filter_nv_sm.rename(index=rename_dict, inplace=True)
    filter_nv_sm.loc['次因子平均收益'] = filter_nv_sm.mean()
    filter_nv_sm['主因子平均收益'] = filter_nv_sm.mean(axis=1)

    print(f'计算双因子平均收益完成，耗时：{datetime.datetime.now() - start_date}')
    return mix_nv, mix_prop, filter_nv_ms, filter_nv_sm


def get_style_corr_double(df, cfg):
    print('正在进行因子风格暴露分析...')
    start_date = datetime.datetime.now()  # 记录开始时间

    # 由于会对原始的数据进行修正，所以需要把数据copy一份
    temp = df.copy()

    temp['排名_主因子'] = temp.groupby(['交易日期'])[cfg.main].rank(ascending=True, method='first')
    # 根据次因子计算次因子的排名
    temp['排名_次因子'] = temp.groupby(['交易日期'])[cfg.sub].rank(ascending=True, method='first')

    # 计算因子IC值
    main_factor_ic = df.groupby('交易日期').apply(
        lambda x: x[cfg.main].corr(x['下周期涨跌幅'], method='spearman')).to_frame()
    main_factor_ic = main_factor_ic.rename(columns={0: 'RankIC'}).reset_index()
    main_factor_ic_mean = main_factor_ic['RankIC'].mean()
    sub_factor_ic = df.groupby('交易日期').apply(
        lambda x: x[cfg.sub].corr(x['下周期涨跌幅'], method='spearman')).to_frame()
    sub_factor_ic = sub_factor_ic.rename(columns={0: 'RankIC'}).reset_index()
    sub_factor_ic_mean = sub_factor_ic['RankIC'].mean()
    double_factor_ic_flag = 1 if main_factor_ic_mean * sub_factor_ic_mean >= 0 else -1

    # 计算双因子等权
    temp['风格因子_双因子'] = temp['排名_主因子'] + temp['排名_次因子'] * double_factor_ic_flag

    # 取出风格列，格式：以 风格因子_ 开头
    factor_style_cols = [col for col in temp.columns if '风格因子_' in col]

    def func(x, factor, style):
        if len(x) > 100:
            res = x[[factor] + style].corr(method='spearman').iloc[0, 1:].to_frame()
        else:
            res = pd.Series()
        return res

    temp.dropna(subset=['排名_次因子', '排名_次因子', '风格因子_双因子'] + factor_style_cols, inplace=True)
    main_res = temp.groupby('交易日期').apply(lambda x: func(x, '排名_主因子', factor_style_cols))
    main_factor_style_corr = main_res.reset_index().groupby('level_1')['排名_主因子'].mean().reset_index()

    sub_res = temp.groupby('交易日期').apply(lambda x: func(x, '排名_次因子', factor_style_cols))
    sub_factor_style_corr = sub_res.reset_index().groupby('level_1')['排名_次因子'].mean().reset_index()

    double_res = temp.groupby('交易日期').apply(lambda x: func(x, '风格因子_双因子', factor_style_cols))
    double_factor_style_corr = double_res.reset_index().groupby('level_1')['风格因子_双因子'].mean().reset_index()

    # 风格因子_双因子 这里是主次因子的相关系数
    max_inx = double_factor_style_corr.idxmax()
    double_factor_style_corr.loc[max_inx, '风格因子_双因子'] = \
        temp[[cfg.main, cfg.sub]].corr(method='spearman').iloc[0, 1]

    # 整理数据
    main_factor_style_corr = main_factor_style_corr.rename(
        columns={'level_1': '风格', '排名_主因子': '相关系数_主因子'})
    sub_factor_style_corr = sub_factor_style_corr.rename(columns={'level_1': '风格', '排名_次因子': '相关系数_次因子'})
    double_factor_style_corr = double_factor_style_corr.rename(
        columns={'level_1': '风格', '风格因子_双因子': '相关系数_双因子'})

    # 合并数据并设置offset
    style_corr = pd.merge(main_factor_style_corr, sub_factor_style_corr, how='left', on='风格')
    style_corr = pd.merge(style_corr, double_factor_style_corr, how='left', on='风格')
    style_corr['风格'] = style_corr['风格'].apply(lambda x: x.split('_')[1])

    # 获取双因子相关系数
    main_sub_corr = float_num_process(style_corr[style_corr['风格'] == '双因子']['相关系数_双因子'].iloc[-1])
    main_comp_corr = float_num_process(style_corr[style_corr['风格'] == '双因子']['相关系数_主因子'].iloc[-1])
    sub_comp_corr = float_num_process(style_corr[style_corr['风格'] == '双因子']['相关系数_次因子'].iloc[-1])

    corr_txt = f'corr(主，次)：{main_sub_corr}    corr(主，复)：{main_comp_corr}    corr(次，复)：{sub_comp_corr}'
    # 删除双因子相关系数
    style_corr = style_corr[style_corr['风格'] != '双因子']
    print(f'因子风格分析完成，耗时：{datetime.datetime.now() - start_date}')
    return style_corr, corr_txt


# endregion

def read_period_and_offset_file(file_path):
    """
    载入周期offset文件
    """
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, encoding='gbk', parse_dates=['交易日期'], skiprows=1)
        return df
    else:
        print(f'文件{file_path}不存在，请获取period_offset.csv文件后再试')
        raise FileNotFoundError('文件不存在')


def import_index_data(path, date_range=(None, None), max_param=0):
    """
    导入指数数据并进行预处理

    参数:
    path (str): 指数数据文件的路径
    date_range (list, optional): 回测的时间范围，格式为 [开始日期, 结束日期]，默认为 [None, None]
    max_param (int, optional): 因子的最大周期数，用于控制开始日期，确保rolling类因子，前置数据不是NaN，默认为 0

    返回:
    DataFrame: 处理后的指数数据，包含交易日期和指数涨跌幅
    """
    # 导入指数数据
    df_index = pd.read_csv(path, parse_dates=['candle_end_time'], encoding='gbk')

    # 计算涨跌幅
    df_index['指数涨跌幅'] = df_index['close'].pct_change()
    # 第一天的指数涨跌幅是开盘买入的涨跌幅
    df_index['指数涨跌幅'] = df_index['指数涨跌幅'].fillna(value=df_index['close'] / df_index['open'] - 1)

    # 去除涨跌幅为空的行
    df_index.dropna(subset=['指数涨跌幅'], inplace=True)

    # 重命名列
    df_index.rename(columns={'candle_end_time': '交易日期'}, inplace=True)

    # 根据日期范围过滤数据
    if date_range[0]:
        if max_param == 0:
            df_index = df_index[df_index['交易日期'] >= pd.to_datetime(date_range[0])]
            # print(f'💡 回测开始时间：{df_index["交易日期"].iloc[0].strftime("%Y-%m-%d")}')
        # 当提供了周期数之后
        else:
            # 计算新的开始日期
            start_index = df_index[df_index['交易日期'] >= pd.to_datetime(date_range[0])].index[0]
            start_date = df_index['交易日期'][start_index].strftime("%Y-%m-%d")

            # 移动周期，获取可以让因子数值不为Nan的开始日期
            shifted_date = df_index['交易日期'].shift(max_param)
            shifted_date.bfill(inplace=True)  # 前置数据不是NaN

            # 过滤前置数据
            df_index = df_index[df_index['交易日期'] >= shifted_date[start_index]]
            new_start_date = df_index['交易日期'].iloc[0].strftime("%Y-%m-%d")
            print(f'💡 回测开始时间：{start_date}，移动{max_param}个周期，最新交易日：{new_start_date}')
    if date_range[1]:
        df_index = df_index[df_index['交易日期'] <= pd.to_datetime(date_range[1])]
        # print(f'回测结束时间：{df_index["交易日期"].iloc[-1].strftime("%Y-%m-%d")}')

    # 按时间排序并重置索引
    df_index.sort_values(by=['交易日期'], inplace=True)
    df_index.reset_index(inplace=True, drop=True)

    return df_index
