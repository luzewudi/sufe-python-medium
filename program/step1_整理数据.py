import warnings
from pathlib import Path

import pandas as pd

from core.data_center import prepare_data
from core.utils.path_kit import get_folder_path
import config


# ====================================================================================================
# ** 配置与初始化 **
# 设置必要的显示选项及忽略警告，以优化代码输出的阅读体验
# ====================================================================================================
warnings.filterwarnings('ignore')  # 忽略不必要的警告
pd.set_option('expand_frame_repr', False)  # 使数据框在控制台显示不换行
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)


# ====================================================================================================
# ** 从backtest_config.py复制的函数 **
# ====================================================================================================
def get_data_availability():
    """检查数据可用性"""
    data_center_path = Path(config.data_center_path)
    stock_data_path = data_center_path / "stock-trading-data-pro"
    index_data_path = data_center_path / "stock-main-index-data"
    fin_data_path = data_center_path / "stock-fin-data"
    return {
        "股票数据": stock_data_path.exists(),
        "指数数据": index_data_path.exists(),
        "财务数据": fin_data_path.exists(),
    }


def get_runtime_folder():
    """获取运行时文件夹"""
    return get_folder_path(config.runtime_data_path, "运行缓存", config.backtest_name)


def get_paths():
    """获取所有路径"""
    data_center_path = Path(config.data_center_path)
    return {
        "data_center_path": data_center_path,
        "stock_data_path": data_center_path / "stock-trading-data-pro",
        "index_data_path": data_center_path / "stock-main-index-data",
        "fin_data_path": data_center_path / "stock-fin-data"
    }


def get_rebalance_time_list():
    """获取换仓时间列表"""
    # 从策略配置中提取换仓时间
    rebalance_time_list = []
    for strategy in config.strategy_list:
        rebalance_time = strategy.get("rebalance_time", "open")
        rebalance_time_list.append(rebalance_time)
    return list(set(rebalance_time_list))


def desc():
    """生成配置描述"""
    paths = get_paths()
    data_info = get_data_availability()
    
    info = f"""🔵 {config.backtest_name}
→ 回测周期：{config.start_date} -> {config.end_date}
→ 初始资金：￥{config.initial_cash:,.2f}
→ 费率设置：手续费{config.c_rate * 10000:,.1f}‱, 印花税{config.t_rate * 1000:,.1f}‰
→ 数据设置:
  - 股票数据: {"✅" if data_info["股票数据"] else "❌"}
  - 指数数据: {"✅" if data_info["指数数据"] else "❌"}
  - 财务数据: {"✅" if data_info["财务数据"] else "❌"}
→ 数据中心路径："{paths['data_center_path']}"
→ 换仓时间：{get_rebalance_time_list() if get_rebalance_time_list() else '∅ 否'}"""
    
    return info


if __name__ == '__main__':
    # 获取路径信息
    paths = get_paths()
    print("📁 路径配置:")
    print(f"  数据中心: {paths['data_center_path']}")
    print(f"  股票数据: {paths['stock_data_path']}")
    print(f"  指数数据: {paths['index_data_path']}")
    print(f"  财务数据: {paths['fin_data_path']}")
    
    # 检查数据可用性
    data_info = get_data_availability()
    print("\n📊 数据可用性:")
    for data_type, available in data_info.items():
        status = "✅" if available else "❌"
        print(f"  {status} {data_type}")
    
    # 显示配置信息
    print(f"\n{desc()}")

    # 获取运行时文件夹
    runtime_folder = get_runtime_folder()
    
    # 调用prepare_data函数，传入具体的路径参数
    prepare_data(
        stock_data_path=paths['stock_data_path'],
        index_data_path=paths['index_data_path'],
        runtime_folder=runtime_folder,
        rebalance_time_list=get_rebalance_time_list(),
        boost=True
    )