import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Set

import pandas as pd

from core.factor_calculator import calculate_factors, get_col_name, FactorHub
from core.utils.path_kit import get_folder_path
import config

# ====================================================================================================
# ** 配置与初始化 **
# 忽略警告并设定显示选项，以优化代码输出的可读性
# ====================================================================================================
warnings.filterwarnings('ignore')
pd.set_option('expand_frame_repr', False)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)


def get_factor_params_dict() -> Tuple[Dict[str, List], List[str]]:
    """
    统一提取所有因子的参数（选股因子和择时因子合并）
    
    Returns:
        Tuple[Dict[str, List], List[str]]: (因子参数字典, 财务列列表)
    """
    # 统一因子参数字典（包含选股因子和择时因子）
    factor_params_dict: Dict[str, Set] = {}
    fin_cols: List[str] = []
    
    print("🔍 提取因子参数...")
    
    for strategy in config.strategy_list:
        strategy_name = strategy.get('name', '未知策略')
        print(f"📋 处理策略：{strategy_name}")
        
        # 处理因子（选股因子和择时因子）
        factor_lists = [
            ("filter_list", "🎯 选股因子"),
            ("timing_list", "⏰ 择时因子")
        ]
        
        for list_key, display_name in factor_lists:
            if list_key in strategy:
                factors = strategy[list_key]
                if factors:
                    print(f"{display_name}：{len(factors)} 个")
                    for factor in factors:
                        factor_name = factor[0]
                        param = factor[2] if len(factor) > 2 else None
                        if factor_name not in factor_params_dict:
                            factor_params_dict[factor_name] = set()
                        factor_params_dict[factor_name].add(param)
        
        # 3. 提取财务列
        if "fin_cols" in strategy:
            strategy_fin_cols = strategy["fin_cols"]
            fin_cols.extend(strategy_fin_cols)
            print(f"💰 财务列：{len(strategy_fin_cols)} 个")

    # 转换set为list并统计
    total_factors = 0
    for factor_name in factor_params_dict:
        factor_params_dict[factor_name] = list(factor_params_dict[factor_name])
        total_factors += len(factor_params_dict[factor_name])
    
    print(f"✅ 因子提取完成：{len(factor_params_dict)} 种因子，{total_factors} 个参数组合")
    print(f"✅ 财务列：{len(set(fin_cols))} 个")
    
    return factor_params_dict, list(set(fin_cols))


def get_factor_col_name_list(factor_params_dict: Dict[str, List]) -> List[str]:
    """
    获取因子的列名称列表
    
    Args:
        factor_params_dict: 因子参数字典
        
    Returns:
        List[str]: 因子列名称列表
    """
    print("📝 生成因子列名称...")
    
    # 生成因子列名
    factor_col_name_list = []
    for factor_name, param_list in factor_params_dict.items():
        for param in param_list:
            col_name = get_col_name(factor_name, param)
            factor_col_name_list.append(col_name)
    
    # 去重并排序
    unique_col_names = list(sorted(set(factor_col_name_list)))
    
    print(f"✅ 因子列名称：{len(unique_col_names)} 个")
    
    return unique_col_names



def main():
    """主函数：执行因子计算流程"""
    print("=" * 60)
    print("🚀 开始因子计算流程")
    print("=" * 60)
    
    try:
        # 1. 获取配置
        print("\n📋 步骤1：获取因子配置")
        factor_params_dict, fin_cols = get_factor_params_dict()
        factor_col_name_list = get_factor_col_name_list(factor_params_dict)
        
        # 2. 获取运行时文件夹
        print("\n📁 步骤2：准备运行时文件夹")
        runtime_folder = get_folder_path(config.runtime_data_path, "运行缓存", config.backtest_name)
        print(f"   运行时文件夹：{runtime_folder}")
        
        # 3. 计算因子（选股因子和择时因子合并）
        print("\n🧮 步骤3：开始计算因子")
        calculate_factors(
            runtime_folder=str(runtime_folder),
            factor_params_dict=factor_params_dict,
            factor_col_name_list=factor_col_name_list,
            fin_cols=fin_cols,
            start_date=config.start_date,
            end_date=config.end_date,
        )
        
        print("\n" + "=" * 60)
        print("✅ 因子计算流程完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 因子计算失败：{e}")
        raise


if __name__ == '__main__':
    main()
