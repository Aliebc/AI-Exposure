import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import logging
import argparse
from pandas.errors import ParserError
from pathlib import Path

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 定义统一的暴露度等级阈值
RISK_THRESHOLDS = {
    'E0': (0, 35),
    'E1': (36, 65),
    'E2': (66, 100)
}

# 列名映射配置
COLUMN_MAPPING = {
    'dwa_id': ['DWA编号', 'dwa_id', 'DWA_ID', 'DWA ID', 'ID'],
    'activity': ['工作活动名称', 'activity', 'DWA_Title', 'Task', '工作活动'],
    'score': ['AI暴露度分数', 'score', '暴露度分数', 'Score', '分数'],
    'risk_level': ['暴露度等级', 'risk_level', 'exposure', 'Risk Level', '暴露度']
}

# 尝试的编码和分隔符
ENCODINGS = ['utf-8', 'gbk', 'latin1', 'ISO-8859-1']
SEPARATORS = ['\t', ',', ';']


def configure_plot_settings():
    """配置图表显示设置"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        logger.info("中文显示设置完成")
    except Exception as e:
        logger.warning(f"配置图表设置时出错: {str(e)}")


def load_model_data(model: str, file_path: Path) -> pd.DataFrame:
    """
    加载模型数据文件并进行标准化处理
    
    参数:
        model (str): 模型名称
        file_path (Path): 文件路径
        
    返回:
        pd.DataFrame: 处理后的数据框
    """
    logger.info(f"开始加载 {model} 数据: {file_path}")
    
    if not file_path.exists():
        logger.error(f"文件不存在: {file_path}")
        return None

    try:
        # 尝试多种编码和分隔符组合
        for encoding in ENCODINGS:
            for sep in SEPARATORS:
                try:
                    df = pd.read_csv(
                        file_path,
                        encoding=encoding,
                        sep=sep,
                        engine='python',
                        on_bad_lines='skip',
                        dtype=str  # 初始读取为字符串以保留原始格式
                    )
                    logger.info(f"使用编码 {encoding} 和分隔符 '{sep}' 成功加载 {model} 数据")
                    
                    # 标准化列名
                    df = standardize_column_names(df)
                    
                    # 确保必要的列存在
                    if not validate_required_columns(df):
                        logger.warning(f"文件 {file_path} 缺少必要列，尝试下一组合")
                        continue
                        
                    # 添加模型标识
                    df['model'] = model
                    
                    # 处理数值型数据
                    df = process_numeric_columns(df)
                    
                    logger.info(f"{model}数据加载成功，记录数: {len(df)}")
                    return df
                    
                except Exception as e:
                    logger.warning(f"编码 {encoding} 和分隔符 '{sep}' 失败: {str(e)}")
                    continue
                    
        logger.error(f"所有编码和分隔符组合都失败: {file_path}")
        return None
        
    except Exception as e:
        logger.error(f"加载{model}数据失败: {str(e)}")
        return None


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """标准化数据框的列名"""
    # 重命名列
    rename_mapping = {}
    for target, possible in COLUMN_MAPPING.items():
        for col in possible:
            if col in df.columns:
                rename_mapping[col] = target
                break
                
    if rename_mapping:
        df = df.rename(columns=rename_mapping)
        
    return df


def validate_required_columns(df: pd.DataFrame) -> bool:
    """验证数据框是否包含必要列"""
    required_cols = ['dwa_id', 'score']
    for col in required_cols:
        if col not in df.columns:
            logger.warning(f"缺少必要列: {col}")
            return False
    return True


def process_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """处理数值型列"""
    if 'score' in df.columns:
        # 清洗分数列
        df['score'] = (
            df['score']
            .astype(str)
            .str.replace(r'[^\d.]', '', regex=True)
            .replace('', np.nan)
        )
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        
        # 移除无效行
        df = df.dropna(subset=['score'])
        
    return df


def calculate_stability_metrics(group: pd.DataFrame, models: list) -> dict:
    """
    为每个DWA组计算稳定性指标
    
    参数:
        group (pd.DataFrame): 分组数据
        models (list): 模型列表
        
    返回:
        dict: 包含稳定性指标的字典
    """
    # 基本统计
    score_count = group['score'].count()
    score_mean = group['score'].mean()
    score_std = group['score'].std()
    
    # 模型间一致性指标
    risk_levels = group['risk_level'].dropna().unique()
    risk_consistency = len(risk_levels) == 1
    
    # 计算模型间分数差异
    score_diff = group['score'].max() - group['score'].min()
    
    # 创建结果字典
    dwa_stat = {
        'dwa_id': group.name[0],
        'activity': group.name[1],
        'num_models': score_count,
        'mean_score': score_mean,
        'score_std': score_std,
        'score_range': score_diff,
        'risk_consistent': risk_consistency
    }
    
    # 添加每个模型的分数
    for model in models:
        model_score = group[group['model'] == model]['score']
        if not model_score.empty:
            dwa_stat[f'{model}_score'] = model_score.values[0]
        else:
            dwa_stat[f'{model}_score'] = np.nan
    
    return dwa_stat


def calculate_stable_score(row: pd.Series, models: list) -> float:
    """
    计算稳定分数，考虑模型间一致性和分数离散程度
    
    参数:
        row (pd.Series): 数据行
        models (list): 模型列表
        
    返回:
        float: 稳定分数
    """
    scores = []
    weights = []
    
    for model in models:
        score_col = f'{model}_score'
        if not pd.isna(row[score_col]):
            scores.append(row[score_col])
            
            # 权重：一致的数据点权重更高
            weight = 1.5 if row['risk_consistent'] else 1.0
            weights.append(weight)
    
    if scores:
        # 计算加权平均值
        return np.average(scores, weights=weights)
    return np.nan


def assign_risk_level(score: float) -> str:
    """
    基于稳定分数分配风险等级
    
    参数:
        score (float): 稳定分数
        
    返回:
        str: 风险等级
    """
    if pd.isna(score):
        return 'Unknown'
    
    for risk, (low, high) in RISK_THRESHOLDS.items():
        if low <= score <= high:
            return risk
    return 'Unknown'


def visualize_results(dwa_stats_df: pd.DataFrame, models: list, output_dir: Path):
    """
    生成并保存可视化结果
    
    参数:
        dwa_stats_df (pd.DataFrame): DWA统计结果
        models (list): 模型列表
        output_dir (Path): 输出目录
    """
    try:
        logger.info("生成可视化结果")
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 稳定分数分布
        plt.figure(figsize=(10, 6))
        sns.histplot(data=dwa_stats_df, x='stable_score', kde=True, bins=20)
        plt.title('Stable AI Replacement Score Distribution')
        plt.xlabel('Stable Replacement Score')
        plt.ylabel('Number of DWAs')
        
        # 添加风险等级阈值线
        for threshold in [35, 65]:
            plt.axvline(x=threshold, color='r', linestyle='--', alpha=0.7)
        plt.text(18, plt.ylim()[1] * 0.9, 'E0', fontsize=12)
        plt.text(50, plt.ylim()[1] * 0.9, 'E1', fontsize=12)
        plt.text(82, plt.ylim()[1] * 0.9, 'E2', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'stable_scores_distribution.png')
        plt.close()
        
        # 模型间分数差异热力图
        plt.figure(figsize=(12, 8))
        score_cols = [f'{model}_score' for model in models]
        score_corr = dwa_stats_df[score_cols].corr()
        sns.heatmap(score_corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation of Replacement Scores Across Models')
        plt.tight_layout()
        plt.savefig(output_dir / 'model_score_correlation.png')
        plt.close()
        
        # 风险等级分布
        plt.figure(figsize=(8, 6))
        risk_distribution = dwa_stats_df['risk_level'].value_counts()
        sns.barplot(x=risk_distribution.index, y=risk_distribution.values)
        plt.title('Distribution of Risk Levels')
        plt.xlabel('Risk Level')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(output_dir / 'risk_level_distribution.png')
        plt.close()
        
        logger.info(f"可视化结果已保存到 {output_dir}")

    except Exception as e:
        logger.error(f"可视化失败: {str(e)}")


def save_results(dwa_stats_df: pd.DataFrame, output_dir: Path):
    """
    保存结果到CSV文件
    
    参数:
        dwa_stats_df (pd.DataFrame): DWA统计结果
        output_dir (Path): 输出目录
    """
    try:
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存完整结果
        output_path = output_dir / 'dwa_stable_scores.csv'
        dwa_stats_df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"综合稳定分数结果已保存到 {output_path}")
        
        # 保存简化结果
        simplified_cols = ['dwa_id', 'activity', 'stable_score', 'risk_level', 'score_std']
        simplified_df = dwa_stats_df[simplified_cols]
        simplified_path = output_dir / 'dwa_stable_scores_simplified.csv'
        simplified_df.to_csv(simplified_path, index=False, encoding='utf-8')
        logger.info(f"简化结果已保存到 {simplified_path}")
        
    except Exception as e:
        logger.error(f"保存结果失败: {str(e)}")


def analyze_stable_scores(data_dir: Path, output_dir: Path):
    """
    主分析流程：计算稳定AI暴露度分数
    
    参数:
        data_dir (Path): 包含模型数据的目录
        output_dir (Path): 输出目录
    """
    # 获取模型文件
    model_files = {}
    for file_path in data_dir.glob('*.csv'):
        model_name = file_path.stem
        model_files[model_name] = file_path
    
    if not model_files:
        logger.error(f"在 {data_dir} 中未找到任何CSV文件")
        return
    
    logger.info(f"找到 {len(model_files)} 个模型文件")
    
    # 加载所有模型数据
    all_data = []
    for model, file_path in model_files.items():
        logger.info(f"正在加载 {model} 数据...")
        df = load_model_data(model, file_path)
        if df is not None and not df.empty:
            all_data.append(df)
            logger.info(f"{model} 数据加载完成，记录数: {len(df)}")
        else:
            logger.warning(f"{model} 数据加载失败或为空")
    
    if not all_data:
        logger.error("没有成功加载任何模型数据，程序终止")
        return
    
    # 合并数据
    all_data_df = pd.concat(all_data, ignore_index=True)
    models = all_data_df['model'].unique().tolist()
    logger.info(f"总数据加载完成，记录数: {len(all_data_df)}，包含模型: {', '.join(models)}")
    
    # 1. 为每个DWA计算稳定性指标
    logger.info("开始为每个DWA计算稳定性指标")
    
    # 按DWA分组处理
    grouped = all_data_df.groupby(['dwa_id', 'activity'])
    
    # 计算每个DWA的稳定性指标
    dwa_stats = []
    for name, group in grouped:
        dwa_stats.append(calculate_stability_metrics(group, models))
    
    # 创建DWA统计DataFrame
    dwa_stats_df = pd.DataFrame(dwa_stats)
    
    # 2. 计算综合稳定分数
    logger.info("计算综合稳定分数")
    dwa_stats_df['stable_score'] = dwa_stats_df.apply(
        lambda row: calculate_stable_score(row, models), 
        axis=1
    )
    
    # 3. 分配最终暴露度等级
    logger.info("分配最终暴露度等级")
    dwa_stats_df['risk_level'] = dwa_stats_df['stable_score'].apply(assign_risk_level)
    
    # 4. 保存结果
    save_results(dwa_stats_df, output_dir)
    
    # 5. 结果分析
    logger.info("进行结果分析")
    
    # 风险等级分布
    risk_distribution = dwa_stats_df['risk_level'].value_counts()
    print("\n暴露度等级分布:")
    print(risk_distribution)
    
    # 模型间一致性分析
    consistency_rate = dwa_stats_df['risk_consistent'].mean() * 100
    print(f"\n模型间暴露度等级一致率: {consistency_rate:.2f}%")
    
    # 6. 可视化
    visualize_results(dwa_stats_df, models, output_dir)
    
    # 7. 输出最不稳定和最稳定的DWA
    unstable_dwa = dwa_stats_df.nlargest(5, 'score_std')
    stable_dwa = dwa_stats_df.nsmallest(5, 'score_std')
    
    print("\n最不稳定的DWA（分数标准差最大）:")
    print(unstable_dwa[['dwa_id', 'activity', 'stable_score', 'score_std']])
    
    print("\n最稳定的DWA（分数标准差最小）:")
    print(stable_dwa[['dwa_id', 'activity', 'stable_score', 'score_std']])


def main():
    """主入口函数"""
    # 配置命令行参数
    parser = argparse.ArgumentParser(description='计算稳定AI替代分数')
    parser.add_argument(
        '-d', '--data-dir', 
        type=Path, 
        default=Path('data'),
        help='包含模型CSV文件的目录路径'
    )
    parser.add_argument(
        '-o', '--output-dir', 
        type=Path, 
        default=Path('results'),
        help='输出目录路径'
    )
    parser.add_argument(
        '-v', '--verbose', 
        action='store_true',
        help='启用详细日志输出'
    )
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
    
    # 配置图表设置
    configure_plot_settings()
    
    # 运行分析
    logger.info(f"开始分析，数据目录: {args.data_dir}, 输出目录: {args.output_dir}")
    analyze_stable_scores(args.data_dir, args.output_dir)
    logger.info("分析完成")


if __name__ == "__main__":
    main()
