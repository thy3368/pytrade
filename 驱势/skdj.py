from trump_analysis import TrumpTradeAnalyzer

# 创建分析器实例
analyzer = TrumpTradeAnalyzer()

# 获取市场分析（默认5分钟K线，24小时数据）
report, analysis_df = analyzer.analyze_market()

# 打印分析报告
analyzer.print_analysis(report)

# 保存分析数据到CSV文件（可选）
analysis_df.to_csv('trump_analysis.csv', index=False)