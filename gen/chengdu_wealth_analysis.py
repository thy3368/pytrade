import pandas as pd
import folium
import matplotlib.pyplot as plt
import seaborn as sns

class ChengduWealthAnalysis:
    def __init__(self):
        # 成都主要区域及其示例数据
        self.districts_data = {
            '高新区': {'wealthy_population': 25000, 'avg_house_price': 35000, 'luxury_cars': 15000},
            '锦江区': {'wealthy_population': 20000, 'avg_house_price': 32000, 'luxury_cars': 12000},
            '武侯区': {'wealthy_population': 18000, 'avg_house_price': 30000, 'luxury_cars': 11000},
            '青羊区': {'wealthy_population': 15000, 'avg_house_price': 28000, 'luxury_cars': 9000},
            '金牛区': {'wealthy_population': 12000, 'avg_house_price': 25000, 'luxury_cars': 7000},
            '成华区': {'wealthy_population': 10000, 'avg_house_price': 23000, 'luxury_cars': 6000},
        }
        
        # 成都各区域中心点经纬度
        self.district_locations = {
            '高新区': [30.5537, 104.0657],
            '锦江区': [30.5983, 104.0838],
            '武侯区': [30.5715, 104.0436],
            '青羊区': [30.6739, 104.0622],
            '金牛区': [30.6912, 104.0525],
            '成华区': [30.6597, 104.1016]
        }

    def create_wealth_map(self):
        """创建财富分布热力图"""
        # 创建成都地图
        chengdu_map = folium.Map(
            location=[30.6570, 104.0650],
            zoom_start=12
        )

        # 添加财富分布热力点
        for district, location in self.district_locations.items():
            wealth_data = self.districts_data[district]
            folium.Circle(
                location=location,
                radius=wealth_data['wealthy_population'] / 10,
                popup=f"{district}<br>富裕人口: {wealth_data['wealthy_population']}",
                color='red',
                fill=True
            ).add_to(chengdu_map)

        # 保存地图
        chengdu_map.save('成都财富分布图.html')

    def analyze_wealth_indicators(self):
        """分析财富指标"""
        df = pd.DataFrame(self.districts_data).T
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # MacOS 系统使用
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        # 创建图表
        plt.figure(figsize=(15, 10))
        
        # 绘制多个指标的对比图
        plt.subplot(2, 1, 1)
        df.plot(kind='bar')
        plt.title('成都各区域财富指标对比')
        plt.xlabel('区域')
        plt.ylabel('数值')
        plt.legend(title='指标')
        
        # 绘制相关性热力图
        plt.subplot(2, 1, 2)
        sns.heatmap(df.corr(), annot=True, cmap='YlOrRd')
        plt.title('财富指标相关性分析')
        
        plt.tight_layout()
        plt.savefig('成都财富分析.png')

    def generate_report(self):
        """生成分析报告"""
        report = "成都财富分布分析报告\n"
        report += "=" * 30 + "\n\n"
        
        # 计算总体统计
        total_wealthy = sum(d['wealthy_population'] for d in self.districts_data.values())
        avg_house_price = sum(d['avg_house_price'] for d in self.districts_data.values()) / len(self.districts_data)
        
        report += f"总富裕人口: {total_wealthy:,} 人\n"
        report += f"平均房价: {avg_house_price:,.0f} 元/平方米\n\n"
        
        # 各区域详细分析
        for district, data in self.districts_data.items():
            report += f"{district}分析：\n"
            report += f"- 富裕人口：{data['wealthy_population']:,} 人\n"
            report += f"- 平均房价：{data['avg_house_price']:,} 元/平方米\n"
            report += f"- 豪车保有量：{data['luxury_cars']:,} 辆\n\n"
        
        # 使用 utf-8 编码写入文件
        with open('成都财富分布报告.txt', 'w', encoding='utf-8') as f:
            f.write(report)

def main():
    analysis = ChengduWealthAnalysis()
    analysis.create_wealth_map()
    analysis.analyze_wealth_indicators()
    analysis.generate_report()
    print("分析完成，已生成地图、图表和报告文件。")

if __name__ == "__main__":
    main()