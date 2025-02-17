import streamlit as st
from wyckoff_analysis import WyckoffAnalyzer
import plotly.graph_objects as go
from datetime import datetime, timedelta

def main():
    st.title("威科夫分析器")
    
    # 侧边栏配置
    st.sidebar.header("分析配置")
    symbol = st.sidebar.text_input("交易对", value="TRUMPUSDT")
    interval = st.sidebar.selectbox(
        "时间间隔",
        options=["1h", "4h", "1d"],
        index=0
    )
    limit = st.sidebar.slider("数据点数量", 100, 1000, 500)
    
    # 显示选项配置
    st.sidebar.header("显示选项")
    show_btc = st.sidebar.checkbox("显示BTC走势", value=True)
    
    # 初始化分析器
    analyzer = WyckoffAnalyzer()
    
    try:
        # 获取数据并分析
        with st.spinner('正在获取数据...'):
            df = analyzer.get_market_data(symbol, interval, limit)
            phase, df = analyzer.analyze_wyckoff_phase(df)
            prediction = analyzer.predict_trend(df)
            
            # 获取BTC相关性
            if show_btc and analyzer.btc_data is not None:
                btc_corr = df['close'].corr(analyzer.btc_data['close'])
                st.sidebar.metric("BTC相关性", f"{btc_corr:.2%}")
        
        # 显示基本信息
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("当前价格", f"{df['close'].iloc[-1]:.2f}")
        with col2:
            st.metric("20日均价", f"{df['price_ma20'].iloc[-1]:.2f}")
        with col3:
            volume_trend = '上升' if df['volume_trend'].iloc[-1] else '下降'
            st.metric("成交量趋势", volume_trend)
        with col4:
            st.metric("预测方向", prediction['direction'])
        
        # 显示威科夫阶段和预测
        st.subheader("威科夫分析结果")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"当前阶段: {phase}")
        with col2:
            st.info(f"趋势预测: {prediction['description']}")
        
        # 显示图表
        st.subheader("技术分析图表")
        fig = analyzer.plot_analysis(df, phase, show_btc=show_btc)
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示详细数据
        if st.checkbox("显示原始数据"):
            st.dataframe(df.tail())
            
    except Exception as e:
        st.error(f"发生错误: {str(e)}")
        st.warning("请检查网络连接或代理设置")

if __name__ == "__main__":
    main()