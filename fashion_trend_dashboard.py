import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# 페이지 설정
st.set_page_config(page_title="패션 트렌드 분석 대시보드", layout="wide")

# CSS 스타일 추가
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #333;
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #fafafa;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #444;
        margin-top: 1rem;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# 제목
st.markdown("<h1 class='main-header'>패션 트렌드 분석 대시보드</h1>", unsafe_allow_html=True)

# 가상 데이터 생성
def generate_trend_keywords():
    seasons = ['SS22', 'FW22', 'SS23', 'FW23', 'SS24']
    keywords = {
        'SS22': {'미니멀': 95, '네온': 80, '오버사이즈': 75, '크롭': 70, '빈티지': 65, 
                '플리츠': 60, '스트라이프': 55, '체크': 50, '트위드': 45, '프릴': 40},
        'FW22': {'오버사이즈': 90, '레이어드': 85, '헤리티지': 80, '니트': 75, '빈티지': 70,
                '퍼': 65, '벨벳': 60, '체크': 55, '레더': 50, '메탈릭': 45},
        'SS23': {'컷아웃': 92, '와이드': 88, '비비드': 82, '크로셰': 78, '트랜스페런트': 72,
                '스트라이프': 68, '패치워크': 64, '플로럴': 60, '시스루': 56, '라인스톤': 52},
        'FW23': {'오버사이즈': 94, '빌드업': 86, '그런지': 84, '코듀로이': 76, '미니멀': 74,
                '체크': 66, '시어링': 64, '레트로': 58, '퀼팅': 54, '멀티포켓': 48},
        'SS24': {'바이오필릭': 96, '아르떼': 90, '소프트': 84, '투명': 82, '핸드메이드': 76,
                '코튼': 72, '메쉬': 68, '버클': 62, '프린지': 58, '리사이클': 56}
    }
    return seasons, keywords

def generate_color_data():
    seasons = ['SS22', 'FW22', 'SS23', 'FW23', 'SS24']
    colors = ['버터 옐로우', '스카이 블루', '세이지 그린', '라벤더', '테라코타', 
             '네이비', '에메랄드', '퍼플', '차콜', '올리브']
    
    data = []
    for season in seasons:
        for color in colors:
            popularity = random.randint(20, 100)
            data.append({'시즌': season, '색상': color, '인기도': popularity})
    
    return pd.DataFrame(data)

def generate_material_data():
    seasons = ['SS22', 'FW22', 'SS23', 'FW23', 'SS24']
    materials = ['코튼', '실크', '리넨', '데님', '울', '가죽', '벨벳', '트위드', '시폰', '저지']
    
    data = []
    for season in seasons:
        for material in materials:
            popularity = random.randint(15, 95)
            data.append({'시즌': season, '소재': material, '인기도': popularity})
    
    return pd.DataFrame(data)

def generate_pattern_data():
    seasons = ['SS22', 'FW22', 'SS23', 'FW23', 'SS24']
    patterns = ['스트라이프', '플로럴', '체크', '도트', '지오메트릭', 
               '타이다이', '카모', '애니멀', '페이즐리', '아가일']
    
    data = []
    for season in seasons:
        for pattern in patterns:
            popularity = random.randint(10, 90)
            data.append({'시즌': season, '패턴': pattern, '인기도': popularity})
    
    return pd.DataFrame(data)

def generate_time_series():
    # 트렌드 변화 추이 데이터 생성
    dates = pd.date_range(start='2022-01-01', end='2024-06-30', freq='MS')
    trends = ['미니멀', '오버사이즈', '빈티지', '네온', '크롭', '레이어드']
    
    data = []
    for trend in trends:
        base = random.randint(30, 70)
        for date in dates:
            # 랜덤한 변동성 추가
            value = base + random.randint(-15, 15) + random.randint(-10, 10) * np.sin(date.month/6*np.pi)
            value = max(5, min(value, 100))  # 5~100 사이로 제한
            
            # 특정 트렌드는 특정 시즌에 더 강하게 나타남
            if trend == '네온' and 3 <= date.month <= 8:  # 봄/여름에 강함
                value += 15
            if trend == '레이어드' and (date.month <= 2 or date.month >= 9):  # 가을/겨울에 강함
                value += 15
                
            # 데이터 포인트 추가
            data.append({'날짜': date, '트렌드': trend, '인기도': value})
    
    return pd.DataFrame(data)

# 데이터 생성
seasons, keywords = generate_trend_keywords()
color_df = generate_color_data()
material_df = generate_material_data()
pattern_df = generate_pattern_data()
trend_time_df = generate_time_series()

# 사이드바 설정
st.sidebar.title("분석 옵션")
selected_season = st.sidebar.selectbox("시즌 선택", seasons, index=len(seasons)-1)
chart_type = st.sidebar.radio("차트 유형", ["막대 차트", "레이더 차트", "히트맵"])
trend_categories = st.sidebar.multiselect("트렌드 추이 선택", 
                                         ['미니멀', '오버사이즈', '빈티지', '네온', '크롭', '레이어드'],
                                         default=['미니멀', '오버사이즈', '빈티지'])

# 대시보드 헤더 - 현재 선택된 시즌 표시
st.markdown(f"<h2 class='sub-header'>{selected_season} 시즌 트렌드 분석</h2>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# 3개 열로 레이아웃 구성
col1, col2 = st.columns([3, 2])

with col1:
    # 트렌드 키워드 시각화
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("트렌드 키워드")
    keywords_df = pd.DataFrame({
        '키워드': list(keywords[selected_season].keys()),
        '인기도': list(keywords[selected_season].values())
    }).sort_values('인기도', ascending=False)
    
    # 키워드 차트
    if chart_type == "막대 차트":
        fig = px.bar(keywords_df.head(10), x='키워드', y='인기도', color='키워드',
                    title=f"{selected_season} 인기 키워드 Top 10")
    elif chart_type == "레이더 차트":
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=keywords_df.head(8)['인기도'].tolist(),
            theta=keywords_df.head(8)['키워드'].tolist(),
            fill='toself',
            name=selected_season
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            title=f"{selected_season} 인기 키워드 분포"
        )
    else:  # 히트맵
        # 시즌별 키워드 인기도 매트릭스 생성
        keyword_matrix = {}
        for season in seasons:
            keyword_matrix[season] = keywords[season]
        df_heatmap = pd.DataFrame(keyword_matrix).fillna(0)
        fig = px.imshow(df_heatmap, color_continuous_scale='Viridis',
                        title="시즌별 키워드 인기도 히트맵")
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 트렌드 변화 추이 그래프
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("트렌드 변화 추이")
    filtered_trends = trend_time_df[trend_time_df['트렌드'].isin(trend_categories)]
    fig = px.line(filtered_trends, x='날짜', y='인기도', color='트렌드', 
                 title="시간에 따른 트렌드 변화",
                 labels={'인기도': '인기도 지수', '날짜': '날짜'},
                 line_shape='spline')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # 인기 색상 차트
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("인기 색상")
    season_colors = color_df[color_df['시즌'] == selected_season].sort_values('인기도', ascending=False).head(5)
    
    if chart_type == "막대 차트":
        fig = px.bar(season_colors, x='색상', y='인기도', color='색상',
                    title=f"{selected_season} 시즌 인기 색상 Top 5",
                    color_discrete_sequence=px.colors.qualitative.Pastel)
    elif chart_type == "레이더 차트":
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=season_colors['인기도'].tolist(),
            theta=season_colors['색상'].tolist(),
            fill='toself',
            name=selected_season
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            title=f"{selected_season} 인기 색상 분포"
        )
    else:  # 히트맵
        pivot_colors = color_df.pivot(index='색상', columns='시즌', values='인기도')
        fig = px.imshow(pivot_colors, text_auto=True, aspect="auto", color_continuous_scale='Viridis',
                       title=f"시즌별 색상 인기도 히트맵")
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 인기 소재 차트
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("인기 소재")
    season_materials = material_df[material_df['시즌'] == selected_season].sort_values('인기도', ascending=False).head(5)
    
    if chart_type == "막대 차트":
        fig = px.bar(season_materials, x='소재', y='인기도', color='소재',
                    title=f"{selected_season} 시즌 인기 소재 Top 5")
    elif chart_type == "레이더 차트":
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=season_materials['인기도'].tolist(),
            theta=season_materials['소재'].tolist(),
            fill='toself',
            name=selected_season
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            title=f"{selected_season} 인기 소재 분포"
        )
    else:  # 히트맵
        pivot_materials = material_df.pivot(index='소재', columns='시즌', values='인기도')
        fig = px.imshow(pivot_materials, text_auto=True, aspect="auto", color_continuous_scale='Viridis',
                       title=f"시즌별 소재 인기도 히트맵")
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 인기 패턴 차트
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("인기 패턴")
    season_patterns = pattern_df[pattern_df['시즌'] == selected_season].sort_values('인기도', ascending=False).head(5)
    
    if chart_type == "막대 차트":
        fig = px.bar(season_patterns, x='패턴', y='인기도', color='패턴',
                    title=f"{selected_season} 시즌 인기 패턴 Top 5")
    elif chart_type == "레이더 차트":
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=season_patterns['인기도'].tolist(),
            theta=season_patterns['패턴'].tolist(),
            fill='toself',
            name=selected_season
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            title=f"{selected_season} 인기 패턴 분포"
        )
    else:  # 히트맵
        pivot_patterns = pattern_df.pivot(index='패턴', columns='시즌', values='인기도')
        fig = px.imshow(pivot_patterns, text_auto=True, aspect="auto", color_continuous_scale='Viridis',
                       title=f"시즌별 패턴 인기도 히트맵")
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# 하단 정보
st.markdown("---")

# 시즌 요약 통계
st.subheader("시즌 통계 요약")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("키워드 수", len(keywords[selected_season]))
with col2:
    avg_keyword = np.mean(list(keywords[selected_season].values()))
    st.metric("평균 키워드 인기도", f"{avg_keyword:.1f}")
with col3:
    top_color = season_colors.iloc[0]['색상']
    st.metric("인기 색상", top_color)
with col4:
    top_material = season_materials.iloc[0]['소재']
    st.metric("인기 소재", top_material)

# 앱 정보
st.caption("이 대시보드는 가상의 데이터를 사용하여 만들어졌습니다. 실제 패션 트렌드 데이터를 활용하면 보다 정확한 분석이 가능합니다.")
st.caption(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# 앱 실행 방법 안내
with st.expander("앱 실행 방법"):
    st.code("""
    # 필요한 라이브러리 설치
    pip install streamlit pandas numpy plotly
    
    # 앱 실행
    streamlit run fashion_trend_dashboard.py
    """) 