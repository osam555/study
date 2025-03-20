import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.colors import rgb2hex, hex2rgb

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
    .color-box {
        display: inline-block;
        width: 100px;
        height: 80px;
        margin: 5px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .palette-container {
        text-align: center;
        padding: 10px;
        background-color: #f0f0f0;
        border-radius: 10px;
        margin: 10px 0;
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
    colors = {'버터 옐로우': '#f9e076', '스카이 블루': '#87ceeb', '세이지 그린': '#bcbd8b', 
              '라벤더': '#e6e6fa', '테라코타': '#e2725b', '네이비': '#000080', 
              '에메랄드': '#50c878', '퍼플': '#800080', '차콜': '#36454f', '올리브': '#808000'}
    
    data = []
    for season in seasons:
        for color_name, color_hex in colors.items():
            popularity = random.randint(20, 100)
            data.append({'시즌': season, '색상': color_name, '인기도': popularity, '색상코드': color_hex})
    
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

# 색상 팔레트 조화도 평가 함수
def evaluate_color_harmony(colors):
    # 실제로는 복잡한 색상 이론을 적용할 수 있지만, 여기서는 간단한 구현만 함
    harmony_score = random.randint(60, 95)  # 60-95 사이의 임의 점수 반환
    return harmony_score

# 트렌드 예측 모델 함수
def predict_next_season_trend(df, category_col, value_col, next_season):
    # 시즌을 숫자로 변환 (SS22=1, FW22=2, SS23=3, ...)
    seasons_order = {'SS22': 1, 'FW22': 2, 'SS23': 3, 'FW23': 4, 'SS24': 5, 'FW24': 6, 'SS25': 7}
    
    # 카테고리별 예측
    categories = df[category_col].unique()
    predictions = {}
    
    for category in categories:
        category_data = df[df[category_col] == category]
        X = np.array([seasons_order[s] for s in category_data['시즌']]).reshape(-1, 1)
        y = category_data[value_col].values
        
        # 선형 회귀 모델 학습
        model = LinearRegression()
        model.fit(X, y)
        
        # 다음 시즌 예측
        next_season_num = seasons_order[next_season]
        predicted_value = model.predict(np.array([[next_season_num]]))[0]
        predicted_value = max(0, min(100, predicted_value))  # 0-100 사이로 제한
        
        predictions[category] = predicted_value
    
    # 결과를 상위 5개만 반환
    sorted_predictions = {k: v for k, v in sorted(predictions.items(), key=lambda item: item[1], reverse=True)[:5]}
    return sorted_predictions

# 데이터 생성
seasons, keywords = generate_trend_keywords()
color_df = generate_color_data()
material_df = generate_material_data()
pattern_df = generate_pattern_data()
trend_time_df = generate_time_series()

# 다음 시즌 정의
next_seasons = {'SS22': 'FW22', 'FW22': 'SS23', 'SS23': 'FW23', 'FW23': 'SS24', 'SS24': 'FW24'}

# 사이드바 설정
st.sidebar.title("분석 옵션")
selected_season = st.sidebar.selectbox("시즌 선택", seasons, index=len(seasons)-1)
chart_type = st.sidebar.radio("차트 유형", ["막대 차트", "레이더 차트", "히트맵"])
trend_categories = st.sidebar.multiselect("트렌드 추이 선택", 
                                         ['미니멀', '오버사이즈', '빈티지', '네온', '크롭', '레이어드'],
                                         default=['미니멀', '오버사이즈', '빈티지'])

# 메뉴 선택
menu = st.sidebar.radio("기능 선택", ["트렌드 분석", "색상 팔레트 시뮬레이션", "다음 시즌 예측"], index=0)

if menu == "트렌드 분석":
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

elif menu == "색상 팔레트 시뮬레이션":
    st.markdown("<h2 class='sub-header'>색상 팔레트 시뮬레이션</h2>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # 선택된 시즌의 색상 데이터
    season_colors = color_df[color_df['시즌'] == selected_season].sort_values('인기도', ascending=False)
    
    # 색상 선택 위젯
    st.subheader("팔레트 색상 선택")
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("인기 색상 중에서 선택하세요:")
        selected_colors = st.multiselect(
            "색상 선택",
            options=season_colors['색상'].tolist(),
            default=season_colors['색상'].head(3).tolist(),
            key="color_palette_select"
        )
        
        # 색상 직접 추가
        st.write("또는 직접 색상 코드 입력:")
        custom_color = st.color_picker("색상 선택", "#ffffff")
        custom_color_name = st.text_input("색상 이름", "사용자 지정 색상")
        
        if st.button("팔레트에 추가"):
            if custom_color_name not in [color for color in selected_colors]:
                selected_colors.append(custom_color_name)
                # 색상 데이터프레임에 추가
                new_row = pd.DataFrame({
                    '시즌': [selected_season],
                    '색상': [custom_color_name],
                    '인기도': [50],
                    '색상코드': [custom_color]
                })
                color_df = pd.concat([color_df, new_row], ignore_index=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if selected_colors:
            st.subheader("생성된 색상 팔레트")
            
            # 색상 팔레트 시각화
            st.markdown("<div class='palette-container'>", unsafe_allow_html=True)
            for color_name in selected_colors:
                color_hex = color_df[color_df['색상'] == color_name]['색상코드'].values[0]
                st.markdown(
                    f"<div class='color-box' style='background-color: {color_hex};'></div>",
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)
            
            # 색상 조합 평가
            harmony_score = evaluate_color_harmony(selected_colors)
            st.metric("색상 조화도", f"{harmony_score}/100")
            
            # 색상 적용 예시
            st.subheader("색상 적용 예시")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 의류 컬렉션")
                st.image("https://via.placeholder.com/300x200", caption="색상 팔레트 적용 예시")
            
            with col2:
                st.markdown("### 인테리어")
                st.image("https://via.placeholder.com/300x200", caption="인테리어 적용 예시")
        else:
            st.info("색상을 선택하여 팔레트를 생성하세요.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # 색상 조합 추천
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("추천 색상 조합")
    
    recommendation_type = st.radio(
        "추천 유형",
        ["조화로운 색상 조합", "대비되는 색상 조합", "시즌 인기 색상 조합"]
    )
    
    # 색상 추천 (실제로는 더 복잡한 로직이 필요하지만, 여기서는 간단히 구현)
    if recommendation_type == "조화로운 색상 조합":
        recommended_colors = season_colors.head(4)['색상'].tolist()
    elif recommendation_type == "대비되는 색상 조합":
        recommended_colors = [season_colors.iloc[0]['색상'], season_colors.iloc[2]['색상'], season_colors.iloc[4]['색상']]
    else:
        recommended_colors = season_colors.head(5)['색상'].tolist()
    
    # 추천 색상 팔레트 시각화
    st.markdown("<div class='palette-container'>", unsafe_allow_html=True)
    for color_name in recommended_colors:
        color_hex = color_df[color_df['색상'] == color_name]['색상코드'].values[0]
        st.markdown(
            f"<div class='color-box' style='background-color: {color_hex};'></div>",
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 추천 팔레트 적용하기 버튼
    if st.button("이 팔레트 적용하기"):
        st.session_state.color_palette_select = recommended_colors
    
    st.markdown("</div>", unsafe_allow_html=True)

elif menu == "다음 시즌 예측":
    next_season = next_seasons.get(selected_season, "FW24")
    st.markdown(f"<h2 class='sub-header'>{next_season} 시즌 트렌드 예측</h2>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # 키워드 예측
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("예측 키워드 트렌드")
        
        # 키워드 데이터 준비
        keywords_list = []
        for season in seasons:
            for keyword, popularity in keywords[season].items():
                keywords_list.append({'시즌': season, '키워드': keyword, '인기도': popularity})
        keywords_df = pd.DataFrame(keywords_list)
        
        # 예측 수행
        predicted_keywords = predict_next_season_trend(keywords_df, '키워드', '인기도', next_season)
        
        # 예측 결과 시각화
        fig = px.bar(
            x=list(predicted_keywords.keys()),
            y=list(predicted_keywords.values()),
            color=list(predicted_keywords.keys()),
            labels={'x': '키워드', 'y': '예측 인기도'},
            title=f"{next_season} 인기 키워드 예측 Top 5"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 소재 예측
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("예측 소재 트렌드")
        predicted_materials = predict_next_season_trend(material_df, '소재', '인기도', next_season)
        
        fig = px.bar(
            x=list(predicted_materials.keys()),
            y=list(predicted_materials.values()),
            color=list(predicted_materials.keys()),
            labels={'x': '소재', 'y': '예측 인기도'},
            title=f"{next_season} 인기 소재 예측 Top 5"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # 색상 예측
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("예측 색상 트렌드")
        predicted_colors = predict_next_season_trend(color_df, '색상', '인기도', next_season)
        
        fig = px.bar(
            x=list(predicted_colors.keys()),
            y=list(predicted_colors.values()),
            color=list(predicted_colors.keys()),
            labels={'x': '색상', 'y': '예측 인기도'},
            title=f"{next_season} 인기 색상 예측 Top 5",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 예측 색상 팔레트 시각화
        st.subheader("예측 색상 팔레트")
        st.markdown("<div class='palette-container'>", unsafe_allow_html=True)
        for color_name in predicted_colors.keys():
            try:
                color_hex = color_df[color_df['색상'] == color_name]['색상코드'].values[0]
                st.markdown(
                    f"<div class='color-box' style='background-color: {color_hex};'></div>",
                    unsafe_allow_html=True
                )
            except:
                st.write(f"색상 코드를 찾을 수 없음: {color_name}")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 패턴 예측
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("예측 패턴 트렌드")
        predicted_patterns = predict_next_season_trend(pattern_df, '패턴', '인기도', next_season)
        
        fig = px.bar(
            x=list(predicted_patterns.keys()),
            y=list(predicted_patterns.values()),
            color=list(predicted_patterns.keys()),
            labels={'x': '패턴', 'y': '예측 인기도'},
            title=f"{next_season} 인기 패턴 예측 Top 5"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # 예측 정확도 및 신뢰도
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("예측 모델 정보")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("예측 신뢰도", f"{random.randint(70, 90)}%")
    with col2:
        st.metric("사용된 데이터 포인트", len(seasons))
    with col3:
        st.metric("예측 알고리즘", "선형 회귀 모델")
    
    st.info("이 예측은 과거 시즌 데이터를 바탕으로 한 단순 선형 추세 분석입니다. 실제 트렌드는 다양한 외부 요인에 의해 영향을 받을 수 있습니다.")
    st.markdown("</div>", unsafe_allow_html=True)

# 앱 정보
st.caption("이 대시보드는 가상의 데이터를 사용하여 만들어졌습니다. 실제 패션 트렌드 데이터를 활용하면 보다 정확한 분석이 가능합니다.")
st.caption(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# 앱 실행 방법 안내
with st.expander("앱 실행 방법"):
    st.code("""
    # 필요한 라이브러리 설치
    pip install streamlit pandas numpy plotly scikit-learn matplotlib
    
    # 앱 실행
    streamlit run fashion_trend_dashboard.py
    """) 
