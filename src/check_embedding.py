import streamlit as st
import numpy as np
from langchain_openai import OpenAIEmbeddings
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 임베딩 모델 초기화
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


def process_embeddings(texts):
    # 임베딩 생성
    embeddings = embedding_model.embed_documents(texts)
    embeddings_array = np.array(embeddings)

    # 임베딩 정보 출력
    st.write(f"텍스트 수: {len(texts)}")
    st.write(f"임베딩 배열 shape: {embeddings_array.shape}")

    return embeddings_array


def create_visualization(embeddings_array, texts):
    n_samples = len(texts)

    if n_samples >= 3:
        # 3D 시각화
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(embeddings_array)
        df = pd.DataFrame(reduced, columns=['PC1', 'PC2', 'PC3'])
        df['text'] = texts

        fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', text='text',
                            title='텍스트 임베딩 3D 시각화')
        fig.update_traces(textposition='top center',
                          marker=dict(size=6)
                          )

    else:
        # 2D 시각화
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings_array)
        df = pd.DataFrame(reduced, columns=['PC1', 'PC2'])
        df['text'] = texts

        fig = px.scatter(df, x='PC1', y='PC2', text='text',
                         title='텍스트 임베딩 2D 시각화')
        fig.update_traces(textposition='top center')

    # 설명된 분산 비율 출력
    explained_variance = pca.explained_variance_ratio_
    st.write("\n각 주성분이 설명하는 분산 비율:")
    for i, ratio in enumerate(explained_variance):
        st.write(f"PC{i + 1}: {ratio:.4f} ({ratio * 100:.2f}%)")

    return fig, df, embeddings_array


def calculate_similarities(embeddings_array, texts):
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

    # 코사인 유사도 계산
    cos_sim = cosine_similarity(embeddings_array)

    # 유클리드 거리 계산
    euc_dist = euclidean_distances(embeddings_array)

    # 결과를 데이터프레임으로 변환
    cos_df = pd.DataFrame(cos_sim, columns=texts, index=texts)
    euc_df = pd.DataFrame(euc_dist, columns=texts, index=texts)

    return cos_df, euc_df


# Streamlit UI
st.title('텍스트 임베딩 시각화')
st.write("텍스트를 입력하면 임베딩 후 시각화합니다. (2개: 2D, 3개 이상: 3D)")

# 사용자 입력
text_input = st.text_area(
    "텍스트를 입력하세요 (줄바꿈으로 구분)",
    height=200,
    placeholder="예시:\n신용카드\n체크카드\n선불카드"
)

if st.button('임베딩 생성 및 시각화'):
    if text_input:
        # 텍스트 전처리
        texts = [text.strip()
                 for text in text_input.split('\n') if text.strip()]

        if len(texts) < 2:
            st.error("최소 2개 이상의 텍스트를 입력해주세요.")
        else:
            try:
                # 임베딩 처리
                embeddings_array = process_embeddings(texts)

                # 시각화 생성
                fig, df, embeddings_array = create_visualization(
                    embeddings_array, texts)
                st.plotly_chart(fig)

                # 유사도 계산 및 표시
                st.subheader("텍스트 간 유사도 측정")
                cos_df, euc_df = calculate_similarities(
                    embeddings_array, texts)

                col1, col2 = st.columns(2)

                with col1:
                    st.write("코사인 유사도 (1에 가까울수록 유사)")
                    st.dataframe(cos_df.style.format("{:.4f}"))

                with col2:
                    st.write("유클리드 거리 (0에 가까울수록 유사)")
                    st.dataframe(euc_df.style.format("{:.4f}"))

                # 가장 유사한/다른 쌍 찾기
                if len(texts) > 1:
                    # 코사인 유사도가 가장 높은 쌍 (대각선 제외)
                    cos_sim_no_diag = cos_df.copy()
                    np.fill_diagonal(cos_sim_no_diag.values, -1)
                    max_cos_idx = cos_sim_no_diag.values.argmax()
                    max_cos_i, max_cos_j = np.unravel_index(
                        max_cos_idx, cos_sim_no_diag.shape)

                    # 유클리드 거리가 가장 가까운 쌍 (대각선 제외)
                    euc_dist_no_diag = euc_df.copy()
                    np.fill_diagonal(euc_dist_no_diag.values, np.inf)
                    min_euc_idx = euc_dist_no_diag.values.argmin()
                    min_euc_i, min_euc_j = np.unravel_index(
                        min_euc_idx, euc_dist_no_diag.shape)

                    st.write("\n가장 유사한 텍스트 쌍:")
                    st.write(
                        f"코사인 유사도 기준: '{texts[max_cos_i]}' - '{texts[max_cos_j]}' (유사도: {cos_df.iloc[max_cos_i, max_cos_j]:.4f})")
                    st.write(
                        f"유클리드 거리 기준: '{texts[min_euc_i]}' - '{texts[min_euc_j]}' (거리: {euc_df.iloc[min_euc_i, min_euc_j]:.4f})")

                # PCA 결과와 상세 정보를 두 컬럼으로 표시
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("PCA 결과")
                    st.dataframe(df)

                with col2:
                    st.subheader("각 텍스트별 상세 정보")
                    for text in texts:
                        with st.expander(f"텍스트: {text}"):
                            idx = list(texts).index(text)
                            st.write("원본 임베딩 (처음 10개 값):")
                            st.write(embeddings_array[idx][:10])
                            st.write("PCA 결과:")
                            st.write(df.iloc[idx])

            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")
                st.write("오류 상세:", e)
    else:
        st.warning("텍스트를 입력해주세요.")

# 사이드바에 설명 추가
with st.sidebar:
    st.header("사용 방법")
    st.write("""
    1. 텍스트 입력 영역에 여러 줄의 텍스트를 입력합니다.
    2. 각 줄은 별도의 텍스트로 처리됩니다.
    3. '임베딩 생성 및 시각화' 버튼을 클릭합니다.
    4. 3개 이상의 텍스트 입력 시 3D, 2개 입력 시 2D로 시각화됩니다.
    5. 그래프에서 마우스로 회전하고 확대/축소할 수 있습니다.
    6. 각 텍스트의 원본 임베딩과 PCA 결과를 확인할 수 있습니다.
    """)
