import os
import streamlit as st
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import math

#########################
# 유사도 임계치(Threshold)
#########################
SIMILARITY_THRESHOLD = 0.6  # 이 값 이상인 문서만 출처로 표시

# 간단한 코사인 유사도 계산 함수
def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if (norm_a * norm_b) == 0:
        return 0.0
    return dot / (norm_a * norm_b)

# .env 파일 로드
load_dotenv()

# OpenAI API 설정
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key
if not api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# 프롬프트 파일 경로 설정
prompt_file_path = "src/prompt/prompt.txt"

# 텍스트 파일을 읽어와 프롬프트 텍스트로 저장
def load_prompt_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# 파일에서 프롬프트 텍스트 읽기
prompt_text = load_prompt_from_file(prompt_file_path)

# 임베딩 모델 생성 - text-embedding-3-small 사용
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# 벡터 저장소 로드 (allow_dangerous_deserialization 인자를 추가)
vectorstore = FAISS.load_local("vdb/faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)

# lambda_mult 값에 따른 검색방법
# 1.0: 순수 유사도 기반 검색(다양성 고려 X) == similarity
# 0.0: 최대 다양성 추구(유사도 고려 X)
# 0.9: 유사도에 90% 가중치, 다양성에 10% 가중치
retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 5, 'fetch_k': 10, 'lambda_mult': 0.9}
)

# RAG 구성 요소 설정
prompt = PromptTemplate(
    input_variables=["question", "context"],  # 필요한 입력 변수 설정
    template=prompt_text  # 텍스트 파일에서 읽어온 프롬프트 텍스트
)

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Streamlit 앱 설정
st.set_page_config(page_title="세무사 챗봇", page_icon="🤖", layout="wide")

st.title("📄 세무사 챗봇")
st.write("질문을 입력하면 관련 문서를 검색하고 답변을 생성합니다.")

# 사용자 입력 폼
with st.form("chat_form"):
    question = st.text_input(
        "질문을 입력하세요:",
        placeholder="예: 대학원생인 배우자가 2024년 6월에 연구용역비로 500만원을 받은 경우 배우자공제가 가능해?"
    )
    submit_button = st.form_submit_button(label="질문하기")

if submit_button and question:
    # 문서 검색 (MMR 사용)
    retrieved_documents = retriever.invoke(question)

    # 검색된 문서가 없을 경우 처리
    if not retrieved_documents:
        st.warning("관련 문서를 찾을 수 없습니다. 다른 질문을 입력해주세요.")
    else:
        # RAG를 사용하여 응답 생성
        with st.spinner("답변 생성 중..."):
            response = rag_chain.invoke(question)

        # ---------------------------
        # 답변 출력
        # ---------------------------
        st.subheader("💡 생성된 답변")
        st.write(response)

        # ---------------------------
        # 출처 표시 전, 질문-문서 유사도 계산
        # ---------------------------
        with st.spinner("출처 문서 점수 계산 중..."):
            # 질문 임베딩 미리 계산
            question_embedding = embedding_model.embed_query(question)

            # MMR로 선정된 문서를 순회하면서 유사도 계산
            documents_to_display = []
            for doc in retrieved_documents:
                # 문서 내용 임베딩 (대략적으로 문서 전체를 query처럼 embed)
                doc_embedding = embedding_model.embed_query(doc.page_content)
                sim = cosine_similarity(question_embedding, doc_embedding)

                # 임계치 이상이면 출처로 표시 목록에 추가
                if sim >= SIMILARITY_THRESHOLD:
                    documents_to_display.append((doc, sim))

        # ---------------------------
        # 리트리버된 문서 중 필터링된 것만 Expand로 출력
        # ---------------------------
        st.subheader("🔍 참조한 문서")
        if not documents_to_display:
            st.info(f"**유사도 {SIMILARITY_THRESHOLD} 이상**인 문서가 없습니다. (출처 없음)")
        else:
            for idx, (doc, sim_score) in enumerate(documents_to_display, 1):
                with st.expander(f"문서 {idx}: {doc.metadata.get('제목', '제목 없음')} (유사도: {sim_score:.3f})"):
                    st.write(f"**본문:** {doc.metadata.get('본문_원본', '없음')}")
                    st.write(f"**출처:** {doc.metadata.get('source', '출처 없음')}")
