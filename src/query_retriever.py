import os
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# OpenAI API 설정
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key
if not api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# 임베딩 모델 생성 - text-embedding-3-small 사용
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# 벡터 저장소 로드 (allow_dangerous_deserialization 인자를 추가)
vectorstore = FAISS.load_local("vdb/faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)

# lambda_mult가 크면 정확도 향상, 작으면 다양성 향상
retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={'k': 10, 'fetch_k': 20, 'lambda_mult': 0.9})

# RAG 구성 요소 설정
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.5)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 질문 반복 처리
while True:
    question = input("\n 질문을 입력하세요 (종료하려면 'c', 'C' 또는 'ㅊ' 입력): ")
    if question.lower() in ["c", "ㅊ"]:
        print("Q&A 루프를 종료합니다.")
        break

    # 리트리버에서 문서 검색
    retrieved_documents = retriever.invoke(question)

    # 검색된 문서가 없을 경우 처리
    if not retrieved_documents:
        print("\n관련 문서를 찾을 수 없습니다.")
        continue

    # 리트리버된 문서 출력
    print("\n리트리버된 문서:")
    for idx, doc in enumerate(retrieved_documents, 1):
        print(f"문서 {idx}:\n제목: {doc.metadata.get('제목', '없음')}\n본문: {doc.page_content}\n출처: {doc.metadata.get('source', '출처 없음')}\n")

    # RAG를 사용하여 응답 생성
    response = rag_chain.invoke(question)
    print("\n응답:", response)
