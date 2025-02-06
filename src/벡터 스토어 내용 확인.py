import os
import faiss
import numpy as np
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ✅ FAISS 벡터스토어 로드
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = FAISS.load_local(
    "vdb/faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True
)

# ✅ FAISS 인덱스 가져오기
dense_index = vectorstore.index

# ✅ 전체 벡터 개수 확인
num_vectors = dense_index.ntotal
if num_vectors == 0:
    print("❌ 벡터스토어에 저장된 벡터가 없습니다.")
    exit()

print(f"✅ 벡터 개수: {num_vectors}")

# ✅ 빈 쿼리로 전체 데이터 검색 (전체 벡터 반환)
query_vector = np.zeros((1, dense_index.d))  # 빈 벡터 사용
D, I = dense_index.search(query_vector, num_vectors)  # 전체 검색

# ✅ FAISS에 저장된 메타데이터 가져오기
documents = []
for idx in I[0]:  # 검색된 인덱스 리스트
    if idx == -1:
        continue

    doc = vectorstore.docstore.search(str(idx))

    if isinstance(doc, str):  # 🔹 doc이 문자열인 경우 (JSON 변환 시도)
        try:
            doc = eval(doc)  # 문자열을 딕셔너리로 변환
        except:
            doc = {}  # 변환 실패 시 빈 딕셔너리 사용

    metadata = doc.get("metadata", {})  # 메타데이터 가져오기

    documents.append({
        "FAISS 인덱스": idx,
        "파일명": metadata.get("파일명", "N/A"),
        "문서명": metadata.get("문서명", "N/A"),
        "제목": metadata.get("제목", "N/A"),
        "본문_원본": metadata.get("본문_원본", "N/A"),
        "source": metadata.get("source", "N/A"),
        "유사도 거리": D[0][np.where(I[0] == idx)][0],  # 유사도 거리
    })

# ✅ DataFrame 생성
df = pd.DataFrame(documents)

# ✅ 엑셀 저장 경로 설정
desktop_path = os.path.join(os.path.expanduser("~"), "바탕화면")
save_path = os.path.join(desktop_path, "FAISS_메타데이터.xlsx")

# ✅ 엑셀로 저장
df.to_excel(save_path, index=False)
print(f"✅ FAISS 메타데이터를 엑셀로 저장했습니다: {save_path}")
print(f"✅ FAISS 벡터 개수 (dense_index): {dense_index.ntotal}")
print(f"✅ FAISS 저장된 문서 개수 (docstore): {len(vectorstore.docstore._dict)}")
