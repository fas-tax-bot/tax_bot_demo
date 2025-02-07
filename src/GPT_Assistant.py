import os
import json
import time
import openai
import re
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 1) 환경 변수 로드 및 설정
# ---------------------------------------------------------------------------
load_dotenv()  # .env 파일 로드

# 환경 변수 불러오기
api_key = os.getenv("OPENAI_ASSISTANT_API_KEY")
ASSISTANT_ID = os.getenv("ASSISTANT_ID")

# 필수 환경 변수 검증
missing_vars = []
if not api_key:
    missing_vars.append("OPENAI_ASSISTANT_API_KEY")
if not ASSISTANT_ID:
    missing_vars.append("ASSISTANT_ID")

if missing_vars:
    raise ValueError(f"🚨 환경 변수가 설정되지 않았습니다: {', '.join(missing_vars)}. .env 파일을 확인하세요.")

# 환경 변수 설정
os.environ["OPENAI_API_KEY"] = api_key
THREADS_FILE = "src/threads.json"  # 스레드 정보 저장 파일

# ---------------------------------------------------------------------------
# 2) 스레드 관리 관련 함수
# ---------------------------------------------------------------------------

# 🔹 저장된 스레드 목록 불러오기
def load_threads():
    """JSON 파일에서 저장된 스레드 목록을 불러옵니다."""
    if os.path.exists(THREADS_FILE):
        with open(THREADS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# 🔹 사람이 읽을 수 있는 KST 시간 변환 함수
def convert_to_kst(timestamp):
    """UTC 시간을 KST(한국 표준시)로 변환합니다."""
    return time.strftime('%Y-%m-%d %H:%M:%S KST', time.localtime(timestamp + 9 * 3600))

# 🔹 새 Thread 생성 및 ID 반환 (생성된 시간과 함께 저장)
def create_new_thread():
    """
    OpenAI API를 사용해 새로운 스레드를 생성합니다.
    생성된 스레드 ID를 반환합니다.
    """
    thread_id = openai.beta.threads.create().id
    return thread_id

# ---------------------------------------------------------------------------
# 3) 메시지 전송 및 응답 처리 관련 함수
# ---------------------------------------------------------------------------

# 🔹 thread_id로 메시지 전송 후 run 반환
def submit_message(assistant_id, thread_id, user_message):
    """
    주어진 thread_id에 사용자의 메시지를 전송하고 실행(run)을 시작합니다.
    """
    openai.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_message
    )
    run = openai.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id
    )
    return run

# 🔹 thread_id의 run 상태 확인
def wait_on_run(run, thread_id):
    """
    실행 중인 run의 상태를 확인하고, 'completed' 상태가 될 때까지 대기합니다.
    """
    while run.status in ["queued", "in_progress"]:
        run = openai.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run

# 🔹 스레드의 run 상태가 완료되었을 때 응답 메시지 가져오기
def get_response(thread_id):
    """
    주어진 thread_id에서 실행 완료된 메시지 리스트를 가져옵니다.
    """
    return openai.beta.threads.messages.list(thread_id=thread_id, order="asc")

# ---------------------------------------------------------------------------
# 4) Assistant의 답변에서 특수 텍스트 처리
# ---------------------------------------------------------------------------

# 🔹 Assistant의 답변에서 【 】로 감싸진 특정 텍스트 처리
def clean_special_brackets(messages):
    """
    Assistant의 답변 중 【 】로 감싸진 텍스트를 처리:
    1. 내부에 한글이 포함된 경우:
        - 【와 † 사이의 글자와 †를 삭제.
    2. 내부에 한글이 포함되지 않은 경우:
        - 【 】 전체를 삭제 (내용 포함).
    """
    for message in messages:
        if message["role"] == "assistant":
            # 【 】로 감싸진 텍스트를 찾는 정규식
            pattern = r"【(.*?)】"

            def replacer(match):
                content = match.group(1)  # 【 】 내부 내용
                if re.search(r"[가-힣]", content):  # 내부에 한글이 포함된 경우
                    # 【와 † 사이의 글자 및 † 삭제
                    return re.sub(r"^(.*?)†", "", f"【{content}】")
                else:
                    # 한글이 포함되지 않은 경우 【 】 전체 삭제
                    return ""

            # 정규식 적용
            cleaned_message = re.sub(pattern, replacer, message["message"])
            message["message"] = cleaned_message.strip()  # 양쪽 공백 제거

# ---------------------------------------------------------------------------
# 5) 메시지 처리 함수
# ---------------------------------------------------------------------------

# 🔹 메시지 처리 함수 (외부 파일에서도 호출 가능)
def process_message(user_message, thread_id=None):
    """
    사용자 질문을 처리하고 응답을 JSON 형태로 반환하는 함수.
    기존 thread_id가 없다면 새 스레드를 생성합니다.
    """
    # 기존 스레드 사용 여부 결정
    if not thread_id:
        thread_id = create_new_thread()
    else:
        print(f"🔹 기존 THREAD_ID 사용: {thread_id}")
    
    # 메시지 전송 후 실행
    run = submit_message(ASSISTANT_ID, thread_id, user_message)

    # 실행 상태 확인 후 완료될 때까지 대기
    wait_on_run(run, thread_id)

    # 응답 메시지 가져오기
    response = get_response(thread_id)

    # JSON 데이터 생성
    conversation_data = {"thread_id": thread_id, "messages": []}

    for res in response.data:
        role = res.role.lower()  # "user" 또는 "assistant"
        if hasattr(res, "content") and isinstance(res.content, list):
            content_text = res.content[0].text.value if hasattr(res.content[0], "text") else res.content[0]
            conversation_data["messages"].append({"role": role, "message": content_text})

    #  【 】로 감싸진 텍스트 처리
    clean_special_brackets(conversation_data["messages"])
    
    return json.dumps(conversation_data, ensure_ascii=False, indent=4)  # JSON 문자열 반환

# ---------------------------------------------------------------------------
# 6) 스레드 목록 출력 함수
# ---------------------------------------------------------------------------

# 🔹 스레드 목록 출력 함수
def list_threads():
    """
    저장된 스레드 목록을 출력합니다.
    """
    threads = load_threads()
    if not threads:
        print("🔹 현재 존재하는 스레드가 없습니다.")
        return

    print("📜 현재 존재하는 스레드 목록:")
    for t in threads:
        print(f"- Thread ID: {t['thread_id']}, 생성일: {t['created_at']}")

# ---------------------------------------------------------------------------
# 7) 직접 실행
# ---------------------------------------------------------------------------

# 🔹 직접 실행할 때만 main() 실행
if __name__ == "__main__":
    USER_MESSAGE = str(input("Query를 입력하세요: "))
    print(process_message(USER_MESSAGE))
    list_threads()
