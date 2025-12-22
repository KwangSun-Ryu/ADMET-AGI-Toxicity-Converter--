import subprocess
import os
import time
import random


# 항상 이 경로에서 실행 (WSL)
BASE_DIR = "/mnt/e/Google Drive/External SSD/HealthCare/ADMET/코드/Repository/Development/ADMET-AGI-Toxicity-Converter--"

# commit 메시지 파일(권장: BASE_DIR 기준 절대경로)
TEXT_PATH = os.path.join(BASE_DIR, "commit_messages.txt")

TOTAL_SECONDS = 5 * 60 * 60 # 12 hours
N = 200

REMOTE = "origin"
BRANCH = "main"

def read_messages(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    # 빈 줄 제거 (빈 메시지 커밋 방지)
    lines = [m for m in lines if m]
    return lines

def run_cmd(args):
    # args: ["git", "status"] 처럼 리스트로 전달 (따옴표/특수문자 안전)
    return subprocess.run(
        args,
        cwd=BASE_DIR,
        capture_output=True,
        text=True
    )

def git_commit_allow_empty(message):
    return run_cmd(["git", "commit", "--allow-empty", "-m", message])

def git_push(remote, branch):
    return run_cmd(["git", "push", remote, branch])

def generate_offsets():
    return sorted(random.uniform(0, TOTAL_SECONDS) for _ in range(N))

def ensure_repo_ready():
    result = run_cmd(["git", "rev-parse", "--show-toplevel"])
    if result.returncode != 0:
        raise RuntimeError(f"git repo 확인 실패:\n{result.stderr}")

def run():
    start = time.monotonic()
    offsets = generate_offsets()

    messages = read_messages(TEXT_PATH)
    if len(messages) < N:
        raise ValueError(f"commit_messages.txt 메시지가 {len(messages)}개입니다. 최소 {N}개가 필요합니다.")

    ensure_repo_ready()

    for idx, off in enumerate(offsets, 1):
        target = start + off
        sleep_sec = target - time.monotonic()
        if sleep_sec > 0:
            time.sleep(sleep_sec)

        message = messages[idx - 1]
        result = git_commit_allow_empty(message)

        if result.returncode != 0:
            print(f"[{idx}/{N}] COMMIT FAILED (code={result.returncode})", flush=True)
            if result.stderr:
                print(result.stderr, flush=True)
        else:
            print(f"[{idx}/{N}] COMMIT OK", flush=True)
            if result.stdout:
                print(result.stdout, flush=True)

    # 마지막에 한 번만 push
    push_result = git_push(REMOTE, BRANCH)
    if push_result.returncode != 0:
        print(f"[PUSH] FAILED (code={push_result.returncode})", flush=True)
        if push_result.stderr:
            print(push_result.stderr, flush=True)
    else:
        print("[PUSH] OK", flush=True)
        if push_result.stdout:
            print(push_result.stdout, flush=True)

if __name__ == "__main__":
    run()
