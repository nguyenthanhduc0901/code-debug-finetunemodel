#!/usr/bin/env python3
import sys
import requests

# API configuration
API_URL = "http://localhost:8001/v1/chat/completions"
MODEL_NAME = "socratic-tutor"

# Pre-defined Socratic tutor instruction
SYSTEM_INSTRUCTION = """Bạn là một giáo viên dạy lập trình theo phương pháp Socratic. Học sinh đã viết code bị lỗi cho một bài toán. Bạn biết đề bài, đoạn code lỗi, mô tả lỗi và cách sửa đúng. Tuy nhiên, bạn TUYỆT ĐỐI KHÔNG ĐƯỢC tiết lộ trực tiếp lỗi sai hoặc cách sửa cho học sinh. Thay vào đó, hãy dẫn dắt học sinh tự phát hiện và sửa lỗi bằng cách đặt các câu hỏi gợi mở, gợi ý các chiến lược debug (như dùng câu lệnh print, dò code) và đưa ra các gợi ý đi đúng hướng. Bạn phải trả lời bằng tiếng Việt ngắn gọn, súc tích (chỉ đặt một câu hỏi tại một thời điểm).

<problem>
Write a function `sum_even(lst)` that takes a list of integers and returns the sum of all even numbers in the list.
Example:
sum_even([1, 2, 3, 4]) => 6
sum_even([1, 3, 5]) => 0
</problem>

<bug_code>
def sum_even(lst):
    total = 0
    for num in lst:
        if num % 2 == 1:
            total += num
    return total
</bug_code>

<bug_desc>
The condition on line 4 is checking if the number is odd (`num % 2 == 1`) instead of checking if it is even (`num % 2 == 0`). Consequently, the function returns the sum of all odd numbers.
</bug_desc>

<bug_fixes>
Change the condition on line 4 from `num % 2 == 1` to `num % 2 == 0`.
</bug_fixes>"""

def main():
    print("=" * 60)
    print("🎓 SOCRATIC TUTOR - INTERACTIVE TERMINAL CHAT")
    print("=" * 60)
    print("\n[Tình huống lập trình giả định]:")
    print("Đề bài: Viết hàm sum_even(lst) tính tổng các số chẵn trong danh sách.")
    print("Học sinh viết code lỗi:")
    print("```python")
    print("def sum_even(lst):")
    print("    total = 0")
    print("    for num in lst:")
    print("        if num % 2 == 1:  # <-- Bị lỗi ở đây (đang kiểm tra số lẻ)")
    print("            total += num")
    print("    return total")
    print("```")
    print("Hãy nhập tin nhắn của học sinh bên dưới để bắt đầu tương tác với Tutor.")
    print("(Gõ 'exit' hoặc 'quit' để thoát)\n")

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION}
    ]

    while True:
        try:
            student_msg = input("\n👤 Học sinh: ")
            if student_msg.strip().lower() in ['exit', 'quit']:
                print("\n🎓 Kết thúc phiên học. Tạm biệt!")
                break
            
            if not student_msg.strip():
                continue

            messages.append({"role": "user", "content": student_msg})

            print("🤖 Tutor đang nghĩ...", end="\r", flush=True)

            payload = {
                "model": MODEL_NAME,
                "messages": messages,
                "max_tokens": 128,
                "temperature": 0.7
            }

            response = requests.post(API_URL, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            tutor_msg = result["choices"][0]["message"]["content"]
            print(f"🤖 Tutor: {tutor_msg}")

            messages.append({"role": "assistant", "content": tutor_msg})

        except KeyboardInterrupt:
            print("\n🎓 Tạm biệt!")
            break
        except Exception as e:
            print(f"\n❌ Lỗi kết nối đến vLLM server: {e}")
            print("Đảm bảo rằng vLLM server đang chạy trên cổng 8001.")
            break

if __name__ == "__main__":
    main()
