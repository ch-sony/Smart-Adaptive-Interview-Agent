import urllib.request
import json
import re

BASE_URL = "http://localhost:8000"

print("1. Testing GET / (Homepage UI)...")
with urllib.request.urlopen(f"{BASE_URL}/") as response:
    html = response.read().decode("utf-8")
    assert response.status == 200
    
    # Assertions for Change 1 (Removal of API Key from frontend)
    assert "geminiApiKey" not in html, "API key input field found in HTML!"
    assert "Google Gemini API Key" not in html, "Gemini API key label found in HTML!"
    assert "Get Free Key" not in html, "Get Free Key link found in HTML!"
    assert "gemini-3.5" not in html and "gemini-2.5" not in html, "Gemini model name exposed in HTML!"
    assert "apiStatusBadge" not in html, "API status badge found in HTML!"
    
    # Assertions for Clean UI elements
    assert "Target Technical Domain" in html
    assert "Starting Difficulty" in html
    assert "Number of Questions" in html
    assert "Start Adaptive Interview" in html
    
    # Assertions for Change 2 (Download Interview Report PDF instead of JSON)
    assert "Download Interview Report" in html, "Download Interview Report button missing!"
    assert "Download Report JSON" not in html, "Download Report JSON button found in HTML!"
    print("   [SUCCESS] Homepage UI is clean, no API keys or models exposed, and PDF download button is ready.")

print("\n2. Testing POST /start-interview (Using server-side GEMINI_API_KEY)...")
req_data = json.dumps({
    "domain": "Python",
    "difficulty": "Medium",
    "total_questions": 2
}).encode("utf-8")
req = urllib.request.Request(f"{BASE_URL}/start-interview", data=req_data, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as response:
    start_res = json.loads(response.read().decode("utf-8"))
    assert response.status == 200
    session_id = start_res["session_id"]
    q1 = start_res["current_question"]
    print(f"   [SUCCESS] Interview started! Session ID: {session_id}")
    print(f"   Q1: {q1[:80]}...")

print("\n3. Testing POST /submit-answer (LangGraph evaluation)...")
ans_data = json.dumps({
    "session_id": session_id,
    "answer": "Python passes arguments by assignment (call-by-sharing). Mutable objects like lists can be mutated in place, while reassigning creates a new reference."
}).encode("utf-8")
req = urllib.request.Request(f"{BASE_URL}/submit-answer", data=ans_data, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as response:
    ans_res = json.loads(response.read().decode("utf-8"))
    assert response.status == 200
    eval_score = ans_res["evaluation"]["score"]
    next_diff = ans_res.get("new_difficulty")
    print(f"   [SUCCESS] Answer evaluated! Score: {eval_score}/10, Next Difficulty: {next_diff}")

print("\n4. Testing POST /finish-interview...")
fin_data = json.dumps({"session_id": session_id}).encode("utf-8")
req = urllib.request.Request(f"{BASE_URL}/finish-interview", data=fin_data, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as response:
    fin_res = json.loads(response.read().decode("utf-8"))
    assert response.status == 200
    final_score = fin_res.get("final_score")
    print(f"   [SUCCESS] Interview finalized! Overall Score: {final_score}/10")

print(f"\n5. Testing GET /interview/{session_id}/report (PDF Generation & Download)...")
pdf_url = f"{BASE_URL}/interview/{session_id}/report"
with urllib.request.urlopen(pdf_url) as response:
    assert response.status == 200
    assert response.headers["Content-Type"] == "application/pdf"
    content_disp = response.headers.get("Content-Disposition", "")
    assert f"adaptive_interview_report_{session_id}.pdf" in content_disp
    
    pdf_bytes = response.read()
    assert pdf_bytes.startswith(b"%PDF-"), "Invalid PDF header!"
    assert len(pdf_bytes) > 2000, "PDF too small!"
    
    # Save a copy locally for inspection
    with open(f"adaptive_interview_report_{session_id}.pdf", "wb") as f:
        f.write(pdf_bytes)
        
    print(f"   [SUCCESS] PDF generated and downloaded! Size: {len(pdf_bytes)} bytes")
    print(f"   Saved to: adaptive_interview_report_{session_id}.pdf")

print("\n6. Verifying PDF contains NO API keys...")
with open(".env", "r") as f:
    env_content = f.read()
    key_match = re.search(r"GEMINI_API_KEY=(.*)", env_content)
    if key_match:
        actual_key = key_match.group(1).strip()
        if actual_key and len(actual_key) > 8:
            assert actual_key.encode("utf-8") not in pdf_bytes, "CRITICAL ERROR: API key leaked into PDF!"
            print(f"   [SUCCESS] Verified: Server GEMINI_API_KEY is NOT leaked in the PDF report!")

print("\n========================================================")
print("ALL LIVE END-TO-END VERIFICATIONS PASSED!")
print("========================================================")
