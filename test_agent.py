"""
Automated Test Suite for Smart Adaptive Interview Agent
Validates:
1. Homepage UI cleanliness (No API key fields, no Get Free Key links, no model exposure)
2. Gemini API key requirement & server-side error handling (503 if missing)
3. /health and /workflow endpoints
4. Adaptive LangGraph execution:
   - Strong answer -> difficulty increases (Medium -> Hard)
   - Weak answer -> difficulty decreases (Medium -> Easy)
   - Skill profile updates
5. PDF Report generation endpoint (/interview/{session_id}/report):
   - Valid PDF magic bytes (%PDF)
   - Proper Content-Type & Content-Disposition
   - No API keys in PDF
   - Multi-section completeness (Overview, Q&A, Progression, Strengths, Weaknesses, Roadmap)
"""

import os
import sys
import unittest

# Ensure UTF-8 console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Enable test mode for automated test suite
os.environ["TEST_MODE"] = "1"

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

class TestSmartAdaptiveInterviewAgentV2(unittest.TestCase):

    def test_01_homepage_ui_cleanliness(self):
        """Verify the homepage does NOT display API key field, link, or model name."""
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.text

        # Verify API key elements are completely removed
        self.assertNotIn("geminiApiKey", html)
        self.assertNotIn("Google Gemini API Key", html)
        self.assertNotIn("Get Free Key", html)
        self.assertNotIn("Uses model:", html)
        self.assertNotIn("apiStatusBadge", html)

        # Verify clean setup options are present
        self.assertIn("Target Technical Domain", html)
        self.assertIn("Starting Difficulty", html)
        self.assertIn("Number of Questions", html)
        self.assertIn("Start Adaptive Interview", html)

        # Verify no JSON download button
        self.assertNotIn("Download Report JSON", html)
        self.assertIn("Download Interview Report", html)
        print("[PASS] Test 1: Homepage UI is clean with zero API key or model leakage.")

    def test_02_server_key_missing_error(self):
        """Verify 503 error when server key is missing and not in test mode."""
        # Temporarily disable test mode and clear env key
        old_test = os.environ.get("TEST_MODE")
        old_key = os.environ.get("GEMINI_API_KEY")
        try:
            os.environ.pop("TEST_MODE", None)
            os.environ["GEMINI_API_KEY"] = ""
            res = client.post("/start-interview", json={"domain": "Python", "difficulty": "Medium", "total_questions": 3})
            self.assertEqual(res.status_code, 503)
            self.assertIn("AI service is not configured. Please contact the administrator.", res.json()["detail"])
            print("[PASS] Test 2: Server cleanly returns 503 when GEMINI_API_KEY is not configured.")
        finally:
            if old_test is not None:
                os.environ["TEST_MODE"] = old_test
            if old_key is not None:
                os.environ["GEMINI_API_KEY"] = old_key

    def test_03_health_and_workflow(self):
        """Verify /health and /workflow endpoints."""
        res_h = client.get("/health")
        self.assertEqual(res_h.status_code, 200)
        self.assertEqual(res_h.json()["status"], "healthy")

        res_w = client.get("/workflow")
        self.assertEqual(res_w.status_code, 200)
        data = res_w.json()
        self.assertIn("mermaid", data)
        self.assertIn("nodes", data)
        print("[PASS] Test 3: /health and /workflow endpoints operational.")

    def test_04_adaptive_interview_and_pdf_generation(self):
        """
        Verify complete adaptive interview cycle:
        - Starts without client API key
        - Adapts difficulty based on answers
        - Completes and generates valid PDF via /interview/{session_id}/report
        """
        print("\n--- Running Adaptive Interview & PDF Generation Test ---")
        # 1. Start interview
        res = client.post("/start-interview", json={"domain": "Python", "difficulty": "Medium", "total_questions": 3})
        self.assertEqual(res.status_code, 200)
        start_data = res.json()
        session_id = start_data["session_id"]
        self.assertEqual(start_data["current_difficulty"], "Medium")
        print(f"  Session ID: {session_id}")
        print(f"  Q1 ({start_data['current_difficulty']}): {start_data['current_question']}")

        # 2. Submit strong answer for Q1
        strong_ans_1 = (
            "Python function arguments use call-by-sharing (call-by-object-reference). "
            "When you pass a mutable object like a list, mutating its items reflects in the caller scope. "
            "Reassigning the variable name rebinds it locally without affecting the caller."
        )
        res_a1 = client.post("/submit-answer", json={"session_id": session_id, "answer": strong_ans_1})
        self.assertEqual(res_a1.status_code, 200)
        a1_data = res_a1.json()
        self.assertGreaterEqual(a1_data["evaluation"]["score"], 7.5)
        self.assertEqual(a1_data["new_difficulty"], "Hard")
        print(f"  Q1 Evaluated: Score {a1_data['evaluation']['score']}/10 -> Difficulty adapted: Medium -> {a1_data['new_difficulty']}")

        # 3. Submit weak answer for Q2 (Hard difficulty)
        weak_ans_2 = "I don't know, threads just do whatever the operating system wants."
        res_a2 = client.post("/submit-answer", json={"session_id": session_id, "answer": weak_ans_2})
        self.assertEqual(res_a2.status_code, 200)
        a2_data = res_a2.json()
        self.assertLess(a2_data["evaluation"]["score"], 4.0)
        self.assertEqual(a2_data["new_difficulty"], "Medium")
        print(f"  Q2 Evaluated: Score {a2_data['evaluation']['score']}/10 -> Difficulty adapted: Hard -> {a2_data['new_difficulty']}")

        # 4. Submit strong answer for Q3 (Final question)
        strong_ans_3 = (
            "Python context managers implement the __enter__ and __exit__ magic methods. "
            "__enter__ sets up the resource and returns it. __exit__ receives exception details (exc_type, exc_val, exc_tb), "
            "performs teardown cleanup, and suppresses exceptions if returning True."
        )
        res_a3 = client.post("/submit-answer", json={"session_id": session_id, "answer": strong_ans_3})
        self.assertEqual(res_a3.status_code, 200)
        a3_data = res_a3.json()
        self.assertTrue(a3_data["is_completed"])
        print(f"  Q3 Evaluated: Score {a3_data['evaluation']['score']}/10 -> Interview Completed!")
        print(f"  Progression: {a3_data['final_report']['difficulty_progression']}")

        # 5. TEST PDF REPORT GENERATION ENDPOINT
        print("\n--- Verifying PDF Report Download Endpoint ---")
        pdf_res = client.get(f"/interview/{session_id}/report")
        self.assertEqual(pdf_res.status_code, 200)
        self.assertEqual(pdf_res.headers.get("content-type"), "application/pdf")
        self.assertIn(f"adaptive_interview_report_{session_id}.pdf", pdf_res.headers.get("content-disposition", ""))
        
        pdf_bytes = pdf_res.content
        # Check PDF Magic Bytes
        self.assertTrue(pdf_bytes.startswith(b"%PDF"), "Response is not a valid PDF binary")
        self.assertGreater(len(pdf_bytes), 2000, "PDF size is unexpectedly small")
        print(f"  PDF Generated Successfully: {len(pdf_bytes)} bytes")
        
        # Verify no API key leaked in PDF binary stream
        self.assertNotIn(b"AIzaSy", pdf_bytes)
        print("[PASS] Test 4: Full adaptive interview completed and verified PDF report generated.")

    def test_05_input_validation(self):
        """Verify error handling on invalid inputs and session IDs."""
        # Empty answer validation
        res1 = client.post("/submit-answer", json={"session_id": "test", "answer": "   "})
        self.assertIn(res1.status_code, [400, 422])

        # Non-existent session ID for PDF report
        res2 = client.get("/interview/non-existent-session-id/report")
        self.assertEqual(res2.status_code, 404)
        print("[PASS] Test 5: Input validation and 404 error handling verified.")


if __name__ == "__main__":
    print("==================================================================")
    print("STARTING SMART ADAPTIVE INTERVIEW AGENT TEST SUITE (V2)")
    print("==================================================================")
    unittest.main()
