import urllib.request
import json
import time

from app.security.jwt import JWTService

def test_api():
    base_url = "http://127.0.0.1:8000/api/v1/chat"
    
    # Generate valid admin JWT token
    token = JWTService().create_access_token({"sub": "admin_user", "role": "admin"})
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    video_id = "VIDEO-2026-08-13-14-20-13.mp4"

    queries = [
        ("How many people are in the CCTV?", "Count All People"),
        ("How many men are in the CCTV?", "Count Men"),
        ("How many women are in the CCTV?", "Count Women"),
        ("Is there any suspicious person in the CCTV?", "Suspicious Behavioral Investigation")
    ]

    print("=" * 80)
    print("🌐 VISTA Live FastAPI Endpoint E2E Verification")
    print("=" * 80)

    for query, label in queries:
        print(f"\n📌 Request: [{label}] \"{query}\"")
        payload = json.dumps({"query": query, "video_id": video_id}).encode("utf-8")
        req = urllib.request.Request(base_url, data=payload, headers=headers, method="POST")

        start = time.time()
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                elapsed = int((time.time() - start) * 1000)

                print(f"  • Status: {data.get('status')}")
                print(f"  • Response Time: {elapsed} ms")
                print(f"  • Answer: \"{data.get('answer')}\"")
                print(f"  • Evidence Count: {len(data.get('evidence', []))}")
                for card in data.get('evidence', []):
                    print(f"    - Card: ID={card.get('evidence_id')[:8]}... | Source={card.get('source')} | Cam={card.get('camera_id')}")
                    print(f"      Description: {card.get('description')}")
                print(f"  • Telemetry Steps: {[step.get('name') for step in data.get('execution', {}).get('steps', [])]}")
        except Exception as e:
            print(f"  ❌ Request failed: {e}")

if __name__ == "__main__":
    test_api()
