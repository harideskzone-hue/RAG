import sys
import os
import requests
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.security.jwt import JWTService

def main():
    parser = argparse.ArgumentParser(description="VISTA AI Interactive Terminal Chat")
    parser.add_argument("--url", default="http://localhost:8000/api/v1/chat", help="API URL")
    args = parser.parse_args()

    # Generate a valid local dev token
    jwt_service = JWTService()
    # For CLI testing, we give unrestricted admin access (allowed_cameras=None)
    token = jwt_service.create_access_token({"sub": "tester", "role": "admin", "allowed_cameras": None})

    print("="*60)
    print("VISTA AI Agentic RAG - Interactive Terminal")
    print("Type 'exit' or 'quit' to stop.")
    print("="*60)

    conversation_id = None

    while True:
        try:
            query = input("\n[You]: ")
            if query.lower() in ['exit', 'quit']:
                break
            if not query.strip():
                continue
                
            payload = {"query": query}
            if conversation_id:
                payload["conversation_id"] = conversation_id
                
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}"
            }
            
            # The API currently doesn't mandate auth for testing, but if it did, we'd add it here
            print("⏳ Agentic RAG processing (this might take a moment)...")
            response = requests.post(args.url, json=payload, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                conversation_id = data.get("conversation_id", conversation_id)
                print("\n[VISTA AI]:")
                print(data.get("answer", "No answer provided."))
                # Print evidence
                evidence = data.get("evidence", [])
                if evidence:
                    print("-" * 60)
                    print(f"Evidence Trace ({len(evidence)} items):")
                    for ev in evidence:
                        print(f" - ID: {ev.get('evidence_id')} | Cam: {ev.get('camera_id')} | Time: {ev.get('timestamp')}")
                        print(f"   Source: {ev.get('source')} | Confidence: {ev.get('confidence')}")
                        if ev.get('description'):
                            print(f"   Desc: {ev.get('description')}")
                    
            else:
                print(f"\n[Error {response.status_code}]: {response.text}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[Error]: {e}")

if __name__ == "__main__":
    main()
