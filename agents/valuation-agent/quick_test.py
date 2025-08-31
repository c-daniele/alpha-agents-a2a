#!/usr/bin/env python3
"""Quick test script for the Valuation Agent."""

import asyncio
import json
import uuid

import httpx


async def quick_test():
    """Run a quick test of the Valuation Agent."""
    print("🚀 Quick Valuation Agent Test")
    print("=" * 40)
    
    # Check if agent is running
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Health check
            print("1. Health Check...")
            health_response = await client.get("http://localhost:3001/health")
            health_response.raise_for_status()
            print(f"   ✅ Agent is running: {health_response.json()}")
            
            # Simple analysis request
            print("\n2. Apple Analysis...")
            request_payload = {
                "jsonrpc": "2.0",
                "method": "message/send",
                "id": "quick-test-001",
                "params": {
                    "message": {
                        "kind": "message",
                        "messageId": str(uuid.uuid4()),
                        "role": "user",
                        "parts": [
                            {
                                "kind": "text",
                                "text": "Analyze Apple stock. Provide key metrics and investment recommendation."
                            }
                        ],
                        "contextId": str(uuid.uuid4())
                    },
                    "metadata": {
                        "risk_tolerance": "neutral"
                    }
                }
            }
            
            print(f"   📤 Sending request...")
            analysis_response = await client.post(
                "http://localhost:3001",
                json=request_payload,
                headers={"Content-Type": "application/json"}
            )
            analysis_response.raise_for_status()
            
            result = analysis_response.json()
            print(f"   📥 Response received")
            
            if "result" in result:
                task = result["result"]
                status = task.get("status", {})
                state = status.get("state", "unknown")
                
                print(f"   📊 Task Status: {state}")
                
                if state == "completed":
                    # Extract the analysis text
                    message = status.get("message", {})
                    if message and "parts" in message and message["parts"]:
                        analysis_text = message["parts"][0].get("text", "No analysis text")
                        print(f"\n📈 ANALYSIS RESULT:")
                        print("-" * 40)
                        # Show first 500 characters
                        print(analysis_text[:500] + ("..." if len(analysis_text) > 500 else ""))
                    else:
                        print("   ⚠️ No analysis text found in response")
                elif state == "failed":
                    error_msg = status.get("message", {}).get("parts", [{}])[0].get("text", "Unknown error")
                    print(f"   ❌ Analysis failed: {error_msg}")
                else:
                    print(f"   ⏳ Task is in '{state}' state")
            else:
                print(f"   ❌ Unexpected response format: {result}")
            
            print("\n✅ Quick test completed!")
            
        except httpx.ConnectError:
            print("❌ Cannot connect to agent!")
            print("   Make sure the agent is running with:")
            print("   uv run python src/valuation_agent/main.py")
        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP Error {e.response.status_code}")
            try:
                error_detail = e.response.json()
                print(f"   Error: {error_detail}")
            except:
                print(f"   Response: {e.response.text}")
        except Exception as e:
            print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(quick_test())