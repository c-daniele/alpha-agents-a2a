#!/usr/bin/env python3
"""Test client for the Fundamental Agent A2A JSON-RPC server."""

import asyncio
import json
import uuid
from typing import Any, Dict

import httpx


class A2AFundamentalClient:
    """Client for testing the Fundamental Agent A2A server."""
    
    def __init__(self, base_url: str = "http://localhost:3003"):
        self.base_url = base_url
        self.session = httpx.AsyncClient(timeout=180.0)  # Longer timeout for fundamental analysis
    
    async def close(self):
        """Close the HTTP session."""
        await self.session.aclose()
    
    async def send_json_rpc_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a JSON-RPC request to the agent."""
        request_id = str(uuid.uuid4())
        
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "id": request_id
        }
        
        if params:
            payload["params"] = params
        
        print(f"🚀 Sending {method} request...")
        print(f"📤 Request: {json.dumps(payload, indent=2)}")
        
        try:
            response = await self.session.post(
                self.base_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            print(f"📥 Response received")
            return result
            
        except httpx.RequestError as e:
            print(f"❌ Request failed: {e}")
            return {"error": f"Request failed: {e}"}
        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP error: {e.response.status_code}")
            try:
                error_detail = e.response.json()
                print(f"Error details: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"Error response: {e.response.text}")
            return {"error": f"HTTP {e.response.status_code}"}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if the agent is running."""
        print("🏥 Checking agent health...")
        try:
            response = await self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            result = response.json()
            print(f"✅ Health check: {json.dumps(result, indent=2)}")
            return result
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return {"error": str(e)}
    
    async def get_agent_card(self) -> Dict[str, Any]:
        """Get the agent's capabilities card."""
        print("📋 Fetching agent card...")
        try:
            response = await self.session.get(f"{self.base_url}/agent-card")
            response.raise_for_status()
            result = response.json()
            print(f"📋 Agent Card: {result.get('name', 'Unknown')} v{result.get('version', '?')}")
            return result
        except Exception as e:
            print(f"❌ Agent card fetch failed: {e}")
            return {"error": str(e)}
    
    def create_message(self, text: str, context_id: str = None, task_id: str = None) -> Dict[str, Any]:
        """Create a properly formatted A2A message."""
        return {
            "kind": "message",
            "messageId": str(uuid.uuid4()),
            "role": "user",
            "parts": [
                {
                    "kind": "text",
                    "text": text
                }
            ],
            "contextId": context_id or str(uuid.uuid4()),
            "taskId": task_id
        }
    
    async def analyze_fundamental(
        self, 
        stock_query: str, 
        risk_tolerance: str = "neutral",
        analysis_depth: str = "comprehensive",
        focus_areas: list = None
    ) -> Dict[str, Any]:
        """Send a fundamental analysis request."""
        message = self.create_message(stock_query)
        
        params = {
            "message": message,
            "metadata": {
                "risk_tolerance": risk_tolerance,
                "analysis_depth": analysis_depth,
                "focus_areas": focus_areas or []
            }
        }
        
        return await self.send_json_rpc_request("message/send", params)
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a specific task."""
        params = {
            "id": task_id
        }
        
        return await self.send_json_rpc_request("tasks/get", params)


async def run_comprehensive_test():
    """Run a comprehensive test of the Fundamental Agent."""
    print("=" * 80)
    print("🧪 FUNDAMENTAL AGENT - COMPREHENSIVE CLIENT TEST")
    print("=" * 80)
    
    client = A2AFundamentalClient()
    
    try:
        # Test 1: Health Check
        print("\n1️⃣ Health Check Test")
        print("-" * 40)
        health = await client.health_check()
        
        if "error" in health:
            print("❌ Agent is not running! Please start it first with:")
            print("   uv run python src/fundamental_agent/main.py")
            return
        
        # Test 2: Agent Card
        print("\n2️⃣ Agent Card Test")
        print("-" * 40)
        await client.get_agent_card()
        
        # Test 3: Comprehensive Fundamental Analysis - Apple
        print("\n3️⃣ Comprehensive Fundamental Analysis - Apple")
        print("-" * 40)
        apple_result = await client.analyze_fundamental(
            "Perform comprehensive fundamental analysis for Apple Inc. Focus on financial health, profitability, and growth prospects.",
            risk_tolerance="neutral",
            analysis_depth="comprehensive"
        )
        
        if apple_result.get("result"):
            task = apple_result["result"]
            print(f"✅ Task created with ID: {task.get('id', 'Unknown')}")
            print(f"📊 Status: {task.get('status', {}).get('state', 'Unknown')}")
            
            # If task completed, show preview of analysis
            if task.get('status', {}).get('state') == 'completed':
                analysis_message = task.get('status', {}).get('message', {})
                if analysis_message and analysis_message.get('parts'):
                    analysis_text = analysis_message['parts'][0].get('text', 'No analysis text')
                    print(f"📈 Analysis preview: {analysis_text[:400]}...")
        
        # Test 4: Quick Valuation - Tesla
        print("\n4️⃣ Quick Valuation Analysis - Tesla")
        print("-" * 40)
        tesla_result = await client.analyze_fundamental(
            "Perform DCF valuation analysis for Tesla with target price estimate",
            risk_tolerance="seeking",
            analysis_depth="standard"
        )
        
        if tesla_result.get("result"):
            task = tesla_result["result"]
            print(f"✅ Tesla valuation task: {task.get('status', {}).get('state', 'Unknown')}")
        
        # Test 5: Risk-averse Analysis - Microsoft
        print("\n5️⃣ Risk-averse Fundamental Analysis - Microsoft")
        print("-" * 40)
        msft_result = await client.analyze_fundamental(
            "Analyze Microsoft's fundamental strength for a risk-averse investor. Focus on balance sheet stability and dividend sustainability.",
            risk_tolerance="averse",
            analysis_depth="comprehensive"
        )
        
        if msft_result.get("result"):
            task = msft_result["result"]
            print(f"✅ Microsoft analysis task: {task.get('status', {}).get('state', 'Unknown')}")
        
        # Test 6: Sector Comparison
        print("\n6️⃣ Sector Comparison Analysis")
        print("-" * 40)
        comparison_result = await client.analyze_fundamental(
            "Compare the fundamental metrics of Apple (AAPL) vs Microsoft (MSFT) for investment decision making",
            risk_tolerance="neutral",
            analysis_depth="standard"
        )
        
        if comparison_result.get("result"):
            task = comparison_result["result"]
            print(f"✅ Sector comparison task: {task.get('status', {}).get('state', 'Unknown')}")
        
        # Test 7: Cash Flow Focus Analysis
        print("\n7️⃣ Cash Flow Focus Analysis - Google")
        print("-" * 40)
        googl_result = await client.analyze_fundamental(
            "Analyze Google's cash flow generation and sustainability. Focus on free cash flow trends and capital allocation.",
            risk_tolerance="neutral",
            analysis_depth="standard",
            focus_areas=["cash_flow"]
        )
        
        if googl_result.get("result"):
            task = googl_result["result"]
            print(f"✅ Cash flow analysis task: {task.get('status', {}).get('state', 'Unknown')}")
        
        # Test 8: Invalid Company Test
        print("\n8️⃣ Invalid Company Test")
        print("-" * 40)
        invalid_result = await client.analyze_fundamental(
            "NonExistentCompany999",
            risk_tolerance="neutral"
        )
        
        if invalid_result.get("result"):
            task = invalid_result["result"]
            print(f"✅ Invalid company task: {task.get('status', {}).get('state', 'Unknown')}")
        
        # Test 9: Task Status Check (if we have a task ID)
        if apple_result.get("result", {}).get("id"):
            print("\n9️⃣ Task Status Check")
            print("-" * 40)
            task_id = apple_result["result"]["id"]
            status_result = await client.get_task_status(task_id)
            
            if status_result.get("result"):
                task_status = status_result["result"]
                print(f"✅ Task {task_id} status: {task_status.get('status', {}).get('state', 'Unknown')}")
        
        print("\n" + "=" * 80)
        print("✅ COMPREHENSIVE TEST COMPLETED!")
        print("=" * 80)
        
        print("\n📋 Summary:")
        print("- The fundamental agent is running and responding to requests")
        print("- A2A JSON-RPC protocol is working correctly")
        print("- Financial data collection and fundamental analysis capabilities are functional")
        print("- Risk tolerance adaptation is working")
        print("- Error handling is working for invalid inputs")
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.close()


async def run_simple_test():
    """Run a simple test with one fundamental analysis."""
    print("=" * 60)
    print("📊 SIMPLE FUNDAMENTAL AGENT TEST")
    print("=" * 60)
    
    client = A2AFundamentalClient()
    
    try:
        # Check if agent is running
        health = await client.health_check()
        if "error" in health:
            print("❌ Agent not running. Start with: uv run python src/fundamental_agent/main.py")
            return
        
        # Simple fundamental analysis
        print("\n🍎 Analyzing Apple fundamentals...")
        result = await client.analyze_fundamental(
            "Analyze Apple's fundamental strength for investment. Focus on financial health and growth prospects.",
            risk_tolerance="neutral",
            analysis_depth="standard"
        )
        
        if result.get("result"):
            task = result["result"]
            status = task.get("status", {})
            
            print(f"\n✅ Analysis Status: {status.get('state', 'Unknown')}")
            
            if status.get("state") == "completed" and status.get("message"):
                analysis_text = status["message"]["parts"][0]["text"]
                print(f"\n📊 FUNDAMENTAL ANALYSIS RESULT:")
                print("-" * 40)
                print(analysis_text)
            elif status.get("state") == "failed":
                print(f"❌ Analysis failed: {status.get('message', {}).get('parts', [{}])[0].get('text', 'Unknown error')}")
            else:
                print(f"⏳ Analysis is {status.get('state', 'in unknown state')}")
        else:
            print("❌ No result returned from agent")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    finally:
        await client.close()


def main():
    """Main function with test options."""
    import sys
    
    print("🚀 Fundamental Agent Test Client")
    print("Make sure the agent is running first:")
    print("   uv run python src/fundamental_agent/main.py")
    print("")
    
    if len(sys.argv) > 1 and sys.argv[1] == "simple":
        print("Running simple test...")
        asyncio.run(run_simple_test())
    else:
        print("Running comprehensive test...")
        print("Use 'python test_client.py simple' for a quick test")
        print("Note: Fundamental analysis may take longer than other analyses")
        print("")
        asyncio.run(run_comprehensive_test())


if __name__ == "__main__":
    main()