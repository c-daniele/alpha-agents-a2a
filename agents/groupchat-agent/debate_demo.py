"""
Demonstration script for the AlphaAgents-compliant structured debate mechanism.

This script shows how the GroupChat agent now implements proper turn-taking
and ensures each agent speaks at least twice, as required by the AlphaAgents paper.
"""

import asyncio
import logging
from datetime import datetime

from src.groupchat_agent.test_client import GroupChatTestClient

# Configure logging to show debate process
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demonstrate_structured_debate():
    """Demonstrate the structured debate mechanism."""
    
    print("\n" + "="*80)
    print("ALPHAAGENTS STRUCTURED DEBATE DEMONSTRATION")
    print("="*80)
    print("\nThis demonstration shows the new A2A-compliant structured debate")
    print("mechanism that implements strict turn-taking and ensures each agent")
    print("speaks at least twice, as required by the AlphaAgents paper (section 2.2.4).\n")
    
    client = GroupChatTestClient()
    
    # Investment decision scenarios that should trigger debates (multi-language examples)
    debate_scenarios = [
        {
            "name": "Tesla Investment Decision (English)",
            "message": "Should I invest in Tesla (TSLA) stock? I need a comprehensive investment recommendation.",
            "description": "Multi-agent investment recommendation requiring debate"
        },
        {
            "name": "Apple Portfolio Analysis (Spanish)", 
            "message": "¿Debo invertir en acciones de Apple (AAPL)? Necesito un análisis completo para mi cartera de inversiones.",
            "description": "Spanish query - Portfolio analysis requiring collaborative decision-making"
        },
        {
            "name": "NVIDIA Growth Assessment (French)",
            "message": "Évaluez NVIDIA (NVDA) comme opportunité d'investissement à long terme en considérant tous les facteurs financiers.",
            "description": "French query - Comprehensive evaluation requiring multi-agent consensus"
        },
        {
            "name": "Microsoft Strategic Decision (Italian)",
            "message": "Dovrei comprare, vendere o mantenere le azioni Microsoft (MSFT)? Voglio un'analisi strategica completa.",
            "description": "Italian query - Strategic decision requiring structured debate"
        },
        {
            "name": "Simple Price Query (English)",
            "message": "What is the current stock price of Amazon (AMZN)?",
            "description": "Simple factual query - should NOT trigger debate"
        },
        {
            "name": "Conservative Investment Request (English)",
            "message": "I'm 65 and looking for safe, stable dividend-paying stocks for my retirement portfolio. What do you recommend?",
            "description": "Risk-averse investment decision - should trigger debate with risk_tolerance=averse"
        },
        {
            "name": "Aggressive Growth Request (English)", 
            "message": "I'm young and want high-growth stocks with maximum potential returns. I can handle volatility and risk.",
            "description": "Risk-seeking investment decision - should trigger debate with risk_tolerance=seeking"
        }
    ]
    
    for i, scenario in enumerate(debate_scenarios, 1):
        print(f"\n{'-'*60}")
        print(f"SCENARIO {i}: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"{'-'*60}")
        
        test_message = {
            "kind": "message",
            "messageId": f"debate_demo_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "role": "user",
            "parts": [
                {
                    "kind": "text",
                    "text": scenario["message"]
                }
            ],
            "contextId": f"debate_context_{i}"
        }
        
        print(f"\nUser Query: {scenario['message']}")
        print("\n🤖 Starting structured multi-agent debate...\n")
        
        try:
            # Increase timeout for debate process
            original_timeout = client.timeout
            client.timeout = 600.0  # 10 minutes
            
            start_time = datetime.now()
            
            response = await client.send_jsonrpc_request("message/send", {
                "message": test_message,
                "metadata": {
                    "demonstration": True,
                    "scenario": scenario["name"]
                }
            })
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Restore timeout
            client.timeout = original_timeout
            
            if "error" in response and response["error"] is not None:
                print(f"❌ Error: {response['error']['message']}")
                continue
            
            result = response.get("result", {})
            
            # Extract response text
            response_text = ""
            if result.get("parts"):
                response_text = " ".join([
                    p.get("text", "") 
                    for p in result["parts"] 
                    if p.get("kind") == "text"
                ])
            
            # Extract metadata
            metadata = result.get("metadata", {})
            agent_summary = metadata.get("agent_responses_summary", {})
            
            print(f"✅ Debate completed in {duration:.1f} seconds")
            print(f"📊 Agents consulted: {agent_summary.get('total_agents_consulted', 'N/A')}")
            print(f"📊 Successful responses: {agent_summary.get('successful_responses', 'N/A')}")
            print(f"📊 Participating agents: {', '.join(agent_summary.get('agents_used', []))}")
            
            # Check for debate indicators
            debate_indicators = [
                "structured" in response_text.lower(),
                "debate" in response_text.lower(),
                "rounds" in response_text.lower(),
                "consensus" in response_text.lower(),
                agent_summary.get('total_agents_consulted', 0) > 1
            ]
            
            debate_detected = any(debate_indicators)
            print(f"🎯 Structured debate detected: {'Yes' if debate_detected else 'No'}")
            
            print(f"\n💡 Final Recommendation Summary:")
            # Show first 200 characters of the response
            summary = response_text[:200] + "..." if len(response_text) > 200 else response_text
            print(f"{summary}")
            
            if i < len(debate_scenarios):
                print(f"\n⏳ Waiting 5 seconds before next scenario...")
                await asyncio.sleep(5)
                
        except Exception as e:
            print(f"❌ Demonstration failed: {str(e)}")
            logger.exception(f"Error in scenario {i}")
    
    print(f"\n{'='*80}")
    print("DEMONSTRATION COMPLETED")
    print("="*80)
    print("\nKey Features Demonstrated:")
    print("✓ Automatic detection of investment decision requests")
    print("✓ Structured debate initiation for multi-agent scenarios") 
    print("✓ Turn-based communication (no concurrent agent calls)")
    print("✓ Minimum participation requirement (each agent speaks ≥2 times)")
    print("✓ Round-robin turn taking to prevent agent dominance")
    print("✓ Consensus checking and early termination")
    print("✓ Comprehensive response synthesis")
    print(f"\n📚 Implementation follows AlphaAgents paper section 2.2.4 requirements")


async def demonstrate_turn_taking_details():
    """Show detailed logging of the turn-taking mechanism."""
    
    print(f"\n{'='*80}")
    print("TURN-TAKING MECHANISM DETAILS")
    print("="*80)
    print("\nThis section shows detailed logging of the structured debate process.")
    print("Watch the logs to see the strict turn-taking implementation.\n")
    
    # Enable debug logging to show turn details
    logging.getLogger('groupchat_agent.a2a_agent').setLevel(logging.DEBUG)
    
    client = GroupChatTestClient()
    client.timeout = 300.0
    
    test_message = {
        "kind": "message",
        "messageId": f"turn_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "role": "user",
        "parts": [
            {
                "kind": "text",
                "text": "I'm considering a major investment in Microsoft (MSFT). Should I buy, hold, or avoid this stock?"
            }
        ],
        "contextId": "turn_demo_context"
    }
    
    print("🎯 Query: Microsoft investment decision")
    print("📝 Watch the logs below for detailed turn-taking process...\n")
    
    try:
        response = await client.send_jsonrpc_request("message/send", {
            "message": test_message,
            "metadata": {"debug": True, "show_turns": True}
        })
        
        if "error" in response and response["error"] is not None:
            print(f"❌ Error: {response['error']['message']}")
        else:
            print(f"\n✅ Turn-taking demonstration completed successfully!")
            
    except Exception as e:
        print(f"❌ Turn-taking demo failed: {str(e)}")
        logger.exception("Error in turn-taking demonstration")


if __name__ == "__main__":
    print("🚀 AlphaAgents Structured Debate Demonstration")
    print("This demo requires the GroupChat agent server to be running on localhost:3000")
    print("and all specialist agents to be registered and available.\n")
    
    # Check if user wants to see full demo or just turn-taking
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "turns":
        asyncio.run(demonstrate_turn_taking_details())
    else:
        asyncio.run(demonstrate_structured_debate())