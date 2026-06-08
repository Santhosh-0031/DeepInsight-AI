import asyncio
import os
from dotenv import load_dotenv
from backend.app.nodes import query_analyzer_hyde
from backend.app.state import ReportState

# Load environment variables
env_path = os.path.join(os.getcwd(), ".env")
load_dotenv(dotenv_path=env_path)

async def verify_accuracy():
    print("--- Verifying Search Accuracy for 'Emergent AI Company' ---")
    
    state = {
        "topic": "Emergent AI Company",
        "depth": "quick"
    }
    
    # Run the query analyzer node
    result = await query_analyzer_hyde(state)
    
    hyde_doc = result.get("hyde_document", "")
    print("\nGenerated HyDE Document:")
    print("-" * 50)
    print(hyde_doc)
    print("-" * 50)
    
    # Check if the HyDE document mentions business/company aspects
    keywords = ["business", "company", "startup", "market", "product", "competitor", "strategy", "investment"]
    found_keywords = [k for k in keywords if k in hyde_doc.lower()]
    
    print(f"\nFound business-related keywords: {found_keywords}")
    
    if len(found_keywords) >= 2:
        print("\nSUCCESS: The HyDE document correctly identifies the business/entity context.")
    else:
        print("\nFAILURE: The HyDE document still seems too theoretical.")

if __name__ == "__main__":
    asyncio.run(verify_accuracy())
