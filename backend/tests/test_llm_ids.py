# import os
# import requests
# from dotenv import load_dotenv, find_dotenv

# # load_dotenv()
# load_dotenv(find_dotenv(), override=True)

# api_key = os.getenv("OPENROUTER_API_KEY")

# def test_model(model_id):
#     print(f"Testing model: {model_id}...")
#     try:
#         response = requests.post(
#             url="https://openrouter.ai/api/v1/chat/completions",
#             headers={
#                 "Authorization": f"Bearer {api_key}",
#                 "HTTP-Referer": "https://github.com/abhigupta/deep_research_agent",
#                 "X-Title": "Test Script",
#             },
#             json={
#                 "model": model_id,
#                 "messages": [
#                     {"role": "user", "content": "Hi"}
#                 ],
#                 "max_tokens": 10
#             }
#         )
#         if response.status_code == 200:
#             print(f"SUCCESS: {model_id} is working.")
#             return True
#         else:
#             print(f"FAILED: {model_id} returned {response.status_code} - {response.text}")
#             return False
#     except Exception as e:
#         print(f"ERROR: {e}")
#         return False

# # List of potential models to test
# models_to_test = [
#     "google/gemini-2.0-flash-001",
#     "anthropic/claude-3.5-haiku",
#     "anthropic/claude-3.5-sonnet",
# ]

# for model in models_to_test:
#     test_model(model)
