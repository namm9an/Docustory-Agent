#!/usr/bin/env python3
"""
Test script for E2E Networks Qwen integration.
Run this to verify the integration is working correctly.
"""

import asyncio
import os
from openai import AsyncOpenAI

# E2E Networks configuration
token = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJGSjg2R2NGM2pUYk5MT2NvNE52WmtVQ0lVbWZZQ3FvcXRPUWVNZmJoTmxFIn0.eyJleHAiOjE3ODcwNTIxNDksImlhdCI6MTc1NTUxNjE0OSwianRpIjoiZjExNWFlNmYtZWYxNi00YmQ4LWFhMTEtZjg3YzY3NjUyNTU0IiwiaXNzIjoiaHR0cDovL2dhdGV3YXkuZTJlbmV0d29ya3MuY29tL2F1dGgvcmVhbG1zL2FwaW1hbiIsImF1ZCI6ImFjY291bnQiLCJzdWIiOiJjOWUzZTQ5OS04ZjQzLTQ1MDItYjY0My03NTRiZmM5YjhlMTkiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJhcGltYW51aSIsInNlc3Npb25fc3RhdGUiOiI2MTRkYWU1Yy1lMjNiLTRmZjUtYWVmZS1kYjk3MzA5NzBmNjQiLCJhY3IiOiIxIiwiYWxsb3dlZC1vcmlnaW5zIjpbIiJdLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImFwaXVzZXIiLCJkZWZhdWx0LXJvbGVzLWFwaW1hbiJdfSwicmVzb3VyY2VfYWNjZXNzIjp7ImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoicHJvZmlsZSBlbWFpbCIsInNpZCI6IjYxNGRhZTVjLWUyM2ItNGZmNS1hZWZlLWRiOTczMDk3MGY2NCIsImVtYWlsX3ZlcmlmaWVkIjpmYWxzZSwiaXNfcGFydG5lcl9yb2xlIjpmYWxzZSwibmFtZSI6Ik5hbWFuIE1vdWRnaWxsIiwicHJpbWFyeV9lbWFpbCI6Im5hbWFuLm1vdWRnaWxsQGUyZW5ldHdvcmtzLmNvbSIsImlzX3ByaW1hcnlfY29udGFjdCI6dHJ1ZSwicHJlZmVycmVkX3VzZXJuYW1lIjoibmFtYW4ubW91ZGdpbGxAZTJlbmV0d29ya3MuY29tIiwiZ2l2ZW5fbmFtZSI6Ik5hbWFuIiwiZmFtaWx5X25hbWUiOiJNb3VkZ2lsbCIsImVtYWlsIjoibmFtYW4ubW91ZGdpbGxAZTJlbmV0d29ya3MuY29tIiwiaXNfaW5kaWFhaV91c2VyIjpmYWxzZX0.OCuD0RF4wKiWx-4sjsyCqGPmQB-pZcJ_UpP41hIxrzl6u4LmKhsBan6WNiDRDkLYI5Y1cULLieqiepTv5b58jy83w6JloSZV4AtkSRYm8i6q32FEqqPiL47f5clMSJIFv9q0cp_UhjiDncJEoPzy8AQdOYrPUzM44lKWwnZ8UAE"
base_url = "https://infer.e2enetworks.net/project/p-6530/endpoint/is-6479/v1/"
model = "Qwen/Qwen2.5-14B-Instruct"


async def test_basic_chat():
    """Test basic chat completion."""
    print("🧪 Testing basic chat completion...")
    
    client = AsyncOpenAI(
        api_key=token,
        base_url=base_url
    )
    
    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Hello! Can you confirm that you are working correctly? Please respond with a brief confirmation."}
            ],
            max_tokens=100,
            temperature=0.7
        )
        
        response = completion.choices[0].message.content
        print(f"✅ Basic chat test successful!")
        print(f"📝 Response: {response}")
        print(f"🔢 Tokens used: {completion.usage.total_tokens if completion.usage else 'N/A'}")
        return True
        
    except Exception as e:
        print(f"❌ Basic chat test failed: {e}")
        return False


async def test_document_qa():
    """Test document Q&A functionality."""
    print("\n🧪 Testing document Q&A...")
    
    client = AsyncOpenAI(
        api_key=token,
        base_url=base_url
    )
    
    # Sample document context
    document_context = """
    This is a technical report on artificial intelligence developments in 2024. 
    The report covers advances in large language models, computer vision, and robotics.
    Key findings include improved efficiency in model training and better performance on reasoning tasks.
    The document also discusses ethical considerations and the impact on various industries.
    """
    
    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a document assistant. Use the provided context to answer questions accurately."
                },
                {
                    "role": "user", 
                    "content": f"Document Context:\n{document_context}\n\nBased on this document, what are the main topics covered?"
                }
            ],
            max_tokens=200,
            temperature=0.3
        )
        
        response = completion.choices[0].message.content
        print(f"✅ Document Q&A test successful!")
        print(f"📝 Response: {response}")
        print(f"🔢 Tokens used: {completion.usage.total_tokens if completion.usage else 'N/A'}")
        return True
        
    except Exception as e:
        print(f"❌ Document Q&A test failed: {e}")
        return False


async def test_streaming():
    """Test streaming response."""
    print("\n🧪 Testing streaming response...")
    
    client = AsyncOpenAI(
        api_key=token,
        base_url=base_url
    )
    
    try:
        print("📡 Streaming response:")
        
        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Explain the concept of machine learning in simple terms. Keep it brief."}
            ],
            max_tokens=150,
            temperature=0.7,
            stream=True
        )
        
        full_response = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        
        print(f"\n✅ Streaming test successful!")
        print(f"📊 Full response length: {len(full_response)} characters")
        return True
        
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        return False


async def run_all_tests():
    """Run all tests."""
    print("🚀 Starting E2E Networks Qwen integration tests...\n")
    
    tests = [
        ("Basic Chat", test_basic_chat),
        ("Document Q&A", test_document_qa),
        ("Streaming", test_streaming)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print(f"\n📊 Test Results Summary:")
    print("=" * 30)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print("=" * 30)
    print(f"Tests passed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("🎉 All tests passed! E2E Networks integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the configuration and API key.")


if __name__ == "__main__":
    asyncio.run(run_all_tests())