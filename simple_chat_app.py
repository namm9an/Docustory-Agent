import streamlit as st
import requests
import json
import time
from typing import Optional

# Configure Streamlit page
st.set_page_config(
    page_title="Docustory.in - Chat with PDFs",
    page_icon="📚",
    layout="wide"
)

# API Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

def upload_document(file) -> Optional[str]:
    """Upload document and get session ID"""
    try:
        files = {"file": (file.name, file, file.type)}
        response = requests.post(f"{API_BASE_URL}/upload_pdf", files=files)
        
        if response.status_code == 200:
            result = response.json()
            return result.get("session_id")
        else:
            st.error(f"Upload failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Upload error: {str(e)}")
        return None

def ask_question(session_id: str, question: str) -> Optional[dict]:
    """Ask a question about the document"""
    try:
        data = {
            "session_id": session_id,
            "query": question,
            "conversation": True,
            "voice_enabled": False
        }
        response = requests.post(f"{API_BASE_URL}/ask", json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Question failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Question error: {str(e)}")
        return None

def ask_question_voice(session_id: str, audio_file) -> Optional[dict]:
    """Ask a question using voice input"""
    try:
        files = {"voice_file": (audio_file.name, audio_file, audio_file.type)}
        data = {
            "session_id": session_id,
            "voice_enabled": True,
            "transcription_language": "auto"
        }
        response = requests.post(f"{API_BASE_URL}/ask_voice", files=files, data=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Voice question failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Voice question error: {str(e)}")
        return None

def main():
    st.title("📚 Docustory.in - Chat with Your PDFs")
    st.markdown("Upload a PDF document and ask questions about it!")
    
    # Initialize session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "document_name" not in st.session_state:
        st.session_state.document_name = None
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📄 Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a PDF document to chat with"
        )
        
        if uploaded_file is not None:
            if st.button("Upload Document", type="primary"):
                with st.spinner("Uploading and processing document..."):
                    session_id = upload_document(uploaded_file)
                    if session_id:
                        st.session_state.session_id = session_id
                        st.session_state.document_name = uploaded_file.name
                        st.session_state.messages = []  # Clear previous messages
                        st.success(f"Document '{uploaded_file.name}' uploaded successfully!")
                        st.rerun()
        
        # Show current document
        if st.session_state.document_name:
            st.markdown("---")
            st.markdown("**Current Document:**")
            st.info(f"📄 {st.session_state.document_name}")
            st.markdown(f"**Session ID:** `{st.session_state.session_id}`")
            
            if st.button("Clear Document"):
                st.session_state.session_id = None
                st.session_state.document_name = None
                st.session_state.messages = []
                st.rerun()
    
    # Main chat interface
    if st.session_state.session_id:
        st.markdown("### 💬 Chat with your document")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your document..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = ask_question(st.session_state.session_id, prompt)
                    
                    if response and "answer" in response:
                        answer = response["answer"]
                        st.markdown(answer)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                        # Show additional info if available
                        if "context_method" in response:
                            with st.expander("ℹ️ Response Details"):
                                st.json({
                                    "Context Method": response.get("context_method"),
                                    "Processing Time": f"{response.get('processing_time_seconds', 0):.2f}s",
                                    "Model": response.get("model_used", "Qwen 2.5-14B")
                                })
                    else:
                        error_msg = "Sorry, I couldn't process your question. Please try again."
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Voice input section using Whisper
        st.markdown("---")
        st.markdown("### 🎤 Ask with Voice (Whisper + Qwen)")
        
        st.markdown("**Record your question as audio and let Whisper transcribe it:**")
        
        try:
            # Streamlit audio input (requires streamlit >= 1.28.0)
            audio_bytes = st.audio_input("Record your question")
            
            if audio_bytes is not None:
                st.audio(audio_bytes, format="audio/wav")
                
                if st.button("🎵 Process Audio Question", type="primary"):
                    with st.chat_message("user"):
                        st.markdown("🎤 Audio question submitted")
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": "🎤 Audio question"
                    })
                    
                    with st.chat_message("assistant"):
                        with st.spinner("🎤 Processing audio with Whisper + Qwen..."):
                            # Create a temporary file-like object
                            from io import BytesIO
                            audio_file = BytesIO(audio_bytes)
                            audio_file.name = "voice_question.wav"
                            
                            response = ask_question_voice(st.session_state.session_id, audio_file)
                            
                            if response and "answer" in response:
                                answer = response["answer"]
                                st.markdown(answer)
                                
                                # Add assistant response to chat history
                                st.session_state.messages.append({"role": "assistant", "content": answer})
                                
                                # Show transcription and details
                                if response.get("metadata", {}).get("transcription"):
                                    trans = response["metadata"]["transcription"]
                                    with st.expander("🎤 Audio Processing Details"):
                                        st.markdown(f"**Transcribed:** {trans['transcribed_text']}")
                                        st.json({
                                            "Confidence": f"{trans.get('confidence', 0):.2%}",
                                            "Language": trans.get('detected_language', 'auto'),
                                            "Processing Time": f"{response.get('processing_time_seconds', 0):.2f}s"
                                        })
                            else:
                                error_msg = "Sorry, I couldn't process your audio. Please try again."
                                st.error(error_msg)
                                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
                    st.rerun()
        
        except Exception as e:
            st.info("💡 **Tip**: For voice input, you can type your question in the text box above, or upgrade Streamlit for audio recording features.")
    
    else:
        # Welcome screen
        st.markdown("""
        ### 🚀 Get Started
        
        1. **Upload a PDF** using the sidebar
        2. **Ask questions** about your document (text or voice)
        3. **Get AI-powered answers** from Qwen 2.5-14B
        
        ### ✨ Features
        - 📄 PDF document processing
        - 🤖 AI-powered Q&A with Qwen 2.5-14B
        - 🎤 **Voice input support** with Whisper transcription
        - 🔍 Smart keyword extraction with YAKE
        - 💬 Multi-turn conversation memory
        - 🎯 Context-aware responses
        """)
        
        # Show API status
        try:
            health_response = requests.get(f"{API_BASE_URL}/health")
            if health_response.status_code == 200:
                st.success("✅ API is running and healthy")
            else:
                st.error("❌ API is not responding properly")
        except:
            st.error("❌ Cannot connect to API. Make sure the server is running on port 8000.")

if __name__ == "__main__":
    main()