import streamlit as st
import requests
import json
import time
from typing import Optional

# Configure Streamlit page with better styling
st.set_page_config(
    page_title="Docustory.in - AI PDF Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for ChatGPT-like styling
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Custom header styling */
    .header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 0;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Upload area styling */
    .upload-section {
        background: #f8f9fa;
        border: 2px dashed #dee2e6;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #667eea;
        background: #f0f2ff;
    }
    
    /* Better spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Chat container styling */
    .chat-container {
        background: white;
        border-radius: 15px;
        box-shadow: 0 2px 15px rgba(0,0,0,0.1);
        padding: 0;
        margin: 2rem 0;
        max-height: 600px;
        overflow-y: auto;
    }
    
    /* Message styling */
    .user-message {
        background: #667eea;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 1rem 1rem 4rem;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    .bot-message {
        background: #f8f9fa;
        color: #333;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 4rem 1rem 1rem;
        border: 1px solid #e9ecef;
    }
    
    /* Voice selection styling */
    .voice-selector {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    
    /* Success/Error styling */
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Audio player styling */
    .stAudio {
        margin: 1rem 0;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

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

def generate_audio_summary(session_id: str, summary_type: str, voice: str = "Alice") -> Optional[bytes]:
    """Generate audio summary of the document"""
    try:
        data = {
            "session_id": session_id,
            "summary_type": summary_type,
            "voice": voice
        }
        response = requests.post(f"{API_BASE_URL}/generate_audio_summary", json=data)
        return response
        
        if response.status_code == 200:
            # Check if it's debug response or actual audio
            content_type = response.headers.get('content-type', '')
            if 'audio' in content_type:
                return response.content  # Raw audio bytes
            else:
                # Debug/error response - show detailed info
                try:
                    debug_info = response.json()
                    if debug_info.get('audio_error'):
                        st.error(f"🚨 **Audio Error**: {debug_info.get('error_message')}")
                        st.error(f"**Error Type**: {debug_info.get('error_type')}")
                        st.info(f"**Session**: {debug_info.get('session_id')}")
                        st.info(f"**Request**: {debug_info.get('summary_type')} with {debug_info.get('voice')} voice")
                    else:
                        st.info(f"🔧 **Debug Response**: {debug_info}")
                    return None
                except:
                    st.error(f"Unexpected response format: {response.text}")
                    return None
        else:
            st.error(f"Audio generation failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Audio generation error: {str(e)}")
        return None

def main():
    # Custom header
    st.markdown("""
    <div class="header">
        <h1>🤖 Docustory.in</h1>
        <p>Your AI-Powered PDF Assistant with Voice Technology</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "document_name" not in st.session_state:
        st.session_state.document_name = None
    
    # Main layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Document upload section
        st.markdown("### 📄 Upload Your PDF")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a PDF document to chat with",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            if st.button("🚀 Process Document", type="primary", use_container_width=True):
                with st.spinner("🔄 Processing your PDF..."):
                    session_id = upload_document(uploaded_file)
                    if session_id:
                        st.session_state.session_id = session_id
                        st.session_state.document_name = uploaded_file.name
                        st.session_state.messages = []  # Clear previous messages
                        st.success(f"✅ Document processed successfully!")
                        st.rerun()
        
        # Show current document and listening options
        if st.session_state.document_name:
            st.markdown("---")
            st.markdown("### 📋 Current Document")
            st.info(f"📄 **{st.session_state.document_name}**")
            
            # STAR FEATURE: PDF Listening Options
            st.markdown("### 🎧 Listen to Your PDF")
            st.markdown("*Transform your document into audio:*")
            
            # Voice selection
            voice_gender = st.radio(
                "Select Voice:",
                ["👩 Alice (Female)", "👨 Bob (Male)"],
                key="voice_selection"
            )
            voice = "Alice" if "Alice" in voice_gender else "Bob"
            
            # Listening options with better styling
            st.markdown("**Choose listening option:**")
            
            if st.button("🎧 **Full PDF**", help="Complete document narration", use_container_width=True):
                with st.spinner(f"🎤 Generating full PDF audio with {voice} voice..."):
                    audio_bytes = generate_audio_summary(st.session_state.session_id, "full", voice)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/wav")
                        st.success("🎉 Full PDF audio is ready!")
            
            if st.button("⚡ **3-Minute Summary**", help="Key points in 3 minutes", use_container_width=True):
                with st.spinner(f"🎤 Generating 3-minute summary with {voice} voice..."):
                    audio_bytes = generate_audio_summary(st.session_state.session_id, "summary_3min", voice)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/wav")
                        st.success("🎉 3-minute summary is ready!")
            
            if st.button("📖 **6-Minute Deep Dive**", help="Comprehensive summary", use_container_width=True):
                with st.spinner(f"🎤 Generating 6-minute summary with {voice} voice..."):
                    audio_bytes = generate_audio_summary(st.session_state.session_id, "summary_6min", voice)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/wav")
                        st.success("🎉 6-minute summary is ready!")
            
            st.markdown("---")
            if st.button("🗑️ Clear Document", use_container_width=True):
                st.session_state.session_id = None
                st.session_state.document_name = None
                st.session_state.messages = []
                st.rerun()
    
    with col2:
        # Chat interface
        if st.session_state.session_id:
            st.markdown("### 💬 Chat with Your Document")
            
            # Chat messages container
            chat_container = st.container()
            
            with chat_container:
                # Display chat messages with better styling
                for message in st.session_state.messages:
                    if message["role"] == "user":
                        st.markdown(f"""
                        <div class="user-message">
                            <strong>You:</strong> {message["content"]}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="bot-message">
                            <strong>🤖 AI Assistant:</strong><br>
                            {message["content"]}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Chat input at the bottom
            st.markdown("---")
            prompt = st.chat_input("💭 Ask a question about your document...")
            
            if prompt:
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Get AI response
                with st.spinner("🤔 AI is thinking..."):
                    response = ask_question(st.session_state.session_id, prompt)
                    
                    if response and "answer" in response:
                        answer = response["answer"]
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                        # Show response details in expander
                        with st.expander("ℹ️ Response Details"):
                            st.json({
                                "Context Method": response.get("context_method"),
                                "Processing Time": f"{response.get('processing_time_seconds', 0):.2f}s",
                                "Model": response.get("model_used", "Qwen 2.5-14B")
                            })
                    else:
                        error_msg = "Sorry, I couldn't process your question. Please try again."
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
                st.rerun()
            
            # Voice input section
            st.markdown("---")
            st.markdown("### 🎤 Voice Questions")
            st.markdown("**Record your question and let Whisper transcribe it:**")
            
            try:
                audio_bytes = st.audio_input("🎙️ Record your question")
                
                if audio_bytes is not None:
                    st.audio(audio_bytes, format="audio/wav")
                    
                    if st.button("🎵 Process Voice Question", type="primary"):
                        st.session_state.messages.append({
                            "role": "user", 
                            "content": "🎤 Audio question"
                        })
                        
                        with st.spinner("🎤 Processing audio with Whisper + Qwen..."):
                            from io import BytesIO
                            audio_file = BytesIO(audio_bytes)
                            audio_file.name = "voice_question.wav"
                            
                            # Process voice question (you'll need to implement this)
                            st.success("🎉 Voice processing coming soon!")
                        
                        st.rerun()
            
            except Exception as e:
                st.info("💡 **Tip**: Voice input requires a compatible browser and microphone access.")
        
        else:
            # Welcome screen - simpler approach
            st.markdown("## 🚀 Get Started")
            st.markdown("Upload a PDF document to unlock the power of AI-driven document analysis")
            
            st.markdown("---")
            
            # Feature cards with columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **📄 Smart PDF Processing**  
                Advanced document parsing with AI-powered analysis
                
                **🎧 Audio Summaries**  
                Listen to your PDFs with high-quality voice synthesis
                """)
            
            with col2:
                st.markdown("""
                **🤖 AI Q&A System**  
                Ask questions and get intelligent answers from your documents
                
                **🎤 Voice Questions**  
                Ask questions using voice input with Whisper transcription
                """)
            
            # Show API status
            try:
                health_response = requests.get(f"{API_BASE_URL}/health")
                if health_response.status_code == 200:
                    st.success("✅ AI services are online and ready")
                else:
                    st.error("❌ AI services are not responding properly")
            except:
                st.error("❌ Cannot connect to AI services. Make sure the server is running on port 8000.")

if __name__ == "__main__":
    main()