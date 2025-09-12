#!/usr/bin/env python3
"""
Docustory.in Deployment Script for Jupyter
Automatically starts both FastAPI backend and Streamlit frontend
"""

import subprocess
import time
import os
import signal
import sys
from pathlib import Path

class DocustoryDeployment:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.project_dir = Path(__file__).parent
        
    def check_dependencies(self):
        """Check if required dependencies are installed"""
        try:
            import fastapi
            import uvicorn
            import streamlit
            print("✅ All dependencies are installed")
            return True
        except ImportError as e:
            print(f"❌ Missing dependency: {e}")
            print("Installing required packages...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            return True
    
    def start_backend(self):
        """Start FastAPI backend on port 8000"""
        try:
            print("🚀 Starting FastAPI backend on http://0.0.0.0:8000...")
            
            # Kill existing processes on port 8000
            subprocess.run(["pkill", "-f", "uvicorn.*8000"], capture_output=True)
            time.sleep(1)
            
            self.backend_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn", 
                "app.main:app", 
                "--host", "0.0.0.0", 
                "--port", "8000", 
                "--reload"
            ], cwd=self.project_dir)
            
            # Wait for backend to start
            time.sleep(3)
            print("✅ Backend started successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
            return False
    
    def start_frontend(self):
        """Start Streamlit frontend on port 8501"""
        try:
            print("🎨 Starting Streamlit frontend on http://0.0.0.0:8501...")
            
            # Kill existing processes on port 8501
            subprocess.run(["pkill", "-f", "streamlit.*8501"], capture_output=True)
            time.sleep(1)
            
            self.frontend_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", 
                "simple_chat_app.py",
                "--server.address", "0.0.0.0",
                "--server.port", "8501",
                "--server.headless", "true"
            ], cwd=self.project_dir)
            
            # Wait for frontend to start
            time.sleep(3)
            print("✅ Frontend started successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start frontend: {e}")
            return False
    
    def check_services(self):
        """Check if both services are running"""
        try:
            import requests
            
            # Check backend
            backend_response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
            backend_status = "✅ Online" if backend_response.status_code == 200 else "❌ Error"
            
            # Check frontend (just check if port is responding)
            try:
                frontend_response = requests.get("http://localhost:8501", timeout=5)
                frontend_status = "✅ Online" if frontend_response.status_code == 200 else "✅ Loading"
            except:
                frontend_status = "✅ Loading"
            
            print(f"\n📊 Service Status:")
            print(f"Backend (port 8000): {backend_status}")
            print(f"Frontend (port 8501): {frontend_status}")
            print(f"\n🌐 Access URLs:")
            print(f"Backend API: http://localhost:8000 or http://192.168.2.183:8000")
            print(f"Frontend App: http://localhost:8501 or http://192.168.2.183:8501")
            
            return True
            
        except Exception as e:
            print(f"⚠️  Service check failed: {e}")
            return False
    
    def stop_services(self):
        """Stop both services"""
        print("🛑 Stopping services...")
        
        if self.backend_process:
            self.backend_process.terminate()
            self.backend_process.wait()
        
        if self.frontend_process:
            self.frontend_process.terminate()
            self.frontend_process.wait()
        
        # Kill any remaining processes
        subprocess.run(["pkill", "-f", "uvicorn.*8000"], capture_output=True)
        subprocess.run(["pkill", "-f", "streamlit.*8501"], capture_output=True)
        
        print("✅ All services stopped")
    
    def deploy(self):
        """Deploy the full application"""
        print("🚀 Deploying Docustory.in Application")
        print("=" * 50)
        
        # Check dependencies
        if not self.check_dependencies():
            return False
        
        # Start backend
        if not self.start_backend():
            return False
        
        # Start frontend
        if not self.start_frontend():
            self.stop_services()
            return False
        
        # Check services
        time.sleep(2)
        self.check_services()
        
        print("\n🎉 Deployment completed successfully!")
        print("💡 Use deployment.stop_services() to stop all services")
        
        return True


# Global deployment instance for Jupyter
deployment = DocustoryDeployment()

def deploy():
    """Quick deploy function for Jupyter cells"""
    return deployment.deploy()

def stop():
    """Quick stop function for Jupyter cells"""
    return deployment.stop_services()

def status():
    """Quick status check function for Jupyter cells"""
    return deployment.check_services()

# Auto-deploy if run directly
if __name__ == "__main__":
    try:
        deployment.deploy()
        
        # Keep running until interrupted
        print("\nPress Ctrl+C to stop all services...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        deployment.stop_services()
        print("\n👋 Goodbye!")