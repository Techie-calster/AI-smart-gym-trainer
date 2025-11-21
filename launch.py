import os
import time
import sys
import subprocess
import webbrowser
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler

# --- CONFIGURATION ---
STREAMLIT_PORT = 8501
HTML_PORT = 8000
HTML_FILENAME = "home.html"
WEBSITE_FOLDER = "website"  # The folder where home.html lives

# Get absolute paths to ensure code works regardless of where we launch it
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
STREAMLIT_SCRIPT = os.path.join(ROOT_DIR, "app.py")
WEBSITE_DIR = os.path.join(ROOT_DIR, WEBSITE_FOLDER)

def run_streamlit():
    """Runs the Streamlit app using the absolute path."""
    print(f"🚀 Starting Streamlit App on port {STREAMLIT_PORT}...")
    
    # We use the absolute path to app.py so it doesn't get lost when we change dirs later
    if not os.path.exists(STREAMLIT_SCRIPT):
        print(f"❌ Error: Could not find {STREAMLIT_SCRIPT}")
        return

    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        STREAMLIT_SCRIPT, 
        "--server.port", str(STREAMLIT_PORT), 
        "--server.headless", "true"
    ]
    subprocess.run(cmd)

def run_html_server():
    """Serves the 'website' folder over HTTP."""
    print(f"🌍 Serving HTML from folder: {WEBSITE_FOLDER}...")
    
    if not os.path.exists(WEBSITE_DIR):
        print(f"❌ Error: '{WEBSITE_FOLDER}' folder not found!")
        return

    # Change directory to 'website' so images inside home.html load correctly
    os.chdir(WEBSITE_DIR)
    
    server_address = ('', HTML_PORT)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    print(f"✅ Website running at: http://localhost:{HTML_PORT}/{HTML_FILENAME}")
    httpd.serve_forever()

if __name__ == "__main__":
    # 1. Start Streamlit in a background thread
    streamlit_thread = threading.Thread(target=run_streamlit)
    streamlit_thread.daemon = True
    streamlit_thread.start()

    # 2. Start HTML Server in a background thread
    html_thread = threading.Thread(target=run_html_server)
    html_thread.daemon = True
    html_thread.start()

    # 3. Wait for servers to initialize
    print("⏳ Waiting for services to spin up...")
    time.sleep(3)

    # 4. Open the Browser
    url = f"http://localhost:{HTML_PORT}/{HTML_FILENAME}"
    print(f"🔗 Opening {url}...")
    webbrowser.open(url)

    # 5. Keep script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down PostuRight...")