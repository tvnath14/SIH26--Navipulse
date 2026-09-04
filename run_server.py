"""
Quick-launch script for the SIH Intelligent Dead Reckoning Backend Server & Web Dashboard.
Usage:
    python run_server.py
Then open http://127.0.0.1:8000 in your browser.
"""

import uvicorn

if __name__ == "__main__":
    print("=" * 70)
    print(" STARTING SIH INTELLIGENT DEAD RECKONING SERVER")
    print(" Live Web Dashboard:  http://127.0.0.1:8000")
    print(" WebSocket Endpoint:  ws://127.0.0.1:8000/ws/telemetry")
    print(" API Documentation:   http://127.0.0.1:8000/docs")
    print("=" * 70)
    uvicorn.run("src.server.app:app", host="127.0.0.1", port=8000, reload=True, log_level="info")
