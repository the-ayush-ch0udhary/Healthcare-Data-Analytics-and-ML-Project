"""
Hospital Patient Analytics - Quick Application Launcher
Runs the Streamlit application with appropriate configuration.
"""

import subprocess
import sys
from pathlib import Path


def main():
    base_dir = Path(__file__).resolve().parent
    app_file = base_dir / "streamlit_app" / "app.py"

    print("=" * 60)
    print("🏥 Launching Hospital Patient Analytics Platform")
    print(f"📁 Target script: {app_file}")
    print("=" * 60)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_file),
        "--browser.gatherUsageStats",
        "false",
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[INFO] Application stopped by user.")
    except Exception as e:
        print(f"\n[ERROR] Failed to launch application: {e}")


if __name__ == "__main__":
    main()
