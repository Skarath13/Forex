import subprocess
import threading
import time

def run_bot():
    subprocess.run([
        "/Users/dylan/Desktop/algo/pyfxtrader_simplema/venv/bin/python",
        "/Users/dylan/Desktop/algo/pyfxtrader_simplema/main.py"
    ])

# Start bot in background thread
bot_thread = threading.Thread(target=run_bot)
bot_thread.daemon = True
bot_thread.start()

# Let it run for 15 seconds
time.sleep(15)
print("\n\nStopping bot after sample run...")