import subprocess
import sys
import io

# Run the bot and capture output
process = subprocess.Popen(
    [sys.executable, "main.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,
    cwd="/Users/dylan/Desktop/algo/pyfxtrader_simplema"
)

# Read output for 20 seconds
import time
start_time = time.time()
output_lines = []

while time.time() - start_time < 20:
    line = process.stdout.readline()
    if line:
        print(line.strip())
        output_lines.append(line)
    
    # Check if process has terminated
    if process.poll() is not None:
        break

# Terminate the process
process.terminate()
process.wait()

# Print any remaining output
remaining, _ = process.communicate()
if remaining:
    print(remaining)