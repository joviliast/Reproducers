import sys
import subprocess
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
helper_path = os.path.join(dir_path, "helper.py")
print("1st proc")
proc = subprocess.run(
    [sys.executable, helper_path],
    capture_output=True,
)

print("2md proc")
proc = subprocess.run(
    [sys.executable, helper_path],
    capture_output=True,
)

print("3rd proc")
proc = subprocess.run(
    [sys.executable, helper_path],
    capture_output=True,
)
