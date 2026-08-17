import sys
import subprocess

p = subprocess.Popen([sys.executable, 'scripts/chat.py', '--mode', 'investigation'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
out, err = p.communicate(input='Find the person in the green outfit.\nexit\n')
print(out)
print(err)
