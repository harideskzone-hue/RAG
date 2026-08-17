import subprocess

p = subprocess.Popen(['.venv/bin/python3', 'scripts/chat.py', '--mode', 'investigation'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
out, err = p.communicate(input='Find the person in the green outfit.\nexit\n')
print(out)
print(err)
