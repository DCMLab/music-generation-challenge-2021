import os

for f in os.listdir(b'.'):
    os.rename(f, f.decode('ISO-8859-1'))
