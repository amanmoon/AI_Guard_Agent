import eel
from pyfiles.ASR.captureAudio import *

eel.init('build')
print("Starting UI... Close the window to exit the application.")
eel.start('index.html', size=(800, 600), mode='chrome-app')