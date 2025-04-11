import os
import sys
import numpy as np
from scipy.io import wavfile

def create_drowsiness_alert_sound(filename="alert_sound.wav", duration=1.0, frequency=440, volume=0.5):
    """
    Create a drowsiness alert sound (standard beep).
    
    :param filename: Output filename
    :param duration: Duration in seconds
    :param frequency: Tone frequency in Hz
    :param volume: Volume (0.0 to 1.0)
    """
    print(f"Creating drowsiness alert sound: {filename}")
    print(f"Duration: {duration} sec, Frequency: {frequency} Hz, Volume: {volume}")
    
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    beep = np.sin(2 * np.pi * frequency * t) * volume
    wavfile.write(filename, sample_rate, (beep * 32767).astype(np.int16))
    
    print(f"Drowsiness alert sound created successfully at: {os.path.abspath(filename)}")

def create_alcohol_alert_sound(filename="alcohol_alert.wav", duration=1.0, volume=0.5):
    """
    Create an alcohol alert sound (two-tone alarm).
    
    :param filename: Output filename
    :param duration: Duration in seconds
    :param volume: Volume (0.0 to 1.0)
    """
    print(f"Creating alcohol alert sound: {filename}")
    print(f"Duration: {duration} sec, Volume: {volume}")
    
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    note1 = np.sin(2 * np.pi * 880 * t[:len(t)//2]) * volume  # Higher pitch
    note2 = np.sin(2 * np.pi * 660 * t[len(t)//2:]) * volume  # Lower pitch
    alcohol_alarm = np.concatenate((note1, note2))
    wavfile.write(filename, sample_rate, (alcohol_alarm * 32767).astype(np.int16))
    
    print(f"Alcohol alert sound created successfully at: {os.path.abspath(filename)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "drowsiness":
            create_drowsiness_alert_sound()
        elif sys.argv[1] == "alcohol":
            create_alcohol_alert_sound()
        else:
            print("Invalid argument. Use 'drowsiness' or 'alcohol'.")
    else:
        create_drowsiness_alert_sound()
        create_alcohol_alert_sound()