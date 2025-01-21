import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import scipy

def lo_tracking():
    RF = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    LO = [1030, 1050, 1070, 1110, 1150, 1200, 1270, 1370, 1470, 1610, 1820]
    LC = [527.27, 554.4, 587.4, 632.1, 680.1, 740.0, 827.8, 962.8, 1120, 1361, 1618]

    plt.plot(RF, LO, marker='o', linestyle='-', color="red", label='LO')
    plt.plot(RF, LC, marker='o', linestyle='-', color="orange", label='LC')
    plt.plot(RF, (np.array(LO) - np.array(LC)), marker='o', linestyle='-', color="green", label='LO-LC')

    plt.xlabel("RF Tuning Setting")
    plt.ylabel("Frequency(kHz)")

    plt.legend()
    plt.grid()
    plt.show()

def if_filter(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8')

    freq = df[" Frequency (Hz)"] / 1000
    gain = df[" Gain (dB)"]
    phase = df[" Phase (�)"]

    print(type(freq))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(freq, gain, label='Gain', linestyle='-', marker='o', markersize=2, color='blue')

    ax2 = ax1.twinx()

    ax2.plot(freq, phase, label='Phase', linestyle='-', marker='o', markersize=2, color='red')

    ax1.set_xlabel("Frequency (kHz)")
    ax1.set_ylabel("Gain(dB)")
    ax2.set_ylabel("Phase(degree)")

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()

def am_demod():
    amp = [0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.7, 3.8, 3.9, 4.0]
    dc = [185, 154, 103, 50.3, 35.4, 96.1, 155, 215, 275, 355, 396, 456, 515, 576, 636, 696, 758, 818, 847, 879, 908, 939]

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(amp, dc)
    lin_reg = np.array(amp) * slope + intercept

    plt.plot(amp, dc, marker='o', linestyle='-', color="darkblue", label='Measured DC RMS output')
    plt.plot(amp, lin_reg, linestyle='--', color="red", label=f'Linear approximation y={slope:.3g}x+{intercept:.3g}')

    plt.xlabel(r"Amplitude of Generated AC Signaln (mV$_{pp}$)")
    plt.ylabel("DC RMS Output (mV)")

    plt.legend()
    plt.grid()
    plt.show()
    
if __name__ == "__main__":
    csv = "data/scope_2.csv"
    if_filter(csv)