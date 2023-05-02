import json

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

# START OF THE EXAMPLES
A = np.array([[1, 2], [4, 5]])
def example1(A,i=int,j=int):
    """2.27The element in row i and column j of A is denoted by A(i,j). For example, A(2,3) is the element from the second row and third column of matrix A"""
    return A[i,j]
print("Example 1 output: ",example1(A,1,0))

B = np.array([1, 2, 4, 5, 6, 7, 8, 9, 5, 3])
def aperiodic_to_periodic(a):
    """3.4 Let’s say we have a sequence of 10 terms and we want to build a signal with 200 samples repeating the initial sequence.If we want to extend through periodicity this signal, we will write the following code and we
will obtain the graphic plotted."""
    plt.stem(a)
    plt.axis([0, len(a)+1, min(a)-1, max(a)+1])
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Discrete aperiodic sequence')

    a_periodic = np.ones((10, 1)) * a
    a_periodic = a_periodic.T.reshape((len(a) * 10, 1))
    plt.figure()
    plt.stem(a_periodic)
    plt.axis([0, len(a_periodic)+1, min(a_periodic)-1, max(a_periodic)+1])
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Discrete periodic sequence')
    
    plt.show()
aperiodic_to_periodic(B)

def saveVar(varT):
    """3.12 Saves the given argument to a JSON file"""
    out = {'varT':varT}
    with open('varT.json','w') as output:
        output.write(json.dumps(out))
saveVar(25)   

def dftmtx(N):
    """4.1 The function outputs the Discrete Fourier Transform Matrix - equivalent of the function dftmtx in Matlab.The function first creates two vectors n and k using the arange and reshape functions, respectively. It then calculates the DFT matrix using the formula and returns it normalized by the square root of N."""
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    # it also works with
    # return M / np.sqrt(N)
    return np.dot(M,N)
print("Example 3 output: ",dftmtx(4))

def butter_lp_filter(fs, fCut, fTrans, Rp, Rs):
    """6.15 We will design a LP Butterworth filter working at 10KHz, which will have the
separation frequency at 2KHz and the transition band of 400Hz. The filter will exhibit a ripple of
1dB in the passing band and an attenuation of 70dB in the stop band."""
    # Calculate the cutting and transition frequencies
    fPass = fCut - fTrans/2
    fStop = fCut + fTrans/2
    
    # Normalize the frequencies
    Wp = fPass/(fs/2)
    Ws = fStop/(fs/2)
    
    # Calculate the order and cutoff frequency of the Butterworth filter
    order, Wc = signal.buttord(Wp, Ws, Rp, Rs, analog=False)
    
    # Design the filter
    b, a = signal.butter(order, Wc, btype='lowpass', analog=False, output='ba')
    
    # Plot the frequency and phase response of the filter
    f, h = signal.freqz(b, a)
    w = f / (2 * np.pi)
    mag = 20 * np.log10(abs(h))
    phase = np.angle(h) * 180 / np.pi
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(w, mag, 'b')
    plt.ylabel('Magnitude [dB]')
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(w, phase, 'g')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase [deg]')
    plt.grid()
    plt.show()
    
    return b, a, order

fs = 10_000
fCut = 2000
fTrans = 400
Rp = 1
Rs = 70
b, a, order = butter_lp_filter(fs, fCut, fTrans, Rp, Rs)

# START OF THE EXERCISES
# ex 2.26
def nonprime_while():
    num = 1001
    while True:
        prime = True
        for i in range(2, int(num/2) + 1):
            if num % i == 0:
                prime = False
                break
        if prime:
            num += 1
            continue
        print(num)
        num += 1
        if num % 2 == 0:
            num += 1
        prime = True
        for i in range(2, int(num/2) + 1):
            if num % i == 0:
                prime = False
                break
        if prime:
            break
def nonprime_for():
    num = 1001
    for i in range(num, num + 1000):
        prime = True
        for j in range(2, int(i/2) + 1):
            if i % j == 0:
                prime = False
                break
        if prime:
            continue
        print(i)
        if i % 2 == 0:
            i += 1
        prime = True
        for j in range(2, int(i/2) + 1):
            if i % j == 0:
                prime = False
                break
        if prime:
            break
def nonprime_flag():
    num = 1001
    prime = False
    while not prime:
        is_prime = True
        for i in range(2, int(num/2) + 1):
            if num % i == 0:
                is_prime = False
                break
        if not is_prime:
            print(num)
            if num % 2 == 0:
                num += 1
            for i in range(num + 1, num + 1000):
                prime = True
                for j in range(2, int(i/2) + 1):
                    if i % j == 0:
                        prime = False
                        break
                if prime:
                    num = i
                    break
            if not prime:
                if num % 2 == 0:
                    num += 1
        else:
            num += 1

# ex 3.11
def signal_symmetry(even_sig, odd_sig):
    """Show that if we denote by O an odd signal and with E an even signal, we have:
Symmetry after arithmetical operation:\n
First signal | Operation | Second signal | Result
\nO + O = O
\nE + E = E
\nO * O = E
\nE * E = E
\nO * E = O\n
Check this graphically, taking the element by element operations for two signals – one Odd
(can be a sine) and one Even (can be a cosine)"""
    # addition of two odd signals results in an odd signal
    o1 = odd_sig + odd_sig
    # addition of two even signals results in an even signal
    e1 = even_sig + even_sig
    # multiplication of two odd signals results in an even signal
    e2 = odd_sig * odd_sig
    # multiplication of two even signals results in an even signal
    e3 = even_sig * even_sig
    # multiplication of an odd and an even signal results in an odd signal
    o2 = odd_sig * even_sig
    
    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    
    axs[0, 0].plot(odd_sig, label='Odd signal')
    axs[0, 0].set_title('Odd signal')
    axs[0, 0].legend()
    
    axs[0, 1].plot(o1, label='Odd + Odd')
    axs[0, 1].set_title('Odd + Odd = Odd')
    axs[0, 1].legend()
    
    axs[0, 2].plot(e1, label='Even + Even')
    axs[0, 2].set_title('Even + Even = Even')
    axs[0, 2].legend()
    
    axs[1, 0].plot(even_sig, label='Even signal')
    axs[1, 0].set_title('Even signal')
    axs[1, 0].legend()
    
    axs[1, 1].plot(e2, label='Odd * Odd')
    axs[1, 1].set_title('Odd * Odd = Even')
    axs[1, 1].legend()
    
    axs[1, 2].plot(o2, label='Odd * Even')
    axs[1, 2].set_title('Odd * Even = Odd')
    axs[1, 2].legend()
    
    plt.tight_layout()
    plt.show()
    
t = np.linspace(0, 2*np.pi, 1000)
cosine = np.cos(t)
sine = np.sin(t)
signal_symmetry(cosine, sine)

# ex 3.4
def reprComplex(x):
    """3.4 Write a function reprComplex(x) that will represent the complex sequence x
in the complex plane. The points will be connected."""
    plt.plot(np.real(x), np.imag(x), 'o-')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.title('Complex Sequence Representation')
    plt.grid(True)
    plt.show()
x = np.array([1+2j, 2-1j, -3+4j, 5])
reprComplex(x)

# ex 4.1
def dft1(x):
    """Write a function dft1(x) that will compute the DFT of the sequence x using
(4.5). You will use two cycles: in the exterior one you will compute the values of X(k) and in the
interior one you will make the summations."""
    N = len(x)
    X = np.zeros(N, dtype=np.complex_)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X
x = [1, 2, 3, 4]
X = dft1(x)
print(X)

# ex 5.5
def convolution(x, h):
    """The function returns the convolution of the signal"""
    m, n = len(x), len(h)
    y = np.zeros(m+n-1)
    for i in range(m):
        for j in range(n):
            y[i+j] += x[i]*h[j]
    return y
x = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])
h1 = np.array([1, 0, 0, 0])
h2 = np.array([0, 0, 1, 0])
y1 = convolution(x, h1)
y2 = convolution(x, h2)
print("Convolution of x with h1:", y1)
print("Convolution of x with h2:", y2)
# Observations:
# We notice that the first convolution with h1 adds four zeros to the end of the sequence, 
# while the second convolution with h2 only keeps the samples where h2 is non-zero. 
# This is because h1 is a low-pass filter that attenuates high frequencies, while h2 is a high-pass filter that attenuates low frequencies.

# ex 6.1
def filter_output(b, a, x):
    y = [0]*len(x)   # Initialize the output to zeros
    for n in range(len(x)):
        y[n] = x[n] - a[1]*y[n-1] - a[2]*y[n-2]
        if n > 0:
            y[n] -= b[1]*x[n-1] - b[2]*x[n-2]
    return y

b = [1, 2, 3]
a = [1, 0, 0]
x = [1, 0, 2, 0]
y = filter_output(b, a, x)
print(y)
# Observations:
# We can notice that the output is a non-zero signal, indicating that the filter has an effect on the input signal.
# The output values are obtained by applying the difference equation of the filter to the input signal.

# ex 6.11
def lp_hanning_filter(fc, fs=10000, order=40):
    nyquist = 0.5 * fs
    wc = fc / nyquist
    
    # Design the filter using the Hanning window method
    b = signal.firwin(order+1, wc, window='hann', pass_zero=True)
    a = 1
    
    # Plot the frequency and phase response
    w, h = signal.freqz(b, a)
    freq = w * nyquist / np.pi
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    
    ax1.plot(freq, 20 * np.log10(abs(h)))
    ax1.set(title=f'LP Hanning filter order {order}', xlabel='Frequency (Hz)', ylabel='Amplitude (dB)')
    ax1.axvline(fc, color='red', linestyle='--', linewidth=1)
    ax1.axhline(-3, color='green', linestyle='--', linewidth=1)
    ax1.axhline(-40, color='orange', linestyle='--', linewidth=1)
    ax1.grid(True, which='both')
    ax1.legend(['Filter response', 'Cutoff frequency', '-3dB point', '-40dB point'])
    
    ax2.plot(freq, np.angle(h))
    ax2.set(xlabel='Frequency (Hz)', ylabel='Phase (radians)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
# will produce a plot showing the frequency and phase response of the filter, 
# the location of the cutoff frequency, the -3dB point, and the -40dB point.
lp_hanning_filter(2000,fs=10_000,order=40)