# PyAudio module needed to run the program, because it uses the microphone
import speech_recognition as sr


def receiveENSpeech_toText(trigger=bool):
    """Method used to start the microphone recording, capture the English speech and print the results of Sphinx, Google Speech and Wit.ai - then writes the captured output to a WAV and FLAC file"""
    # obtain audio from the microphone
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source)

    # recognize speech using Sphinx
    try:
        print("Sphinx thinks you said:\n" +
              recognizer.recognize_sphinx(audio))
    except sr.UnknownValueError:
        print("Sphinx could not understand the audio")
    except sr.RequestError as e:
        print("Sphinx error; {0}".format(e))

    # recognize speech using Google Speech Recognition
    try:
        print("Google Speech Recognition thinks you said:\n" +
              recognizer.recognize_google(audio, language='ro-RO'))
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio")
    except sr.RequestError as e:
        print(
            "Could not request results from Google Speech Recognition service; {0}".format(e))

    # write audio to a WAV file
    with open("microphone-results.wav", "wb") as f:
        f.write(audio.get_wav_data())


receiveENSpeech_toText(True)
