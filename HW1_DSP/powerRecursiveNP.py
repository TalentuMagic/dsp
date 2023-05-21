import numpy as np
import matplotlib.pyplot as plt


# the variable n represents the power of the base
def powerRecursive(base, n):
    if n == 0:
        return 1
    else:
        return base * powerRecursive(base, n-1)


def main():
    numsExercises = np.array([2.26, 3.11, 3.4, 4.1, 5.5, 6.1, 6.11])
    numsExamples = np.array([2.27, 3.4, 3.12, 4.1, 6.15])

    # vectorizing the results = outputs a single numpy array from a sequence of np objects
    power = int(input("Choose the power of the recursive function:"))
    resultExamples = np.vectorize(powerRecursive)(numsExamples, power)
    resultExercises = np.vectorize(powerRecursive)(numsExercises, power)

    # creating the plots
    plt.plot(numsExercises, resultExercises, label="Exercises")
    plt.plot(numsExamples, resultExamples, label="Examples")
    plt.scatter(numsExercises, resultExercises, c='b',
                marker='o', label='Exercises Number')
    plt.scatter(numsExamples, resultExamples, c='r',
                marker='o', label='Examples Number')
    plt.xlabel('Number')
    plt.ylabel('Result')
    plt.title('Power Recursive')
    plt.legend()
    plt.grid()
    plt.show()

    # to print the exact numeric results for both number sets
    print("For exercises numbers:")
    for num in numsExercises:
        result = powerRecursive(num, power)
        print(f"{num} raised to the power of {power} is {round(result,5)}")
    print("\nFor examples numbers:")
    for num in numsExamples:
        result = powerRecursive(num, power)
        print(f"{num} raised to the power of {power} is {round(result,5)}")


if __name__ == '__main__':
    main()
