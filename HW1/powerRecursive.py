# the function computes the Power recursively until the power reaches 0
def powerRecursive(num, power):
    if power == 0:
        return 1
    else:
        return num * powerRecursive(num, power-1)


def main():
    numbers = [2.26, 3.11, 3.4, 4.1, 5.5, 6.1, 6.11]
    power = int(input("Choose the power of the recursive function:"))

    for num in numbers:
        result = powerRecursive(num, power)
        print(f"{num} raised to the power of {power} is {result}")


if __name__ == "__main__":
    main()
