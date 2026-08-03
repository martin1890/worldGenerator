#include <array>
#include <iostream>
#include <random>

static constexpr int GRID_SIZE = 40;
static constexpr int REGION_LENGTH = 20;

static int area[GRID_SIZE * GRID_SIZE] = {};

// Each row contains the transition probabilities from one state.
static constexpr double MARKOV_MATRIX[5][5] = {
    {0.10, 0.90, 0.00, 0.00, 0.00},
    {0.20, 0.30, 0.50, 0.00, 0.00},
    {0.00, 0.35, 0.30, 0.35, 0.00},
    {0.00, 0.00, 0.50, 0.30, 0.20},
    {0.00, 0.00, 0.00, 0.90, 0.10}
};

// Absolute directions, rotating counterclockwise.
static constexpr std::array<int, 8> DX = {
    1, 1, 0, -1, -1, -1, 0, 1
};

static constexpr std::array<int, 8> DY = {
    0, 1, 1, 1, 0, -1, -1, -1
};

static std::mt19937 generator(std::random_device{}());

int randomInt(int lower, int upper)
{
    std::uniform_int_distribution<int> distribution(lower, upper);
    return distribution(generator);
}

double randomDouble()
{
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    return distribution(generator);
}

bool isInsideGrid(int x, int y)
{
    return x >= 0 && x < GRID_SIZE &&
           y >= 0 && y < GRID_SIZE;
}

int gridIndex(int x, int y)
{
    return x + y * GRID_SIZE;
}

int sampleNextState(int currentState)
{
    const double randomValue = randomDouble();
    double cumulativeProbability = 0.0;

    for (int nextState = 0; nextState < 5; ++nextState) {
        cumulativeProbability +=
            MARKOV_MATRIX[currentState][nextState];

        if (randomValue < cumulativeProbability) {
            return nextState;
        }
    }

    // Protection against tiny floating-point errors.
    return 4;
}

void markovStep(
    int& x,
    int& y,
    int& state,
    int mainDirection)
{
    /*
        State relative to the main direction:

        0 = -90 degrees
        1 = -45 degrees
        2 = straight
        3 = +45 degrees
        4 = +90 degrees
    */

    const int relativeDirection = state - 2;

    // Adding 8 before modulo prevents negative results.
    const int absoluteDirection =
        (mainDirection + relativeDirection + 8) % 8;

    x += DX[absoluteDirection];
    y += DY[absoluteDirection];

    state = sampleNextState(state);
}

void createMarkovCurve(
    int startX,
    int startY,
    int mainDirection,
    int length)
{
    int x = startX;
    int y = startY;

    // Start in one of the five relative directions.
    int state = randomInt(0, 4);

    for (int i = 0; i < length; ++i) {
        if (!isInsideGrid(x, y)) {
            break;
        }

        area[gridIndex(x, y)] = 1;

        markovStep(
            x,
            y,
            state,
            mainDirection
        );
    }
}

void printArea()
{
    for (int y = GRID_SIZE - 1; y >= 0; --y) {
        for (int x = 0; x < GRID_SIZE; ++x) {
            if (area[gridIndex(x, y)] == 1) {
                std::cout << "# ";
            }
            else {
                std::cout << ". ";
            }
        }

        std::cout << '\n';
    }
}

int main()
{
    /*
        Main directions:

        0 = east
        1 = northeast
        2 = north
        3 = northwest
        4 = west
        5 = southwest
        6 = south
        7 = southeast
    */

    const int startX = GRID_SIZE / 2;
    const int startY = GRID_SIZE / 2;

    const int mainDirection = randomInt(0, 7);

    createMarkovCurve(
        startX,
        startY,
        mainDirection,
        REGION_LENGTH
    );

    printArea();
}