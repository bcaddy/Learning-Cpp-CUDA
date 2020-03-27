#include "PrintShapesUtility.h"
#include <iostream>
using std::cout;
using std::endl;

void MultiPrint(int const &n, char const &letter)
{
    for (int i=0; i < n; i++)
    {
        cout << letter;
    }
}

void PrintTopTriangle(int const &base, int const &height, char const &letter)
{
    // Declare variables for spacing
    int LeftSpace;
    int LeftLetter;
    int CenterSpace;
    int CenterLetter;
    int RightLetter;
    int RightSpace;

    cout << endl;

    for (int i=0; i < height; i++)
    {
        
    }

    cout << endl;
}

void PrintBottomTrianle(int const &base, int const &height, char const &letter)
{
    cout << "bottom tri";
}


void PrintRectangle(int const &base, int const &height, char const& letter)
{
    cout << endl;

    MultiPrint(base, letter);
    cout << endl;
    for (int i=0; i < height-2; i++)
    {
        MultiPrint(1, letter);
        MultiPrint(base-2, ' ');
        MultiPrint(1, letter);
        cout << endl;
    }
    MultiPrint(base, letter);

    cout << endl;
}
