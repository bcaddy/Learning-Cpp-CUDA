// Take two numbers as input from the user and return their product

#include <iostream>
using std::cout;
using std::endl;
using std::cin;

int main()
{
    // Declarations
    double a, b; // input variables
    double c;    // output, the product of a and b

    cout << "Input two numbers to multiply" << endl;
    cout << "A = ";
    cin >> a;
    cout << "B = ";
    cin >> b;

    c = a*b;
    
    cout << "C = " << c;
}