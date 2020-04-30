#include <iostream> 
#include <string>
using std::cout;
using std::cin;
using std::endl;
using std::is_floating_point;
/* 
test comment color
 */

int main()
{
    cout.precision(20);
    const double d=0.12345678901234567890;
    const float  f=0.12345678901234567890;

    #define PRINT(x) (cout << "hi" << "\n"; cout.flush())

    // cout << "Please input a number: ";
    // float Num1;
    // cin >> Num1;

    // cout << "Please input a number: ";
    // float Num2;
    // cin >> Num2;

    // if (Num1 > Num2)
    // {
    //     cout << Num1 << " is greater than " << Num2 << ".";
    // }
    // else if (Num2 > Num1)
    // {
    //     cout << Num2 << " is greater than " << Num1 << ".";
    // }
    // else
    // {
    //     cout << Num1 << " and " << Num2 << " are the same.";
    // }
}

