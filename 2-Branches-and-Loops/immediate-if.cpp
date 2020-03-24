// Playing with immediate if statements

#include <iostream>
using std::cin;
using std::cout;
using std::endl;
using std::string;

int main()
{
    string choice = "pi";
    // if the conditional is true then result is set to the first value.  If false then the second.
    int result = (choice == "1")? 1: 2;

    cout << result;
}