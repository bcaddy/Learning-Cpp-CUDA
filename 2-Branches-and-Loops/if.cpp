// Playing with if statements

#include <iostream>
using std::cin;
using std::cout;
using std::endl;
using std::string;

int main()
{
    string choice = "p2";

    if (choice == "p1")
    {
        cout << "if";
    }
    else if (choice == "p2")
    {
        cout << "elif";
    }
    else
    {
        cout << "Else";
    }
    
}