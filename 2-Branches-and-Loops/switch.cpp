// Playing with if statements

#include <iostream>
using std::cin;
using std::cout;
using std::endl;
using std::string;

int main()
{
    int choice = 3;

    switch (choice)
    {
    case 1:
        cout << "a";
        break;
    case 2: 
        cout << "b";
    case 3: 
        cout << "c";
        break;
    }
}