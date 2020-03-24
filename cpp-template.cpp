/*
================================================================================
 Written by Robert Caddy.  Created on DATE

 Description (in paragraph form)

 Dependencies:

 Changelog:
     Version 1.0 - First Version
================================================================================
*/

#pragma once
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