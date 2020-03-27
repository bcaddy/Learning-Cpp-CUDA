/*
================================================================================
 Gravitational Force
 Written by Robert Caddy.  Created on March 27, 2020

 Description: 
     Practice with functions by calling a function that finds the gravitational 
     force between two objects

 Dependencies:
     grav_force.h
     grav_force.cpp
================================================================================
*/

#include <iostream>
#include "grav_force.h"

using std::cin;
using std::cout;
using std::endl;
using std::string;

int main()
{
    double MEarth  = 5.9E24;
    double MSata   = 500.;
    double const G  = 6.67E-11;
    double Distance = 42.E6;

    double force = grav_force(G, MEarth, MSata, Distance);

    cout << force;

}