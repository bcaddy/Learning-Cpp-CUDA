/*
================================================================================
 fcuntion: grav_force
 Written by Robert Caddy.  Created on March 27, 2020

 Description: 
     find the gravitation force between two objects

 Dependencies:
     
================================================================================
*/

#include <cmath>
#include "grav_force.h"

double grav_force(double const &G, double const &M1, double const &M2, double const &r)
{
    double f;
    f = G * M1 * M2 / pow(r,2);
    return f;
}