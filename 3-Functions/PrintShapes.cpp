/*
================================================================================
 Print Shapes
 Written by Robert Caddy.  Created on March 27, 2020

 Description: 
     Prints the char C in different shapes using the MultiPrint and Print*SHAPE*
     functions.

 Dependencies:
     MultiPrint.h
     MultiPrint.cpp
================================================================================
*/

#include "PrintShapesUtility.h"

    int
    main()
{
    int TriangleHeight = 6;
    int TriangleBase   = 5;
    int RecHeight = 5;
    int RecBase   = 10;
    char letter   = 'C';
    //MultiPrint(0,'C');

    PrintTopTriangle(TriangleBase, TriangleHeight, letter);
    //PrintBottomTrianle(TriangleBase, TriangleHeight, letter);
    PrintRectangle(RecBase, RecHeight, letter);
}