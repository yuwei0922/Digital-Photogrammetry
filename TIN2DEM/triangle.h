#pragma once
#include <iostream>
#include"point.h"

class triangle
{
protected:
	point first, second, third;
public:
	triangle();
	point getfirstPoint();
	point getsecondPoint();
	point getthirdPoint();
	void setTriangle(point, point, point);
};