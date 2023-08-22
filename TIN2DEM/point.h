#pragma once
#include <iostream>

class point
{
protected:
	double X, Y, Z;
public:
	point();
	void setPoint(double, double, double);
	double getX();
	double getY();
	double getZ();
};