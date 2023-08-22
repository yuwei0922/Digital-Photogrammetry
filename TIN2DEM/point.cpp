#include"point.h"
using namespace std;

point::point()
{
	X = 0;
	Y = 0;
	Z = 0;
}

//Get 3D points
void point::setPoint(double x, double y, double z)
{
	X = x;
	Y = y;
	Z = z;
}

//Get coordinate of each dimension
double point::getX()
{
	return X;
}

double point::getY()
{
	return Y;
}

double point::getZ()
{
	return Z;
}