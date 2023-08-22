#include"triangle.h"
using namespace std;

triangle::triangle()
{
	first.setPoint(0, 0, 0);
	second.setPoint(0, 0, 0);
	third.setPoint(0, 0, 0);
}

//Get vertex coordinates
point triangle::getfirstPoint()
{
	return first;
}

point triangle::getsecondPoint()
{
	return second;
}

point triangle::getthirdPoint()
{
	return third;
}

//Set triangle
void triangle::setTriangle(point First, point Second, point Third)
{
	first = First;
	second = Second;
	third = Third;
}