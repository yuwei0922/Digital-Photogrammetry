#define _CRT_SECURE_NO_WARNINGS
#include<opencv2\opencv.hpp>
#include <opencv2/core/core.hpp>  
#include<opencv2\opencv_modules.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <Windows.h>
#include <time.h>
#include <vector>
#include "point.h"
#include "triangle.h"

using namespace std;
using namespace cv;

#define TP_num 9898   // 9898 TIN points
#define T_num 19745   // 19745 TIN triangles

void read_file(char* filepath, vector<point>& points, vector<triangle>& triangles) {
    FILE* fp = NULL;
    errno_t err = 0;
    if ((err = fopen_s(&fp, filepath, "r")) != 0)
    {
        printf("Cannot open this file\n");
    };

    point Point;
    triangle Triangle;
    double x = 0;
    double y = 0;
    double z = 0;
    int first = 0;
    int second = 0;
    int third = 0;
    int c;
    int i = 0;

    while (i < 2 && (c = fgetc(fp)) != EOF)
    {
        if (c == '\n') 
            i++;
    }
    // start to read point and triangle 
    for (int j = 0; j < TP_num + T_num + 1; j++)
    {
        if (j < TP_num)
        {
            fscanf(fp, "%lf", &x);
            fscanf(fp, "%lf", &y);
            fscanf(fp, "%lf", &z);
            Point.setPoint(x, y, z);
            points.push_back(Point);
        }
        // skip a line
        else if (j == TP_num)
        {
            int n;
            fscanf(fp, "%d", &n);
        }
        else if (j > TP_num)
        {
            fscanf(fp, "%d", &first);
            fscanf(fp, "%d", &second);
            fscanf(fp, "%d", &third);
            Triangle.setTriangle(points[first], points[second], points[third]);
            triangles.push_back(Triangle);
        }
    }
    fclose(fp);
}

void Area(vector<point>& TIN_points, double xmax, double* xmin, double ymax, double* ymin, double interval, int* col, int* row)
{
    for (int i = 0; i < TP_num; i++)
    {
        if (TIN_points[i].getX() > xmax)
            xmax = TIN_points[i].getX();
        if (TIN_points[i].getX() < *xmin)
            *xmin = TIN_points[i].getX();
        if (TIN_points[i].getY() > ymax)
            ymax = TIN_points[i].getY();
        if (TIN_points[i].getY() < *ymin)
            *ymin = TIN_points[i].getY();
    }
    double distx = xmax - *xmin;
    double disty = ymax - *ymin;

    *col = (int)(distx / interval + 1);
    *row = (int)(disty / interval + 1);
}

void DEM(double xmin, double ymin, double interval, int col, int row, vector<triangle>& triangles)
{
    //start the clock
    clock_t start = clock();
    clock_t finish;
    vector<point>DEM_points;
    point DEM_point;
    int DEM_num = col * row;
    //initialize DEM grid(-99999)
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++) {
            DEM_point.setPoint(xmin + j * interval, ymin + i * interval, -99999);
            DEM_points.push_back(DEM_point);
        }
    }

    // identify which triangle cotaining the DEM grid point and interpolate the elevation
    vector<int>Triangle_num(DEM_num); //to save which triangle cotaining the point
    for (int i = 0; i < DEM_num; i++)
    {
        for (int j = 0; j < T_num; j++)
        {
            point PA, PB, PC, AB, BC, CA;
            PA.setPoint(triangles[j].getfirstPoint().getX() - DEM_points[i].getX(), triangles[j].getfirstPoint().getY() - DEM_points[i].getY(), 0);
            PB.setPoint(triangles[j].getsecondPoint().getX() - DEM_points[i].getX(), triangles[j].getsecondPoint().getY() - DEM_points[i].getY(), 0);
            PC.setPoint(triangles[j].getthirdPoint().getX() - DEM_points[i].getX(), triangles[j].getthirdPoint().getY() - DEM_points[i].getY(), 0);
            AB.setPoint(triangles[j].getsecondPoint().getX() - triangles[j].getfirstPoint().getX(), triangles[j].getsecondPoint().getY() - triangles[j].getfirstPoint().getY(), 0);
            BC.setPoint(triangles[j].getthirdPoint().getX() - triangles[j].getsecondPoint().getX(), triangles[j].getthirdPoint().getY() - triangles[j].getsecondPoint().getY(), 0);
            CA.setPoint(triangles[j].getfirstPoint().getX() - triangles[j].getthirdPoint().getX(), triangles[j].getfirstPoint().getY() - triangles[j].getthirdPoint().getY(), 0);
            double t1 = PA.getX() * AB.getY() - AB.getX() * PA.getY();//   PAxAB
            double t2 = CA.getX() * AB.getY() - AB.getX() * CA.getY();//   CAxAB
            if (t1 * t2 >= 0)   // P and C are on the same side of AB
            {
                double t3 = PB.getX() * BC.getY() - BC.getX() * PB.getY();//   PBxBC
                double t4 = AB.getX() * BC.getY() - BC.getX() * AB.getY();//   ABxBC
                if (t3 * t4 >= 0)   // P and A are on the same side of BC
                {
                    double t5 = PC.getX() * CA.getY() - CA.getX() * PC.getY();//   PCxCA
                    double t6 = BC.getX() * CA.getY() - CA.getX() * BC.getY();//   BCxCA
                    if (t5 * t6 >= 0)   // P and B are on the same side of CA
                    {
                        Triangle_num[i] = j;
                        // interpolate the elevation
                        double x1 = triangles[j].getfirstPoint().getX(), y1 = triangles[j].getfirstPoint().getY(), z1 = triangles[j].getfirstPoint().getZ();
                        double x2 = triangles[j].getsecondPoint().getX(), y2 = triangles[j].getsecondPoint().getY(), z2 = triangles[j].getsecondPoint().getZ();
                        double x3 = triangles[j].getthirdPoint().getX(), y3 = triangles[j].getthirdPoint().getY(), z3 = triangles[j].getthirdPoint().getZ();
                        double x21 = x2 - x1, x31 = x3 - x1;
                        double y21 = y2 - y1, y31 = y3 - y1;
                        double z21 = z2 - z1, z31 = z3 - z1;
                        double z = (z1 - ((DEM_points[i].getX() - x1) * (y21 * z31 - y31 * z21) + (DEM_points[i].getY() - y1) * (z21 * x31 - z31 * x21)) / (x21 * y31 - x31 * y21));
                        DEM_points[i].setPoint(DEM_points[i].getX(), DEM_points[i].getY(), z);
                        break;
                    }
                }
            }
        }
    }

    // output DEM
    ofstream DEM("./TIN2DEM_0.5m.ddem");
    DEM << xmin << " " << ymin << " " << 0 << " " << interval << " " << interval << " " << col << " " << row << " " << "\n";
    for (int i = 0; i < DEM_num; i++)
    {
        DEM << DEM_points[i].getZ() << " ";
        if ((i + 1) % col == 0)
        {
            DEM << "\n";
        }

    }
    finish = clock();
    cout << "running time£º" << (double)(finish - start) / CLOCKS_PER_SEC << "s" << endl;
    DEM.close();
}

int main()
{
    char filepath[] = "./facePointsTin.xyztri";
    vector<point>TIN_points;
    vector<triangle>triangles;
    read_file(filepath, TIN_points, triangles);

    // caculate DEM area and col,row
    double xmax = -100, xmin = 100, ymax = -100, ymin = 100;
    int col = 0, row = 0;
    double interval ;//set interval of DEM grid
    cout << "Please set interval of DEM:";
    cin >> interval;
    Area(TIN_points, xmax, &xmin, ymax, &ymin, interval, &col, &row);

    // generation of DEM
    DEM(xmin, ymin, interval, col, row, triangles);
    cout << "Success!" << endl;
    cout << "points of DEM:" << col * row << endl;
    return 0;
}
