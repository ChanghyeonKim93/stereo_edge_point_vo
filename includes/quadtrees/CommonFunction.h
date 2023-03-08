#ifndef _COMMONFUNCTION_H_
#define _COMMONFUNCTION_H_
#include <iostream>
#include "CommonStruct.h"

// 빠른 거리 계산을 위한 inline 처리.
// 빠른 계산을 위한 macro 처리.
#define DIST_EUCLIDEAN_PT(p1, p2) ( (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y) )
#define DIST_EUCLIDEAN_SQRT_PT(p1, p2) sqrt( (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y) )
#define DIST_MANHATTAN_PT(p1, p2) ( abs(p1.x - p2.x) + abs(p1.y - p2.y) )

inline float distEuclidean(float& u1, float& u2, float& v1, float& v2)
{
	return (u1 - u2)*(u1 - u2) + (v1 - v2)*(v1 - v2);
};
inline float distEuclidean(chk::Point2f& pt1, chk::Point2f& pt2)
{
	return (pt1.x - pt2.x)*(pt1.x - pt2.x) + (pt1.y - pt2.y)*(pt1.y - pt2.y);
};

inline float distEuclideanSqrt(float& u1, float& u2, float& v1, float& v2)
{
	return sqrt((u1 - u2)*(u1 - u2) + (v1 - v2)*(v1 - v2));
};
inline float distEuclideanSqrt(chk::Point2f& pt1, chk::Point2f& pt2)
{
	return sqrt((pt1.x - pt2.x)*(pt1.x - pt2.x) + (pt1.y - pt2.y)*(pt1.y - pt2.y));
};

inline float distManhattan(float& u1, float& u2, float& v1, float& v2)
{
	return abs(u1 - u2) + abs(v1 - v2);
};
inline float distManhattan(chk::Point2f& pt1, chk::Point2f& pt2)
{
	return abs(pt1.x - pt2.x) + abs(pt1.y- pt2.y);
};
#endif