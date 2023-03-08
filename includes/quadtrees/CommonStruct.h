#ifndef _COMMONSTRUCT_H_
#define _COMMONSTRUCT_H_

#define _OPENCV_COMPATIBLE_

#include <iostream>
#include <vector>
#include <memory>

#ifdef _OPENCV_COMPATIBLE_
	#include "opencv2/highgui/highgui.hpp"
	#include "opencv2/imgproc/imgproc.hpp"
	#include "opencv2/opencv.hpp"
	#include "Eigen/Dense" 
#endif

#define PI 3.141592653589793238
#define D2R 0.017453292519943
#define R2D 57.295779513082320

// ����ϱ� ���ϰ� ��������.
typedef unsigned short ushort; // 2����Ʈ
typedef unsigned char uchar; // 1����Ʈ
using namespace std;

namespace chk {
	// double �� ushort�� �� �� ���� �������̽��� ����ϱ� ���� ���ø����� ����.
	template <typename Tp_> 
	struct Point2 { // 2���� �� (feature)
		Tp_ x;
		Tp_ y;
		Point2() : x(0), y(0) {}; // �⺻ ������
		Point2(Tp_ x_, Tp_ y_) : x(x_), y(y_) {};
		Point2(const Point2& pt) { x = pt.x; y = pt.y;}; // ���� ������ (�ʱ�ȭ���� = ������ ����)

		// operator overloading
		double norm() { return sqrt((double)x*x + (double)y*y); };
        double dot(const Point2& pt) { return (x*pt.x + y*pt.y); };
        double cross2d(const Point2& pt) { return (-pt.y*x + pt.x*y); };
        template <typename Tin_>// for arbitrary datatype.
        friend Point2 operator* (Point2& pt1, Tin_ scalar) { return Point2(pt1.x*(float)scalar, pt1.y*(float)scalar); };
        
        friend Point2 operator+ (Point2& pt1, Point2& pt2) { return Point2(pt1.x + pt2.x, pt1.y + pt2.y); };
        friend Point2 operator- (Point2& pt1, Point2& pt2) { return Point2(pt1.x - pt2.x, pt1.y - pt2.y); };
        Point2& operator+=(const Point2& rhs) { x += rhs.x; y += rhs.y; return *this; };
        Point2& operator-=(const Point2& rhs) { x -= rhs.x; y -= rhs.y; return *this; };

        template <typename Tin_>// for arbitrary datatype.
        Point2& operator/=(const Tin_& rhs) { x /= (double)rhs; y /= (double)rhs; return *this; };
        template <typename Tin_>
        Point2& operator*=(const Tin_& rhs) { x *= (double)rhs; y *= (double)rhs; return *this; };

        // ����Ʈ ���� �����ڴ� �츮�� �𸣰� �����ϴµ�, �׳� �ּҸ� �����ϴ� �����̴�.
		// ���� ���� ����������Ѵ�.
		Point2& operator=(const Point2& ref) { x = ref.x; y = ref.y; return *this; };
        // ����. �ʱ�ȭ ���� ������. ���� �ּҸ� ���������� �ʾƿ�. 
		// �� ? �����Ҵ��� ������ �ʾƿ�.
		friend ostream& operator<<(ostream& os, const Point2& pt) {
			os << "[" << pt.x << "," << pt.y << "]";
			return os;
		};
		cv::Point2f chk2cv() { return cv::Point2f(x, y); };
	};

	template <typename Tp_>
	struct Point3 { // 3���� �� (3d point)
		Tp_ x;
		Tp_ y;
		Tp_ z;
		Point3() : x(0), y(0), z(0) {}; // �⺻ ������
		Point3(Tp_ x_, Tp_ y_, Tp_ z_) : x(x_), y(y_), z(z_) {};
		Point3(const Point3& pt) { x = pt.x; y = pt.y; z = pt.z; }; // ���� ������

        double norm() { return sqrt((double)x*x + (double)y*y + (double)z*z); };
        double dot(const Point3& pt) { return (x*pt.x + y*pt.y + z*pt.z); };
        // double cross3d(const Point3& pt) { return (-pt.y*x + pt.x*y); };
        template <typename Tin_>// for arbitrary datatype.
        friend Point3 operator* (Point3& pt1, Tin_ scalar) { return Point3(pt1.x*(float)scalar, pt1.y*(float)scalar, pt1.z*(float)scalar); };

        friend Point3 operator+ (Point3& pt1, Point3& pt2) { return Point3(pt1.x + pt2.x, pt1.y + pt2.y, pt1.z + pt2.z); };
        friend Point3 operator- (Point3& pt1, Point3& pt2) { return Point3(pt1.x - pt2.x, pt1.y - pt2.y, pt1.z - pt2.z); };
        Point3& operator+=(const Point3& rhs) { x += rhs.x; y += rhs.y; z += rhs.z; return *this; };
        Point3& operator-=(const Point3& rhs) { x -= rhs.x; y -= rhs.y; z -= rhs.z; return *this; };

        template <typename Tin_>// for arbitrary datatype.
        Point3& operator/=(const Tin_& rhs) { x /= (double)rhs; y /= (double)rhs; z /= (double)rhs; return *this; };
        template <typename Tin_>
        Point3& operator*=(const Tin_& rhs) { x *= (double)rhs; y *= (double)rhs; z *= (double)rhs; return *this; };

        // ����Ʈ ���� �����ڴ� �츮�� �𸣰� �����ϴµ�, �׳� �ּҸ� �����ϴ� �����̴�.
        // ���� ���� ����������Ѵ�.
        Point3& operator=(const Point3& ref) { x = ref.x; y = ref.y; z = ref.z; return *this; };
        // ����. �ʱ�ȭ ���� ������. ���� �ּҸ� ���������� �ʾƿ�. 
        // �� ? �����Ҵ��� ������ �ʾƿ�.
        friend ostream& operator<<(ostream& os, const Point3& pt) {
            os << "[" << pt.x << "," << pt.y <<","<<pt.z<< "]";
            return os;
        };
        cv::Point3f chk2cv() { return cv::Point3f(x, y, z); };
	};


    // defines nicknames
    typedef Point2<uchar> Point2uc;
    typedef Point2<char> Point2c;
	typedef Point2<ushort> Point2us;
	typedef Point2<short> Point2s;
	typedef Point2<int> Point2i;
	typedef Point2<float> Point2f;
	typedef Point2<double> Point2d;

	typedef Point3<uchar> Point3uc;
    typedef Point3<char> Point3c;
	typedef Point3<ushort> Point3us;
	typedef Point3<short> Point3s;
	typedef Point3<int> Point3i;
	typedef Point3<float> Point3f;
	typedef Point3<double> Point3d;

#ifdef _OPENCV_COMPATIBLE_
	template <typename Tp_>
	inline void chk2cv(const chk::Point2<Tp_>& pt, cv::Point2f& pt_cv) {
		pt_cv.x = pt.x;
		pt_cv.y = pt.y;
	};
	template <typename Tp_>
	inline void chk2cv(const chk::Point3<Tp_>& pt, cv::Point3f& pt_cv) {
		pt_cv.x = pt.x;
		pt_cv.y = pt.y;
		pt_cv.z = pt.z;
	};
#endif
};



struct Bound {
	chk::Point2us nw; // �»��.
	chk::Point2us se; // ���ϴ�.
	Bound() : nw(chk::Point2us(0, 0)), se(chk::Point2us(0, 0)) {}; // �⺻ ������
	Bound(chk::Point2us nw_, chk::Point2us se_) :nw(nw_), se(se_) {};
	void doDefault() {
		nw.x = -1;
		nw.y = -1;
		se.x = -1;
		se.y = -1;
	}
};

// �̹����� ũ�Ⱑ 65536 �̻��� ���, �������� ����.
// �ٷ� ������ ���� 4���� children ��, ù��° child�� ����Ű�� �ּ� ��.
// 1,2,3,4 ������� HL, HR, BL, BR. (�»�, ���, ����, ����)
// Z �������
//
// | 00 | 01 |   | HL | HR |
// |----|----| = |----|----| : LSB(�¿�), MSB(����)
// | 10 | 11 |   | BL | BR |
//
// first_child + 0: HL, first_child + 1: HR, first_child + 2: BL, first_child + 3: BR
// 64 bits �ü���̹Ƿ� 8 bytes�� �ּҸ� ���� �� �� �ִ�.
struct Node { // size = 32 bytes
	Node* parent;    // 8 bytes
	Node* first_child; // 8 first_child + 1: HR, first_child + 2: BL, first_child + 3: BR
	Bound bound;     // 8 bytes  0 ~ 65536 ������ �̹��� ��ǥ�� �Ҽ������� �������� ū �ǹ̾���.
	uchar depth;     // 1, ���̵� ����ܰ�� �������� �ʱ� ������ 0~255�̸� ���.
	bool isleaf;     // 1, leaf���� �ƴ��� �˷���.
	bool isvisited;     // 1, �湮�ߴ��� �ƴ��� �˷���. (cached version�� ����)
	int header_elem; // 4 ���� leaf����̸�, ����Ʈ ����Ʈ�� ����� ����Ʈ�� �ּҷ� �����Ѵ�.

					 // ���ο� ��� ������.
	Node() : parent(nullptr), depth(0), isleaf(false), isvisited(false),
		bound(Bound(chk::Point2us(-1, -1), chk::Point2us(-1, -1))), header_elem(-1) {};

	Node(Node* parent_, int depth_, bool isleaf_)
		: parent(parent_), depth(depth_), isleaf(isleaf_), isvisited(false),
		bound(Bound(chk::Point2us(-1, -1), chk::Point2us(-1, -1))), header_elem(-1) {};

	void doDefault() {
		parent = nullptr;
		first_child = nullptr;
		bound.doDefault();
		depth  = 0;
		isleaf = false;
		isvisited = false;
		header_elem = -1;
	}

	// � ��忡 �ڽ��� �����µ�, query�� �����ϸ� �����ϴ� �Լ�.
	void initializeChildren() {
		// �޸� �Ҵ�. ����ִ� �ڽĵ��� �ʱ�ȭ���ش�.
		// this->first_child = (Node*)malloc(sizeof(Node) * 4);
		// ��ü �迭 �����Ҵ��� �ʱ�ȭ�� �Ұ����ϴ�.. �⺻�����ڿ��� �ް��Ѱ� �ذ��ؾ��Ѵ�.
		for (int i = 0; i < 4; i++) {
			(this->first_child + i)->parent = this;
			(this->first_child + i)->first_child = nullptr;
			(this->first_child + i)->isleaf = false;
			(this->first_child + i)->isvisited = false;
			(this->first_child + i)->depth = 0;
			(this->first_child + i)->header_elem = -1;
		}
	};

	void showNodeSpec() {
		cout << "node spec.\n";
		cout << "lvl: " << (int)this->depth;
		cout << ", parent: " << this->parent;
		cout << ", this  : " << this;
		cout << ", bound: [" << this->bound.nw.x << "," << this->bound.nw.y <<
			"] / [" << this->bound.se.x << "," << this->bound.se.y << "]";
		cout << ", leaf: " << this->isleaf;
		cout << ", visit: " << this->isvisited;
		cout << ", 1stelem: " << this->header_elem << endl;
	}
};

// node���� �����ϴ� stack���δٰ� ������.
template <typename T>
class PointerStack {
public:
	int size;
	int MAX_SIZE;
	int total_access;
	T** values; // ���� �Ҵ� ��, ������ �迭�̶�� ���� �ȴ�.
	PointerStack() {
		MAX_SIZE = 65536; // stack�� �׷��� ũ�� �ʾƵ� �Ǵ���.
		values = (T**)malloc(sizeof(T*)*MAX_SIZE); // 8 * max_size
		size = 0; // ���� �ƹ��͵� �� ����.
		total_access = 0; // �� ����̳� ��忡 �����ߴ���? leaf�� BOB ����� ��忡 ���ؼ��� ���. 
						  // �� ��Ʈ�� ������ ��Ī�ϱ��������� �ʱ�ȭ X.
	}
	~PointerStack() { delete[] values; }// �����ߴ� stack ����.
	void push(T* value_) { // ����带 �����鼭 stack size 1�� Ű��.
		if (!isFull()) {
			*(values + size) = value_;
			++size;
		}
		else printf("[ERROR]: NodeStack is full.\n");
	}
	void pop() {
		if (!empty()) --size;
		else printf("[ERROR]: NodeStack is empty.\n");
	}
	T* top() {
		if (!empty()) return *(values + (size - 1));
		else return nullptr;
	}
	// ���� �ִ��� ������ Ȯ��.
	bool empty() {
		if (size < 1) return true;
		else return false;
	}
	// �� á���� �ƴ��� Ȯ��.
	bool isFull() { return (size == MAX_SIZE); }
	// stack�� ����ϰ� ������ ���� �ʿ�� ������, size=0���� ���������Ѵ�.
	void clear() { size = 0; }
};

// �� Element�� doubly one-way directed list.
//  ---------     |    2   [|--->
// |        [|--->|---------|
// |    1    | 
// |        [|--->|---------|
//  ---------     |    3   [|--->
//
// (1) 1���� ���⸸ �����ϴ� ����. 
// first_next > 0 & second_next == nullptr;
// first_dir > -1 & second_dir == -1
// (2) 2���� ������ �����ϴ� ����. 
// second_next > 0 & first_next > 0
// second_dir > first_dir > -1
// -> dir�� ���� ũ�� second�� �̵���Ų��.
struct ElemOverlap { // 40 bytes
					 // ���� ����� Elem 2���� �����Ϳ� ���� �� �� �ִ� ����
					 // first_next: ù��° gradient ����, first_next + 1: �ι�° gradient ����.
	ElemOverlap* first_next;  // 8 bytes, ù��° ����
	ElemOverlap* second_next; // 8 bytes, �ι�° ����, �̰� NULL�� �ƴϸ�, Shared region�� �ִ°��̴�.
	chk::Point2f pt;

	int id;       // 4 bytes,  �����ͺ��̽��� ���õ� �� ��ȣ.
				  // first dir�� �׻� second_dir���� ���������� �Ѵ�.
	char first_dir;  // 1 byte
	char second_dir; // 1 byte

	ElemOverlap() : first_next(nullptr), second_next(nullptr), pt(chk::Point2f(-1, -1)), first_dir(-1), second_dir(-1), id(-1) {}; // �⺻ ������, �Ⱦ���

	ElemOverlap(float& u_, float& v_, char& dir_, int& id_)
		: first_next(nullptr), second_next(nullptr), pt(chk::Point2f(u_, v_)), first_dir(dir_), second_dir(-1), id(id_) {}; // �ʱ�ȭ ������ (dir 1��)
	ElemOverlap(chk::Point2f& pt_, char& dir_, int& id_)
		: first_next(nullptr), second_next(nullptr), pt(pt_), first_dir(dir_), second_dir(-1), id(id_) {}; // �ʱ�ȭ ������ (dir 1��)

	ElemOverlap(float& u_, float& v_, char& dir1_, char& dir2_, int& id_)  // �ʱ�ȭ ������ (dir 2��)
		: first_next(nullptr), second_next(nullptr), pt(chk::Point2f(u_, v_)), id(id_)
	{
		if (dir1_ < dir2_) {
			first_dir = dir1_;
			second_dir = dir2_;
		}
		else {
			first_dir = dir2_;
			second_dir = dir1_;
		}
	};
	ElemOverlap(chk::Point2f& pt_, char& dir1_, char& dir2_, int& id_)  // �ʱ�ȭ ������ (dir 2��)
		: first_next(nullptr), second_next(nullptr), pt(pt_), id(id_)
	{
		if (dir1_ < dir2_) {
			first_dir = dir1_;
			second_dir = dir2_;
		}
		else {
			first_dir = dir2_;
			second_dir = dir1_;
		}
	};
};

struct Elem {   // 24 bytes, singly index linked list. (���� ������ �Ҵ��ؾ��ϴµ�)
	Elem* next; //  8 bytes, ������ ����� ����. NULL�̸� ����� ���Ұ� ����.
	chk::Point2f pt;   // 8 bytes, ���� ���� ��ǥ.
	int id;     // 4 bytes, // �ش� element�� id ��ȣ. (Ʈ�� ���ο��� ���Ǵ� ��)
	int idx_pts; // 4 bytes, // vector<chk::Point2f<float>> chk::Points ������ id ��ȣ. not necessarily id == id_pts

	Elem() : next(nullptr), pt(chk::Point2f(-1, -1)), id(-1), idx_pts(-1) {}; // �⺻������
	Elem(const float& x_, const float& y_) : next(nullptr), pt(chk::Point2f(x_, y_)), id(-1), idx_pts(-1) {};
	Elem(chk::Point2f& pt_) : next(nullptr), pt(pt_), id(-1), idx_pts(-1) {};
	void doDefault() {
		next = nullptr;
		pt.x = -1;
		pt.y = -1;
		id = -1;
        idx_pts = -1;
	}
};

#endif