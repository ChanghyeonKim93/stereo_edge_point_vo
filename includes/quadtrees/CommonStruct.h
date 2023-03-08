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

// 사용하기 편하게 정의하자.
typedef unsigned short ushort; // 2바이트
typedef unsigned char uchar; // 1바이트
using namespace std;

namespace chk {
	// double 과 ushort를 둘 다 같은 인터페이스를 사용하기 위해 템플릿으로 정의.
	template <typename Tp_> 
	struct Point2 { // 2차원 점 (feature)
		Tp_ x;
		Tp_ y;
		Point2() : x(0), y(0) {}; // 기본 생성자
		Point2(Tp_ x_, Tp_ y_) : x(x_), y(y_) {};
		Point2(const Point2& pt) { x = pt.x; y = pt.y;}; // 복사 생성자 (초기화이전 = 나오면 복사)

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

        // 디폴트 대입 연산자는 우리가 모르게 존재하는데, 그냥 주소를 공유하는 구조이다.
		// 따라서 새로 정의해줘야한다.
		Point2& operator=(const Point2& ref) { x = ref.x; y = ref.y; return *this; };
        // 대입. 초기화 이후 변수들. 같은 주소를 공유하지는 않아용. 
		// 왜 ? 동적할당을 가지지 않아요.
		friend ostream& operator<<(ostream& os, const Point2& pt) {
			os << "[" << pt.x << "," << pt.y << "]";
			return os;
		};
		cv::Point2f chk2cv() { return cv::Point2f(x, y); };
	};

	template <typename Tp_>
	struct Point3 { // 3차원 점 (3d point)
		Tp_ x;
		Tp_ y;
		Tp_ z;
		Point3() : x(0), y(0), z(0) {}; // 기본 생성자
		Point3(Tp_ x_, Tp_ y_, Tp_ z_) : x(x_), y(y_), z(z_) {};
		Point3(const Point3& pt) { x = pt.x; y = pt.y; z = pt.z; }; // 복사 생성자

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

        // 디폴트 대입 연산자는 우리가 모르게 존재하는데, 그냥 주소를 공유하는 구조이다.
        // 따라서 새로 정의해줘야한다.
        Point3& operator=(const Point3& ref) { x = ref.x; y = ref.y; z = ref.z; return *this; };
        // 대입. 초기화 이후 변수들. 같은 주소를 공유하지는 않아용. 
        // 왜 ? 동적할당을 가지지 않아요.
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
	chk::Point2us nw; // 좌상단.
	chk::Point2us se; // 우하단.
	Bound() : nw(chk::Point2us(0, 0)), se(chk::Point2us(0, 0)) {}; // 기본 생성자
	Bound(chk::Point2us nw_, chk::Point2us se_) :nw(nw_), se(se_) {};
	void doDefault() {
		nw.x = -1;
		nw.y = -1;
		se.x = -1;
		se.y = -1;
	}
};

// 이미지의 크기가 65536 이상인 경우, 동작하지 않음.
// 바로 다음에 오는 4개의 children 중, 첫번째 child를 가리키는 주소 값.
// 1,2,3,4 순서대로 HL, HR, BL, BR. (좌상, 우상, 좌하, 우하)
// Z 모양으로
//
// | 00 | 01 |   | HL | HR |
// |----|----| = |----|----| : LSB(좌우), MSB(상하)
// | 10 | 11 |   | BL | BR |
//
// first_child + 0: HL, first_child + 1: HR, first_child + 2: BL, first_child + 3: BR
// 64 bits 운영체제이므로 8 bytes에 주소를 저장 할 수 있다.
struct Node { // size = 32 bytes
	Node* parent;    // 8 bytes
	Node* first_child; // 8 first_child + 1: HR, first_child + 2: BL, first_child + 3: BR
	Bound bound;     // 8 bytes  0 ~ 65536 어차피 이미지 좌표를 소수점으로 나눠봤자 큰 의미없다.
	uchar depth;     // 1, 깊이도 수백단계로 내려가지 않기 때문에 0~255이면 충분.
	bool isleaf;     // 1, leaf인지 아닌지 알려줌.
	bool isvisited;     // 1, 방문했는지 아닌지 알려줌. (cached version을 위함)
	int header_elem; // 4 만약 leaf노드이면, 포인트 리스트에 저장된 포인트의 주소로 접근한다.

					 // 새로운 노드 생성자.
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

	// 어떤 노드에 자식이 없었는데, query가 접근하면 동작하는 함수.
	void initializeChildren() {
		// 메모리 할당. 비어있는 자식들을 초기화해준다.
		// this->first_child = (Node*)malloc(sizeof(Node) * 4);
		// 객체 배열 동적할당은 초기화가 불가능하다.. 기본생성자에서 앵간한건 해결해야한다.
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

// node만을 저장하는 stack으로다가 만들자.
template <typename T>
class PointerStack {
public:
	int size;
	int MAX_SIZE;
	int total_access;
	T** values; // 동적 할당 된, 포인터 배열이라고 보면 된다.
	PointerStack() {
		MAX_SIZE = 65536; // stack은 그렇게 크지 않아도 되더라.
		values = (T**)malloc(sizeof(T*)*MAX_SIZE); // 8 * max_size
		size = 0; // 아직 아무것도 안 넣음.
		total_access = 0; // 총 몇번이나 노드에 접근했는지? leaf나 BOB 통과한 노드에 대해서만 사용. 
						  // 한 세트의 점군을 매칭하기전까지는 초기화 X.
	}
	~PointerStack() { delete[] values; }// 생성했던 stack 해제.
	void push(T* value_) { // 새노드를 넣으면서 stack size 1개 키움.
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
	// 뭔가 있는지 없는지 확인.
	bool empty() {
		if (size < 1) return true;
		else return false;
	}
	// 꽉 찼는지 아닌지 확인.
	bool isFull() { return (size == MAX_SIZE); }
	// stack을 사용하고 내용을 지울 필요는 없지만, size=0으로 만들어줘야한다.
	void clear() { size = 0; }
};

// 각 Element의 doubly one-way directed list.
//  ---------     |    2   [|--->
// |        [|--->|---------|
// |    1    | 
// |        [|--->|---------|
//  ---------     |    3   [|--->
//
// (1) 1개의 방향만 지지하는 상태. 
// first_next > 0 & second_next == nullptr;
// first_dir > -1 & second_dir == -1
// (2) 2개의 방향을 지지하는 상태. 
// second_next > 0 & first_next > 0
// second_dir > first_dir > -1
// -> dir은 값이 크면 second로 이동시킨다.
struct ElemOverlap { // 40 bytes
					 // 다음 연결된 Elem 2개의 포인터에 접근 할 수 있는 변수
					 // first_next: 첫번째 gradient 방향, first_next + 1: 두번째 gradient 방향.
	ElemOverlap* first_next;  // 8 bytes, 첫번째 방향
	ElemOverlap* second_next; // 8 bytes, 두번째 방향, 이게 NULL이 아니면, Shared region에 있는것이다.
	chk::Point2f pt;

	int id;       // 4 bytes,  데이터베이스에 마련된 점 번호.
				  // first dir은 항상 second_dir보다 작은수여야 한다.
	char first_dir;  // 1 byte
	char second_dir; // 1 byte

	ElemOverlap() : first_next(nullptr), second_next(nullptr), pt(chk::Point2f(-1, -1)), first_dir(-1), second_dir(-1), id(-1) {}; // 기본 생성자, 안쓸듯

	ElemOverlap(float& u_, float& v_, char& dir_, int& id_)
		: first_next(nullptr), second_next(nullptr), pt(chk::Point2f(u_, v_)), first_dir(dir_), second_dir(-1), id(id_) {}; // 초기화 생성자 (dir 1개)
	ElemOverlap(chk::Point2f& pt_, char& dir_, int& id_)
		: first_next(nullptr), second_next(nullptr), pt(pt_), first_dir(dir_), second_dir(-1), id(id_) {}; // 초기화 생성자 (dir 1개)

	ElemOverlap(float& u_, float& v_, char& dir1_, char& dir2_, int& id_)  // 초기화 생성자 (dir 2개)
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
	ElemOverlap(chk::Point2f& pt_, char& dir1_, char& dir2_, int& id_)  // 초기화 생성자 (dir 2개)
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

struct Elem {   // 24 bytes, singly index linked list. (연속 공간에 할당해야하는뎅)
	Elem* next; //  8 bytes, 다음에 연결된 원소. NULL이면 연결된 원소가 없음.
	chk::Point2f pt;   // 8 bytes, 현재 점의 좌표.
	int id;     // 4 bytes, // 해당 element의 id 번호. (트리 내부에서 사용되는 값)
	int idx_pts; // 4 bytes, // vector<chk::Point2f<float>> chk::Points 에서의 id 번호. not necessarily id == id_pts

	Elem() : next(nullptr), pt(chk::Point2f(-1, -1)), id(-1), idx_pts(-1) {}; // 기본생성자
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