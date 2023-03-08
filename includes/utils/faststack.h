#ifndef _FASTSTACK_H_
#define _FASTSTACK_H_

#include <iostream>
#include <memory>
#include <exception>
using namespace std;

// node만을 저장하는 stack으로다가 만들자.
template <typename T>
class FastStack {
public:
	int sz;
	int MAX_SIZE;
	T* mem; // 동적 할당 된, 포인터 배열이라고 보면 된다.
	FastStack() {
		MAX_SIZE = 65536; // stack은 그렇게 크지 않아도 되더라.
		mem = (T*)malloc(sizeof(T)*MAX_SIZE); // 8 * max_size
		sz = 0; // 아직 아무것도 안 넣음.
		// cout << "FASTSTACK init" << endl;
	}
	~FastStack() { free(mem); }// 생성했던 stack 해제. // malloc은 free
	void Push(T& value_) { // 새노드를 넣으면서 stack size 1개 키움.
		if (!isFull()) {
			*(mem + sz) = value_;
			++sz;
		}
		else throw std::runtime_error("[ERROR]: FastStack is full.\n");
	}
	void pop() {
		if (!empty()) --sz;
		else throw std::runtime_error("[ERROR]: FastStack is empty.\n");
	}
	T top() {
		if (!empty()) return *(mem + (sz - 1));
		else return -1;
	}
	// 뭔가 있는지 없는지 확인.
	bool empty() {
		if (sz < 1) return true;
		else return false;
	}
	// 꽉 찼는지 아닌지 확인.
	bool isFull() { return (sz == MAX_SIZE); }
	// stack을 사용하고 내용을 지울 필요는 없지만, size=0으로 만들어줘야한다.
	void clear() { sz = 0; }
};

#endif
