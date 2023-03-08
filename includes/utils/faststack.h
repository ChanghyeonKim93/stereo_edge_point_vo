#ifndef _FASTSTACK_H_
#define _FASTSTACK_H_

#include <iostream>
#include <memory>
#include <exception>
using namespace std;

// node���� �����ϴ� stack���δٰ� ������.
template <typename T>
class FastStack {
public:
	int sz;
	int MAX_SIZE;
	T* mem; // ���� �Ҵ� ��, ������ �迭�̶�� ���� �ȴ�.
	FastStack() {
		MAX_SIZE = 65536; // stack�� �׷��� ũ�� �ʾƵ� �Ǵ���.
		mem = (T*)malloc(sizeof(T)*MAX_SIZE); // 8 * max_size
		sz = 0; // ���� �ƹ��͵� �� ����.
		// cout << "FASTSTACK init" << endl;
	}
	~FastStack() { free(mem); }// �����ߴ� stack ����. // malloc�� free
	void Push(T& value_) { // ����带 �����鼭 stack size 1�� Ű��.
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
	// ���� �ִ��� ������ Ȯ��.
	bool empty() {
		if (sz < 1) return true;
		else return false;
	}
	// �� á���� �ƴ��� Ȯ��.
	bool isFull() { return (sz == MAX_SIZE); }
	// stack�� ����ϰ� ������ ���� �ʿ�� ������, size=0���� ���������Ѵ�.
	void clear() { sz = 0; }
};

#endif
