#ifndef _OBJECTPOOL_H_
#define _OBJECTPOOL_H_

#include <iostream>
#include <memory>

// Attaboy!

// 해당 ObjectPool은 연속된 공간을 할당하고, stack 기반으로 메모리를 할당하는 구조이다.
// 임의의 데이터타입 T의 pool을 생성하는 함수.
// new T() 기본 생성자를 정의해줘야한다.

// 거대한 memory chunk. static으로 정의하면 ObjectPool이라는 class를 상속하는 모든 
// class는 같은 memory_chunk라는 변수를 상속받아 공유한다 ...
// 20200215 - 굳이 상속하지 않아도 되는걸? 어차피 static member는 그냥 바로 생기니까;
// 걍 객체로 가지도록 하자.
// 총 N개의 object를 저장한다고 하면, N-1, N-2, ... , 2, 1, 0 순으로 스택에 들어간다.
// 20200221 queue 쓰면 안된다. 중복 반납;... 어차피 지금의 tree 구조에서는 순서대로 쓰고, 한번에 반납하니까
// 걍 index로 처리하고, doDefault 형식으로 가자. 
template <typename T>
class ObjectPool {
private:
	T* _memory_chunk; // 같은 class를 template으로 받으면, 무조건 _memory_chunk를 공유한다. 
	int max_num_obj;
	int index_now;

public:
	// 생성자
	ObjectPool(int max_num_object) {
		max_num_obj = max_num_object;
		index_now = 0;
		_memory_chunk = (T*)malloc(sizeof(T)*max_num_obj);
		printf("object size: %d / allocated # of objects: %d / total memory consumption: %d [Mbytes]\n",
			(int)sizeof(T), max_num_obj, (int)(sizeof(T)*max_num_obj) / (1024 * 1024));
	};

	// memory chunk의 앞부분을 가져온다.
	T* getObject() {
		if (index_now < max_num_obj) {
			T* ptr = (_memory_chunk + index_now);
			++index_now;
			return ptr;
		}
		else {
			printf("ERROR: no object pool is remaining.\n");
			return nullptr;
		}
	}

	T* getObjectQuadruple() {
		if (index_now < max_num_obj) {
			T* ptr = (_memory_chunk + index_now);
			index_now += 4;
			return ptr;
		}
		else {
			printf("ERROR: no object pool is remaining.\n");
			return nullptr;
		}
	}

	void doDefault() { // 단순히 index를 0으로 바꿔줄 뿐! 
					   // 모두 사용 후, 따로 메모리 해제 할 필요없이 index만 0으로 옮겨주면 바로 쓸 수 있다.
		index_now = 0;
	};

	// 소멸자
	~ObjectPool() {
		if (_memory_chunk != nullptr) {
			free(_memory_chunk);
			printf("\n Memory chunk in object pool is successfully returned.\n");
		}
		else {
			printf("\n Already empty.\n");
		}
	};

	void showRemainedMemory() {
		printf("remained mem: %d\n", max_num_obj - index_now);
	};

	// 원래는 new, delete를 overloading 할려했는데 ... 안했다 걍
	//	T* getMemory();
	//void retunrMemory(T* ptr);
	/*void* operator new(size_t size)
	{
	return GObjectPool::getInstance()->AllocObject(size);
	}
	void operator delete(void* pointer, size_t size)
	{
	GObjectPool::getInstance()->FreeObject(pointer, size);
	}*/
};

// // 정적멤버변수들 초기화.
// template <typename T>
// T* ObjectPool<T>::_memory_chunk = nullptr;
// template <typename T>
// size_t ObjectPool<T>::_used_count = 0;
// template <typename T>
// std::queue<T*> ObjectPool<T>::_obj_queue;

#endif