#ifndef _OBJECTPOOL_H_
#define _OBJECTPOOL_H_

#include <iostream>
#include <memory>

// Attaboy!

// �ش� ObjectPool�� ���ӵ� ������ �Ҵ��ϰ�, stack ������� �޸𸮸� �Ҵ��ϴ� �����̴�.
// ������ ������Ÿ�� T�� pool�� �����ϴ� �Լ�.
// new T() �⺻ �����ڸ� ����������Ѵ�.

// �Ŵ��� memory chunk. static���� �����ϸ� ObjectPool�̶�� class�� ����ϴ� ��� 
// class�� ���� memory_chunk��� ������ ��ӹ޾� �����Ѵ� ...
// 20200215 - ���� ������� �ʾƵ� �Ǵ°�? ������ static member�� �׳� �ٷ� ����ϱ�;
// �� ��ü�� �������� ����.
// �� N���� object�� �����Ѵٰ� �ϸ�, N-1, N-2, ... , 2, 1, 0 ������ ���ÿ� ����.
// 20200221 queue ���� �ȵȴ�. �ߺ� �ݳ�;... ������ ������ tree ���������� ������� ����, �ѹ��� �ݳ��ϴϱ�
// �� index�� ó���ϰ�, doDefault �������� ����. 
template <typename T>
class ObjectPool {
private:
	T* _memory_chunk; // ���� class�� template���� ������, ������ _memory_chunk�� �����Ѵ�. 
	int max_num_obj;
	int index_now;

public:
	// ������
	ObjectPool(int max_num_object) {
		max_num_obj = max_num_object;
		index_now = 0;
		_memory_chunk = (T*)malloc(sizeof(T)*max_num_obj);
		printf("object size: %d / allocated # of objects: %d / total memory consumption: %d [Mbytes]\n",
			(int)sizeof(T), max_num_obj, (int)(sizeof(T)*max_num_obj) / (1024 * 1024));
	};

	// memory chunk�� �պκ��� �����´�.
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

	void doDefault() { // �ܼ��� index�� 0���� �ٲ��� ��! 
					   // ��� ��� ��, ���� �޸� ���� �� �ʿ���� index�� 0���� �Ű��ָ� �ٷ� �� �� �ִ�.
		index_now = 0;
	};

	// �Ҹ���
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

	// ������ new, delete�� overloading �ҷ��ߴµ� ... ���ߴ� ��
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

// // ������������� �ʱ�ȭ.
// template <typename T>
// T* ObjectPool<T>::_memory_chunk = nullptr;
// template <typename T>
// size_t ObjectPool<T>::_used_count = 0;
// template <typename T>
// std::queue<T*> ObjectPool<T>::_obj_queue;

#endif