#ifndef _QUADTREEFASTPOOLED_H_
#define _QUADTREEFASTPOOLED_H_

#define MAX_POOL 131072 // object pool의 최대 원소갯수.
#define SELECT_CHILD(nd_,elem_,flag_ew,flag_sn) flag_ew = (nd_->bound.nw.x + nd_->bound.se.x) < elem_->pt.x * 2 ? 1 : 0; flag_sn = (nd_->bound.nw.y + nd_->bound.se.y) < elem_->pt.y * 2 ? 1 : 0;
#define SELECT_CHILD_PT(nd_,pt_,flag_ew,flag_sn) flag_ew = (nd_->bound.nw.x + nd_->bound.se.x) < pt_.x * 2 ? 1 : 0; flag_sn = (nd_->bound.nw.y + nd_->bound.se.y) < pt_.y * 2 ? 1 : 0;
#define INBOUND(nd_,elem_q_) ((nd_->bound.nw.x < elem_q_->pt.x) && (elem_q_->pt.x < nd_->bound.se.x) && (nd_->bound.nw.y < elem_q_->pt.y) && (elem_q_->pt.y < nd_->bound.se.y))
#define INBOUND_PT(nd_,pt_q_) ((nd_->bound.nw.x < pt_q_.x) && (pt_q_.x < nd_->bound.se.x) && (nd_->bound.nw.y < pt_q_.y) && (pt_q_.y < nd_->bound.se.y))

#include <iostream>
#include <vector>
#include <memory>
#include "CommonStruct.h"
#include "CommonFunction.h"
#include "ObjectPool.h"
using namespace std;

// Point2, Bound, Node, NodeStack, ElemOverlap, Elem : CommonStruct에 있다.
// Node와 Elem에 대해서 object pooling을 수행한다.
// 상속 모호성을 없애기 위해, 각  function으로 접근 할 때 한정자를 제대로 써줘야한다.
class QuadTreeFastPooled
{
public:
	Node* root;// root node이다.
private:
	ObjectPool<Node>* objpool_node;// 두가지 objectpool이다.
	ObjectPool<Elem>* objpool_elem;

	// root node 에서는 전체 rectangular size를 가지고있다. 
	// 어차피 65536 보다 큰 사이즈의 이미지를 쓰지 않으므로, ushort를 쓴다.
	ushort width;
	ushort height;

	// 최대 깊이.
	ushort MAX_DEPTH;
	float eps; // approximated quadtree에 쓰이는 epsilon 마진. 허용오차.
	float scale; // 1-eps 이다.
	float scale2; // (1-eps)^2 -> BOB랑 BWB에 쓰면 되나 ?

				   // point들을 담고있는 vector이다. point 갯수만큼 길이를 가진다.
	vector<Elem*> elem_vector;
	vector<Node*> node_vector; // 모든 노드들을 담고있는 vector이다.

	size_t n_elem;// point 갯수랑 동일하다.
public:
	// 자주 쓰이는 변수. 선할당 해놓고 쓰자! 
	float min_dist;
	float dist_temp;
	float dist_thres;
	// recursive searching을 위한 Nodestack
	PointerStack<Node> stack;

	// private methods
private:
	// insert와 관련된 함수들
	//bool inBoundary();
	// 포인터의 레퍼런스로 하면, 포인터 8바이트의 복사가 일어나지 않는다.
	void selectChild(Node*& nd_, Elem*& elem_, int& flag_ew, int& flag_sn);
	void selectChildpt(Node*& nd_, chk::Point2f& pt_, int& flag_ew, int& flag_sn); // overload

	void calcQuadrantRect(Node*& parent_, Node*& child_, int& flag_ew, int& flag_sn);
	void insert(const int& depth_, Elem* elem_q_, Node* nd_); // non-recursive version. (DFS에 대한 stack 이용)

															  // search와 관련된 함수들
	void findNearestElem(Node*& nd_, Elem*& elem_q_, int& id_matched, Node*& nd_matched_);
	void findNearestElempt(Node*& nd_, chk::Point2f& pt_q_, int& id_matched, Node*& nd_matched_);
	bool BWBTest(Node*& nd_, Elem*& elem_q_);// Ball within bound
	bool BWBTest(Node*& nd_, chk::Point2f& pt_q_);// Ball within bound

	bool BOBTest(Node*& nd_, Elem*& elem_q_);// Ball Overlap bound
	bool BOBTest(Node*& nd_, chk::Point2f& pt_q_);// Ball Overlap bound

	bool inBound(Node*& nd_, Elem*& elem_q_); // in boundary test
	bool inBoundpt(Node*& nd_, chk::Point2f& pt_q_);

public:
	// for multiple,
	void insertPublic(const chk::Point2f& pt_, const int& id_, const int& idx_pts_);

public:
	int searchNNSingleQuery(chk::Point2f& pt_q_, Node*& node_matched_); // Stack version. non-recursion.
	int searchNNSingleQueryCached(chk::Point2f& pt_q_, Node*& node_cached_and_matched_);

	void searchNNMultipleQueries(vector<chk::Point2f>& pts, vector<int>& index_matched, vector<Node*>& nodes_matched);
	void searchNNMultipleQueriesCached(vector<chk::Point2f>& pts, vector<int>& index_matched, vector<Node*>& nodes_cached_and_matched);

	// public methods
public:
	// 생성자
	QuadTreeFastPooled();
	QuadTreeFastPooled(
        const vector<chk::Point2f>& pts_in_,
		const int& n_rows_, const int& n_cols_,
		const int& max_depth_,
		const float& eps_,
		const float& dist_thres_,
		ObjectPool<Node>* objpool_node_, ObjectPool<Elem>* objpool_elem_);
	QuadTreeFastPooled( // pts 없이, 나머지 모두 초기화 하는 버전. (multiple 위해)
		const int& n_rows_, const int& n_cols_,
        const int& max_depth_,
        const float& eps_,
        const float& dist_thres_,
		ObjectPool<Node>* objpool_node_, ObjectPool<Elem>* objpool_elem_);

	// 소멸자. 모든 메모리 뿌수자.
	~QuadTreeFastPooled();

	// 모든 노드 정보를 보여준다.
	void showAllNodes();

};

/*
* ----------------------------------------------------------------------------
*                                IMPLEMENTATION
* ----------------------------------------------------------------------------
*/
QuadTreeFastPooled::QuadTreeFastPooled() {
	this->root = nullptr;
	printf("QuadtreePooled fast initiates.\n");
};

QuadTreeFastPooled::~QuadTreeFastPooled() {

	// 복잡한 점 : new 와 malloc이 섞여있으면 안된다 ... new - delete / malloc - free.
	for (int i = 0; i < node_vector.size(); i++) {
		//node_temp = node_vector[i];
		//delete elem_temp;
		node_vector[i]->doDefault(); // 내용을 초기화해주고, 반납한다. 안그러면 다음번 사용때 이미 할당되어있는것처럼 인식된다.
									 //objpool_node->returnObject(node_vector[i]);
	}
	for (int i = 0; i < elem_vector.size(); i++) {
		//elem_vector[i]->next = nullptr;
		//elem_vector[i]->id = -1;
		//elem_temp = elem_vector[i];
		//delete elem_temp;
		elem_vector[i]->doDefault();
		//objpool_elem->returnObject(elem_vector[i]);
	}
	node_vector.resize(0);
	elem_vector.resize(0);
	//objpool_elem->showRemainedMemory();
	//objpool_node->showRemainedMemory();
	objpool_elem->doDefault();
	objpool_node->doDefault();
	// printf("tree error가 나면 이부분일 가능성이 높다.\n");
};


QuadTreeFastPooled::QuadTreeFastPooled(
	const int& n_rows_, const int&  n_cols_,
    const int&  max_depth_,
    const float& eps_,
    const float& dist_thres_,
	ObjectPool<Node>* objpool_node_, ObjectPool<Elem>* objpool_elem_)
{
	objpool_node = objpool_node_;
	objpool_elem = objpool_elem_;

	// 우선, root부터 초기화 한다.
	// root = new Node(nullptr, 0, false); // root는 당연히~ parent가 없다.
	root = objpool_node->getObject();
	root->parent = nullptr; // root니까 당연히 parent가 없다.
	root->first_child = nullptr;
	root->isvisited = false;
	root->isleaf = false; // root는 시작될때 당연히 유일한 leaf이다.
	root->depth = 0; // 깊이는 0이다.
	root->bound.nw = chk::Point2us(0, 0); // root의 바운더리 설정.
	root->bound.se = chk::Point2us(n_cols_, n_rows_);

	// parent, first_child메모리 초기화, isleaf=0, header_elem = -1;
	node_vector.push_back(root);

	// root->first_child = (Node*)malloc(sizeof(Node) * 4);
	root->first_child = objpool_node->getObjectQuadruple();
	node_vector.push_back(root->first_child);
	node_vector.push_back(root->first_child + 1);
	node_vector.push_back(root->first_child + 2);
	node_vector.push_back(root->first_child + 3);

	root->initializeChildren(); // 혹시모르니, 4개의 child를 할당 안된 상태로 초기화 한다.

								// 파라미터 초기화. 
	MAX_DEPTH = max_depth_; // 최대 깊이.
	width = n_cols_; // 이미지 사이즈
	height = n_rows_;
	eps = eps_; // 허용 approximate params
	scale = 1.0 - eps;
	scale2 = scale*scale;

	// 자주쓰는 동적 할당 변수 정의.
	dist_thres = dist_thres_*dist_thres_; // 제곱 거리를 이용한다.
	dist_temp = 0.0f;
	min_dist = 1e15f;
};

QuadTreeFastPooled::QuadTreeFastPooled(
    const vector<chk::Point2f>& pts_in_,
    const int& n_rows_, const int& n_cols_,
    const int& max_depth_,
    const float& eps_,
    const float& dist_thres_,
    ObjectPool<Node>* objpool_node_, ObjectPool<Elem>* objpool_elem_)
{
	objpool_node = objpool_node_;
	objpool_elem = objpool_elem_;

	// 우선, root부터 초기화 한다.
	// root = new Node(nullptr, 0, false); // root는 당연히~ parent가 없다.
	root = objpool_node->getObject();

	root->parent = nullptr;
	root->first_child = nullptr;
	root->isvisited = false;
	root->isleaf = false; // root는 시작될때 당연히 유일한 leaf이다.
	root->depth = 0; // 깊이는 0이다.
	root->bound.nw = chk::Point2us(0, 0); // root의 바운더리 설정.
	root->bound.se = chk::Point2us(n_cols_, n_rows_);

	// parent, first_child메모리 초기화, isleaf=0, header_elem = -1;
	node_vector.push_back(root);

	// root->first_child = (Node*)malloc(sizeof(Node) * 4);
	// Four children are allocated on the contiguous memories each other.
	root->first_child = objpool_node->getObjectQuadruple();
	node_vector.push_back(root->first_child);
	node_vector.push_back(root->first_child + 1);
	node_vector.push_back(root->first_child + 2);
	node_vector.push_back(root->first_child + 3);

	root->initializeChildren();

	// 파라미터 초기화 
	MAX_DEPTH = max_depth_; // 최대 깊이.

	// 이미지 사이즈
	width  = n_cols_;
	height = n_rows_;

	// 허용 approximate params
	eps = eps_;
	scale = 1.0 - eps;
	scale2 = scale*scale;

	// 자주쓰는 동적 할당 변수 정의.
	dist_thres = dist_thres_*dist_thres_; // 제곱 거리다.
	dist_temp  = 0;
	min_dist   = 1e15;

	// elem 초기화
	n_elem = pts_in_.size();
	Elem* elem_temp = nullptr;
	for (int i = 0; i < n_elem; i++) {
		// = new Elem(points_input_[i]);
		elem_temp = objpool_elem->getObject();
		//elem_temp = new Elem();
		elem_temp->pt = pts_in_[i];
		elem_temp->id = i;
		elem_temp->idx_pts = i; // single case 에서는 id와 idx_pts 가 같다.
		elem_temp->next = nullptr;
		elem_vector.push_back(elem_temp);

		// 해당 elem을 트리에 넣는다.
		insert(0, elem_temp, root);
	}

};

void QuadTreeFastPooled::selectChild(Node*& nd_, Elem*& elem_, int& flag_ew, int& flag_sn) {
	//Point<ushort> center;
	//center.x = ((nd_->bound.nw.x + nd_->bound.se.x) >> 1);
	//center.y = ((nd_->bound.nw.y + nd_->bound.se.y) >> 1);
	flag_ew = (nd_->bound.nw.x + nd_->bound.se.x) < elem_->pt.x * 2 ? 1 : 0; // 동쪽이면 1
	flag_sn = (nd_->bound.nw.y + nd_->bound.se.y) < elem_->pt.y * 2 ? 1 : 0; // 남쪽이면 1
};
void QuadTreeFastPooled::selectChildpt(Node*& nd_, chk::Point2f& pt_, int& flag_ew, int& flag_sn) {
	//Point<ushort> center;
	//center.x = ((nd_->bound.nw.x + nd_->bound.se.x) >> 1);
	//center.y = ((nd_->bound.nw.y + nd_->bound.se.y) >> 1);
	flag_ew = (nd_->bound.nw.x + nd_->bound.se.x) < pt_.x * 2 ? 1 : 0; // 동쪽이면 1
	flag_sn = (nd_->bound.nw.y + nd_->bound.se.y) < pt_.y * 2 ? 1 : 0; // 남쪽이면 1
};

void QuadTreeFastPooled::calcQuadrantRect(
	Node*& parent_, Node*& child_, int& flag_ew, int& flag_sn)
{
	// 사분면 별 중심점과 우측 변 길이, 하단 변 길이를 저장.
	chk::Point2us center;
	center.x = ((parent_->bound.nw.x + parent_->bound.se.x) >> 1);
	center.y = ((parent_->bound.nw.y + parent_->bound.se.y) >> 1);

	if (flag_ew) { // 동쪽이면,
		child_->bound.nw.x = center.x;
		child_->bound.se.x = parent_->bound.se.x;
	}
	else { // 서쪽
		child_->bound.nw.x = parent_->bound.nw.x;
		child_->bound.se.x = center.x;
	}

	if (flag_sn) { // 남쪽이면,
		child_->bound.nw.y = center.y;
		child_->bound.se.y = parent_->bound.se.y;
	}
	else { // 북쪽이면,
		child_->bound.nw.y = parent_->bound.nw.y;
		child_->bound.se.y = center.y;
	}
};

// ==================== 설명 ====================
// - 기본적으로, 들어오는 elem_q_->next = NULL이어야 한다.
// - nd_를 reference 로 전달 할 방법을 찾아보자.
// - child가 하나라도 생겨야 하는 상황이면, 한꺼번에 4개를 연속으로 할당하자.
// non-max depth -> bridge or leaf. (bridge는 다 똑같고, leaf는 nonmax leaf)
// max depth     -> leaf.
//
//                     [쿼리 별 ]
//                     [판단트리]
//                       /    \
//                      /      \
//                     /        \
//         [최대 깊이 O]         [최대 깊이 X]
//          무조건 leaf          leaf or bridge
//          /  \                            /   \
//         /    \                          /     \
//  [active O]   [active X]         [leaf]        [bridge]
//   append    activate+append    bridge로 변경,   그냥 자식으로 간다.
//                                여기 있던 점과
//                                함께 자식으로.
//
//      | 00 | 01 |   | HL | HR |
//      |----|----| = |----|----| : LSB(좌우), MSB(상하)
//      | 10 | 11 |   | BL | BR |
//
// [Leaf 특성]
// - 원소 존재 O (id_elem  > -1)
// [bridge 특성]
// - 원소 존재 X (id_elem == -1)

// [노드가 생성 될 때]
// - 새 노드가 생성 될때는 항상 leaf이기 때문에 생성되는 것임을 명심하자.
// - leaf가 생성 될 때는, 
//   -> isleaf = 1, id_elem = #, center와 half를 절반때려서 넣어줘야함.
//
// [leaf -> bridge]
//   -> node에 들어있던 elem + 새로 온 elem을 현재 node의 자식노드로 travel 해야한다.
//   -> isleaf = 0, id_elem = -1.

// 이것도 사실 stack기반으로 만들 수 있을 것 같은데 ...
void QuadTreeFastPooled::insert(const int& depth_, Elem* elem_q_, Node* nd_) {
	// 1) 최대 깊이가 아닌 경우.
	if (depth_ < MAX_DEPTH) {
		// leaf 혹은 bridge이다.
		if (nd_->isleaf) {
			// leaf인 경우 -> bridge로 바뀌고, 
			// 기존점+쿼리점 둘 다 각각 원하는 곳의 자식쪽으로 간다.
			// 1. isleaf = false. 어차피 bridge될거임.
			nd_->isleaf = false; // bridge 설정 완료.

								 //2 . 지금 node에 있던 elem의 포인터를 뽑아오고, node의 header는 -1로 바꿈.
			Elem* elem_temp = this->elem_vector[nd_->header_elem];
			nd_->header_elem = -1;

			// 3. children의 공간을 할당 (및 일부초기화)를 수행한다. 
			// parent, first_child메모리 초기화, isleaf=0,  header_elem = -1;
			//nd_->first_child = (Node*)malloc(sizeof(Node) * 4);
			nd_->first_child = objpool_node->getObjectQuadruple();
			node_vector.push_back(nd_->first_child);
			node_vector.push_back(nd_->first_child + 1);
			node_vector.push_back(nd_->first_child + 2);
			node_vector.push_back(nd_->first_child + 3);
			nd_->initializeChildren();

			// 4. elem_temp을 travel.
			int flag_ew, flag_sn, num_child; // 동쪽이면 flag_ew: 1, 남쪽이면 flag_sn: 1
			SELECT_CHILD(nd_, elem_temp, flag_ew, flag_sn)
			num_child = (flag_sn << 1) + flag_ew;
			Node* child = (nd_->first_child + num_child); // 선택된 자식 방향.

			child->isleaf = true; // 원래 bridge였으므로, 자식 체크 필요 X. 걍 leaf로 만들어준다.
			child->depth = depth_ + 1;
			// 자식 노드가 된 녀석의 모서리 값을 계산해준다.
			calcQuadrantRect(nd_, child, flag_ew, flag_sn);
			// header_elem의 값을 지금의 id로 주고, 현재 노드의 element 1개로 늘린다.
			child->header_elem = elem_temp->id;


			// 5. elem_q_을 travel. 여기서는 원하는곳에 자식이 있는지 없는지 확인해야한다.
            SELECT_CHILD(nd_, elem_q_, flag_ew, flag_sn)
			num_child = (flag_sn << 1) + flag_ew;
			child = (nd_->first_child + num_child);
			// 자식이 leaf인지, 아니면 아무 할당이 되어있지 않은지 테스트해야함.
			if (!child->isleaf) { // child node가 비어있으면, 해당 노드를 leaf로 만들어주고, elem을 넣음.
				child->isleaf = true; // leaf로 만들어줌.
				child->depth = depth_ + 1;
				calcQuadrantRect(nd_, child, flag_ew, flag_sn); // 자식 노드가 된 녀석의 모서리 값을 계산해준다.
				child->header_elem = elem_q_->id;
			}
			else { // child node가 채워져 있을때 (non-maxdepth leaf일때) 는 그냥 아래로 간다.
				insert(depth_ + 1, elem_q_, child);
			}
		}
		else {
			// non-maxdepth bridge인 경우 -> 바로 자식으로 가면 된다. 
			// 원하는 곳에 자식이 있는 경우 / 없는 경우로 나뉨.
			int flag_ew, flag_sn, num_child; // 동쪽이면 flag_ew: 1, 남쪽이면 flag_sn: 1
            SELECT_CHILD(nd_, elem_q_, flag_ew, flag_sn)
			num_child = (flag_sn << 1) + flag_ew;
			Node* child = nd_->first_child + num_child;

			// 자식이( bridge 또는 leaf인지), 아니면 아무 할당이 되어있지 않은지 테스트해야함.
			// 해당 자식노드가 초기화되어있지 않았다면, depth = 0이다. depth =0은 root빼고 없다.
			if (!child->depth) {
				child->isleaf = true; // leaf로 만들어줌.
				child->depth = depth_ + 1;
				// 자식 노드가 된 녀석의 모서리 값을 계산해준다.
				calcQuadrantRect(nd_, child, flag_ew, flag_sn);
				child->header_elem = elem_q_->id;
			}
			else { // child node가 채워져 있을때는 그냥 아래로 ㄱㄱ
				insert(depth_ + 1, elem_q_, child);
			}
		}
	}
	// 2) 최대 깊이인 경우에는 무조건 1개 이상의 원소를 가진 leaf이기 때문에, 그냥 붙이면된다.
	else {
		// 지금 leaf의 header이다.
		// 다음 것이 없다면,현재 leaf의 elem중 마지막 것을 찾는다.
		Elem* elem_ptr = this->elem_vector[nd_->header_elem];
		while (elem_ptr->next) elem_ptr = elem_ptr->next;
		elem_ptr->next = elem_q_; // next랑 elem_q_랑 연결되었다!
	}
};

void QuadTreeFastPooled::insertPublic(const chk::Point2f& pt_, const int& id_, const int& idx_pts_) {
	//Elem* elem_temp = new Elem();
	Elem* elem_temp = objpool_elem->getObject();
	elem_temp->pt = pt_;
	elem_temp->id = id_; // 내부적으로 매기는 elem의 id임. (in C++, 해당 트리 내에서의 elem 번호이다.)
	elem_temp->idx_pts = idx_pts_; // 외부적으로 매겨진 idx_pts. (in C++, salient_ids 에 해당)
	elem_temp->next = nullptr;
	elem_vector.push_back(elem_temp);

	// insert this element.
	insert(0, elem_temp, root);
};

void QuadTreeFastPooled::findNearestElem(Node*& nd_, Elem*& elem_q_, int& idx_matched, Node*& nd_matched_) {
	Elem* elem_ptr = this->elem_vector[nd_->header_elem]; // 지금 leaf의 header이다.
														  // 현재 점과의 거리를 구하고, 다음 점으로 넘겨준다. 다음 점이 nullptr이면 종료.
	while (elem_ptr) {
		dist_temp = DIST_EUCLIDEAN_PT(elem_ptr->pt, elem_q_->pt);
		if (min_dist > dist_temp) {
			min_dist = dist_temp;
			idx_matched = elem_ptr->idx_pts;
			nd_matched_ = nd_;
		}
		elem_ptr = elem_ptr->next;
	}
};
void QuadTreeFastPooled::findNearestElempt(Node*& nd_, chk::Point2f& pt_q_, int& idx_matched, Node*& nd_matched_) {
	Elem* elem_ptr = this->elem_vector[nd_->header_elem]; // 지금 leaf의 header이다.
														  // 현재 점과의 거리를 구하고, 다음 점으로 넘겨준다. 다음 점이 nullptr이면 종료.
	while (elem_ptr) {
		dist_temp = DIST_EUCLIDEAN_PT(elem_ptr->pt, pt_q_);
		if (min_dist > dist_temp) {
			min_dist = dist_temp;
			idx_matched = elem_ptr->idx_pts;
			nd_matched_ = nd_;
		}
		elem_ptr = elem_ptr->next;
	}
};

// Ball-Within-Bound (node 내부에 query 점과 현재 min_dist로 이루어진 원이 들어오는지?)
bool QuadTreeFastPooled::BWBTest(Node*& nd_, Elem*& elem_q_) {
	if (nd_->parent == nullptr) return true; // root node로 왔네?
	float d_hori, d_vert;
	// !!!! 제대로 elem_q를 포함하는 노드로 왔음을 보장해야한다.
	// 왼쪽 벽, 오른쪽 벽 부딪히는 경우 고려.
	if (nd_->bound.nw.x == 0) d_hori = nd_->bound.se.x - elem_q_->pt.x;
	else if (nd_->bound.se.x == this->width) d_hori = elem_q_->pt.x - nd_->bound.nw.x;
	else d_hori = nd_->bound.se.x - elem_q_->pt.x < elem_q_->pt.x - nd_->bound.nw.x ? nd_->bound.se.x - elem_q_->pt.x : elem_q_->pt.x - nd_->bound.nw.x;

	// 위쪽 벽, 아랫쪽 벽 부딪히는 경우 고려.
	if (nd_->bound.nw.y == 0) d_vert = nd_->bound.se.y - elem_q_->pt.y;
	else if (nd_->bound.se.y == this->height) d_vert = elem_q_->pt.y - nd_->bound.nw.y;
	else d_vert = nd_->bound.se.y - elem_q_->pt.y < elem_q_->pt.y - nd_->bound.nw.y ? nd_->bound.se.y - elem_q_->pt.y : elem_q_->pt.y - nd_->bound.nw.y;

	float d_min = d_hori < d_vert ? d_hori : d_vert;
	return (min_dist*scale2 < d_min*d_min);
};
bool QuadTreeFastPooled::BWBTest(Node*& nd_, chk::Point2f& pt_q_) {
	/*float d_a = pt_q_.y - (float)nd_->bound.nw.y;
	float d_b = (float)nd_->bound.se.y - pt_q_.y;
	float d_vert = d_a < d_b ? d_a : d_b;

	float d_c = pt_q_.x - (float)nd_->bound.nw.x;
	float d_d = (float)nd_->bound.se.x - pt_q_.x;
	float d_hori = d_c < d_d ? d_c : d_d;*/
	if (nd_->parent == nullptr) return true;

	float d_hori, d_vert;
	if (nd_->bound.nw.x == 0)
		d_hori = (float)nd_->bound.se.x - pt_q_.x;
	else if (nd_->bound.se.x == this->width)
		d_hori = pt_q_.x - (float)nd_->bound.nw.x;
	else
		d_hori = (float)nd_->bound.se.x - pt_q_.x < pt_q_.x - (float)nd_->bound.nw.x
		? (float)nd_->bound.se.x - pt_q_.x : pt_q_.x - (float)nd_->bound.nw.x;

	if (nd_->bound.nw.y == 0)
		d_vert = nd_->bound.se.y - pt_q_.y;
	else if (nd_->bound.se.y == this->height)
		d_vert = pt_q_.y - nd_->bound.nw.y;
	else
		d_vert = (float)nd_->bound.se.y - pt_q_.y < pt_q_.y - (float)nd_->bound.nw.y
		? (float)nd_->bound.se.y - pt_q_.y : pt_q_.y - (float)nd_->bound.nw.y;

	float d_min = d_hori < d_vert ? d_hori : d_vert;
	// cout << "a,b,c,d: " << d_a << ", " << d_b << ", " << d_c << ", " << d_d << endl;
	return (min_dist*scale2 < d_min*d_min);
};

// node 내부에 점이 존재하는지 확인.
bool QuadTreeFastPooled::inBound(Node*& nd_, Elem*& elem_q_) {
	return ((nd_->bound.nw.x < elem_q_->pt.x)
		&& (elem_q_->pt.x < nd_->bound.se.x)
		&& (nd_->bound.nw.y < elem_q_->pt.y)
		&& (elem_q_->pt.y < nd_->bound.se.y));
};
bool QuadTreeFastPooled::inBoundpt(Node*& nd_, chk::Point2f& pt_q_) {
	return ((nd_->bound.nw.x < pt_q_.x)
		&& (pt_q_.x < nd_->bound.se.x)
		&& (nd_->bound.nw.y < pt_q_.y)
		&& (pt_q_.y < nd_->bound.se.y));
};

bool QuadTreeFastPooled::BOBTest(Node*& nd_, Elem*& elem_q_) {
	// 좌
	float min_dist_scale = min_dist*scale2;
	if (elem_q_->pt.x < nd_->bound.nw.x)
		// 좌상
		if (elem_q_->pt.y < nd_->bound.nw.y)
			return min_dist_scale >
			(elem_q_->pt.x - nd_->bound.nw.x)*(elem_q_->pt.x - nd_->bound.nw.x)
			+ (elem_q_->pt.y - nd_->bound.nw.y)*(elem_q_->pt.y - nd_->bound.nw.y);
	// 좌중
		else if (elem_q_->pt.y < nd_->bound.se.y)
			return min_dist_scale >
			(elem_q_->pt.x - nd_->bound.nw.x)*(elem_q_->pt.x - nd_->bound.nw.x);
	// 좌하
		else
			return min_dist_scale >
			(elem_q_->pt.x - nd_->bound.nw.x)*(elem_q_->pt.x - nd_->bound.nw.x)
			+ (elem_q_->pt.y - nd_->bound.se.y)*(elem_q_->pt.y - nd_->bound.se.y);
	// 중
	else if (elem_q_->pt.x < nd_->bound.se.x)
		// 중상
		if (elem_q_->pt.y < nd_->bound.nw.y)
			return min_dist_scale >
			(elem_q_->pt.y - nd_->bound.nw.y)*(elem_q_->pt.y - nd_->bound.nw.y);
	// 중중은 없다.
		else if (elem_q_->pt.y < nd_->bound.se.y)
			return true; // 무조건 겹친다.
						 // 중하
		else
			return min_dist_scale >
			(elem_q_->pt.y - nd_->bound.se.y)*(elem_q_->pt.y - nd_->bound.se.y);
	// 우
	else
		// 우상
		if (elem_q_->pt.y < nd_->bound.nw.y)
			return min_dist_scale >
			(elem_q_->pt.x - nd_->bound.se.x)*(elem_q_->pt.x - nd_->bound.se.x)
			+ (elem_q_->pt.y - nd_->bound.nw.y)*(elem_q_->pt.y - nd_->bound.nw.y);
	// 우중
		else if (elem_q_->pt.y < nd_->bound.se.y)
			return min_dist_scale >
			(elem_q_->pt.x - nd_->bound.se.x)*(elem_q_->pt.x - nd_->bound.se.x);
	// 우하
		else
			return min_dist_scale >
			(elem_q_->pt.x - nd_->bound.se.x)*(elem_q_->pt.x - nd_->bound.se.x)
			+ (elem_q_->pt.y - nd_->bound.se.y)*(elem_q_->pt.y - nd_->bound.se.y);
};
//bool QuadTreeFastPooled::BOBTest(Node*& nd_, chk::Point2f& pt_q_) {
//	// 그 node의 밖에 있다고 가정한다.
//	// 좌
//	float min_dist_scale = min_dist*scale2;
//	if (pt_q_.x < nd_->bound.nw.x)
//		// 좌상
//		if (pt_q_.y < nd_->bound.nw.y)
//			return min_dist_scale >
//			(pt_q_.x - nd_->bound.nw.x)*(pt_q_.x - nd_->bound.nw.x)
//			+ (pt_q_.y - nd_->bound.nw.y)*(pt_q_.y - nd_->bound.nw.y);
//	// 좌중
//		else if (pt_q_.y < nd_->bound.se.y)
//			return min_dist_scale >
//			(pt_q_.x - nd_->bound.nw.x)*(pt_q_.x - nd_->bound.nw.x);
//	// 좌하
//		else
//			return min_dist_scale >
//			(pt_q_.x - nd_->bound.nw.x)*(pt_q_.x - nd_->bound.nw.x)
//			+ (pt_q_.y - nd_->bound.se.y)*(pt_q_.y - nd_->bound.se.y);
//	// 중
//	else if (pt_q_.x < nd_->bound.se.x)
//		// 중상
//		if (pt_q_.y < nd_->bound.nw.y)
//			return min_dist_scale >
//			(pt_q_.y - nd_->bound.nw.y)*(pt_q_.y - nd_->bound.nw.y);
//	// 중중은 없다. ( 내부에 위치하는 경우이며, 이때는 BOB 무조건 true이다. )
//		else if (pt_q_.y < nd_->bound.se.y)
//			return true;
//	// 중하
//		else
//			return min_dist_scale >
//			(pt_q_.y - nd_->bound.se.y)*(pt_q_.y - nd_->bound.se.y);
//	// 우
//	else
//		// 우상
//		if (pt_q_.y < nd_->bound.nw.y)
//			return min_dist_scale >
//			(pt_q_.x - nd_->bound.se.x)*(pt_q_.x - nd_->bound.se.x)
//			+ (pt_q_.y - nd_->bound.nw.y)*(pt_q_.y - nd_->bound.nw.y);
//	// 우중
//		else if (pt_q_.y < nd_->bound.se.y)
//			return min_dist_scale >
//			(pt_q_.x - nd_->bound.se.x)*(pt_q_.x - nd_->bound.se.x);
//	// 우하
//		else
//			return min_dist_scale >
//			(pt_q_.x - nd_->bound.se.x)*(pt_q_.x - nd_->bound.se.x)
//			+ (pt_q_.y - nd_->bound.se.y)*(pt_q_.y - nd_->bound.se.y);
//};

bool QuadTreeFastPooled::BOBTest(Node*& nd_, chk::Point2f& pt_q_) {
    // 그 node의 밖에 있다고 가정한다.
    // 좌
    float min_dist_scale = min_dist*scale2;
    float close_x = pt_q_.x;
    float close_y = pt_q_.y;

    if (pt_q_.x < nd_->bound.nw.x) close_x = nd_->bound.nw.x;
    else if (pt_q_.x > nd_->bound.se.x) close_x = nd_->bound.se.x;

    if (pt_q_.y < nd_->bound.nw.y) close_y = nd_->bound.nw.y;
    else if (pt_q_.y > nd_->bound.se.y) close_y = nd_->bound.se.y;

    return (((pt_q_.x-close_x)*(pt_q_.x - close_x) + (pt_q_.y - close_y)*(pt_q_.y - close_y)) < min_dist_scale);
};

// stack기반의 NN search이다. 가장 가까운 쌍을 찾는 과정.
int QuadTreeFastPooled::searchNNSingleQuery(chk::Point2f& pt_q_, Node*& node_matched_) {
	// root를 stack 처음에 넣는다.
	stack.push(this->root);

	// 거리 변수 초기화
	min_dist    = 1e20;
	dist_temp   = 0;
	Node* node  = nullptr;
	Node* child = nullptr;
	int flag_ew, flag_sn;
	int idx_elem_matched = -1;
	node_matched_ = nullptr;

	while (stack.size > 0) {
		node = stack.top();
		stack.pop();
		// leaf node이면 가장 가까운 점을 찾는다.
		if (node->isleaf) {
			stack.total_access++;
			findNearestElempt(node, pt_q_, idx_elem_matched, node_matched_);
			if (INBOUND_PT(node, pt_q_) && BWBTest(node, pt_q_)) break;
		}
		else { // leaf node가 아니면, 어디를 가야 할 지 결정한다.
			   // 각각의 child가 존재하지 않거나, BOB 만족하지 않으면, 자식 노드로 갈 필요가 없다.
			stack.total_access++;
            SELECT_CHILD_PT(node, pt_q_, flag_ew, flag_sn) // 가장 가능성있는 node를 선택한다. 여기에 leaf가 없을수도 있다.
            
            // child->depth가 곧 child의 존재여부를 말한다.
			child = node->first_child + (!flag_sn << 1) + flag_ew;
			if (child->depth && BOBTest(child, pt_q_)) stack.push(child);
			child = node->first_child + (flag_sn << 1) + !flag_ew;
			if (child->depth && BOBTest(child, pt_q_)) stack.push(child);
			child = node->first_child + (!flag_sn << 1) + !flag_ew;
			if (child->depth && BOBTest(child, pt_q_)) stack.push(child);
			// 선택된 child. 점이 없을수도 있다.
			child = node->first_child + (flag_sn << 1) + flag_ew;
			if (child->depth && BOBTest(child, pt_q_)) stack.push(child);
		}
	}

	stack.clear(); // stack 초기화. (메모리 내용만 초기화)
	if (min_dist > dist_thres) { // 정해준 dist_thres보다 멀면 버린다.
        idx_elem_matched = -2; // 매칭되지 않았으므로, point index를 -2로 설정한다.
		node_matched_ = this->root; // 매칭되지 않았으므로, cached node를 root로 설정한다.
	}
	return idx_elem_matched;
};

int QuadTreeFastPooled::searchNNSingleQueryCached(chk::Point2f& pt_q_, Node*& node_cached_and_matched_) {
	// 0) 거리 변수 초기화
	min_dist = 1e15f;
	dist_temp = 0.0f;
	stack.total_access++; // node 방문했으니, counter 올림.
	int idx_elem_matched = -2; // 매칭된 점의 index
	Node* node = node_cached_and_matched_; // 우선 cached node를 불러온다. node_cached_and_matched로 매칭 결과를 다시 출력해준다.

										   // 일단, query가 root 내부 (이미지 내부)에 위치하는지부터 확인하자.
	if (INBOUND_PT(this->root, pt_q_)) {
		// root가 들어왔으면 이전에 매칭된 것이 없다는 의미. 일반적인 searchNN을 실행.
		if (node == this->root) {
			idx_elem_matched = searchNNSingleQuery(pt_q_, node_cached_and_matched_);
			goto flagFinish;
		}
		// 매칭된 점이 존재하는 leaf node의 포인터.
		// 1) nd_cached_내의 점에서 q와 가장 가까운 점을 찾는다. (무조건 leaf여야 한다...)
		// 만일, q가 cached node의 bound 안에 위치하고, BWB까지 만족하면, 그냥 끝!
		// 만약, inbound 인데, BWB만 만족하지 않았던 것이라면, parent로 갔다가 다시 여기로 내려올듯...
		node_cached_and_matched_ = nullptr;
		findNearestElempt(node, pt_q_, idx_elem_matched, node_cached_and_matched_);
		// cached node에서 BWB만족하는 매칭쌍이 있다면, 바로 종료.
		if (INBOUND_PT(node, pt_q_) && BWBTest(node, pt_q_)) {
			goto flagFinish;
		}

		// 2) BWBTest == true 인 bridge까지 올라간다.
		while (true) {
			node = node->parent;
			stack.total_access++;
			if (BWBTest(node, pt_q_)) break;
			//if (inBound(node, pt_q_)) break; // PWB Test only.
			if (node->parent == nullptr) break; // root
		}

		// 3) 현재 위치한 node부터 아래로 정방향 search 시작.
		stack.push(node);
		// 거리 변수 초기화
		int flag_ew, flag_sn;
		Node* child = nullptr;
		while (stack.size > 0) {
			node = stack.top();
			stack.pop();
			// leaf node이면 가장 가까운 점을 찾는다.
			if (node->isleaf) {
				stack.total_access++;
				findNearestElempt(node, pt_q_, idx_elem_matched, node_cached_and_matched_);
				if (INBOUND_PT(node, pt_q_) && BWBTest(node, pt_q_)) goto flagFinish;
				// if (inBound(node, pt_q_)) goto flagFinish; // PWB Test only.
			}
			// leaf node가 아니면, 어디를 가야 할 지 결정한다.
			else {
				// 현재의 node가 BOB 만족하지 않으면, 자식 노드로 갈 필요가 없다.
				// BOB를 만족하면, 자식 노드로 향한다.
				stack.total_access++;
                SELECT_CHILD_PT(node, pt_q_, flag_ew, flag_sn)
				// 지금 가려는 child가 아닌 children이 나중에 연산되어야 하니, 먼저 stack에 들어감.
				// 먼저 != nullptr && BOB pass 인 점들만 넣음.
				// 다른 child들, 초기화 된 놈은 depth가 무조건 >0 이다.
				// 초기화 되지 않은 놈은 0으로 되어있다. root 제외하고는 0을 가질수없다.
				child = node->first_child + (!flag_sn << 1) + flag_ew;
				if (child->depth && BOBTest(child, pt_q_)) stack.push(child);
				child = node->first_child + (flag_sn << 1) + !flag_ew;
				if (child->depth && BOBTest(child, pt_q_)) stack.push(child);
				child = node->first_child + (!flag_sn << 1) + !flag_ew;
				if (child->depth && BOBTest(child, pt_q_)) stack.push(child);
				// 지금 child.
				child = node->first_child + (flag_sn << 1) + flag_ew;
				if (child->depth && BOBTest(child, pt_q_)) stack.push(child);
			}
		}
	}
	else { // 내부에 위치하지 않으면, 매칭된 것이 없다!
		node_cached_and_matched_ = this->root;
		idx_elem_matched = -2;
		goto flagFinish;
	}


	// 스택을 다 사용하였으니, top을 0으로 만들어준다. (실제로 데이터를 지울 필요는 없다.)
flagFinish:
	stack.clear();
	if (min_dist > dist_thres) {
		node_cached_and_matched_ = this->root;
		idx_elem_matched = -2;
	}
	return idx_elem_matched;
};


void QuadTreeFastPooled::searchNNMultipleQueries(vector<chk::Point2f>& pts, vector<int>& index_matched, vector<Node*>& nodes_matched) {
	int n_pts = (int)pts.size();

	for (int i = 0; i < n_pts; i++) {
		if (INBOUND_PT(this->root, pts[i])) { // 이미지 내부에 있으면 search
			index_matched[i] = searchNNSingleQuery(pts[i], nodes_matched[i]);
		}
		else { // 이미지 내부가 아니면, 기각.
			index_matched[i] = -2;
			nodes_matched[i] = this->root;
		}
	}
};

void QuadTreeFastPooled::searchNNMultipleQueriesCached(
	vector<chk::Point2f>& pts,
	vector<int>& index_matched, vector<Node*>& nodes_cached_and_matched) {
	int n_pts = (int)pts.size();
	for (int i = 0; i < n_pts; i++) {
		if (INBOUND_PT(this->root, pts[i])) { // 이미지 내부에 있으면 search
			index_matched[i] = searchNNSingleQueryCached(pts[i], nodes_cached_and_matched[i]); // cached node는 입력.
		}
		else { // 이미지 내부가 아니면, 기각.
			index_matched[i] = -2;
			nodes_cached_and_matched[i] = this->root;
		}
	}
};

void QuadTreeFastPooled::showAllNodes() {
	// 첫번째는 무조건 root node이다.
	Node* node_temp = this->root;
	node_temp->showNodeSpec();
	for (int i = 1; i < node_vector.size(); i++) {
		node_temp = node_vector[i];
		if (node_temp->parent != nullptr)
			node_temp->showNodeSpec();
	}
};

#endif