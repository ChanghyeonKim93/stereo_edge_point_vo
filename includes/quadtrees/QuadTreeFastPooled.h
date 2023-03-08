#ifndef _QUADTREEFASTPOOLED_H_
#define _QUADTREEFASTPOOLED_H_

#define MAX_POOL 131072 // object pool�� �ִ� ���Ұ���.
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

// Point2, Bound, Node, NodeStack, ElemOverlap, Elem : CommonStruct�� �ִ�.
// Node�� Elem�� ���ؼ� object pooling�� �����Ѵ�.
// ��� ��ȣ���� ���ֱ� ����, ��  function���� ���� �� �� �����ڸ� ����� ������Ѵ�.
class QuadTreeFastPooled
{
public:
	Node* root;// root node�̴�.
private:
	ObjectPool<Node>* objpool_node;// �ΰ��� objectpool�̴�.
	ObjectPool<Elem>* objpool_elem;

	// root node ������ ��ü rectangular size�� �������ִ�. 
	// ������ 65536 ���� ū �������� �̹����� ���� �����Ƿ�, ushort�� ����.
	ushort width;
	ushort height;

	// �ִ� ����.
	ushort MAX_DEPTH;
	float eps; // approximated quadtree�� ���̴� epsilon ����. ������.
	float scale; // 1-eps �̴�.
	float scale2; // (1-eps)^2 -> BOB�� BWB�� ���� �ǳ� ?

				   // point���� ����ִ� vector�̴�. point ������ŭ ���̸� ������.
	vector<Elem*> elem_vector;
	vector<Node*> node_vector; // ��� ������ ����ִ� vector�̴�.

	size_t n_elem;// point ������ �����ϴ�.
public:
	// ���� ���̴� ����. ���Ҵ� �س��� ����! 
	float min_dist;
	float dist_temp;
	float dist_thres;
	// recursive searching�� ���� Nodestack
	PointerStack<Node> stack;

	// private methods
private:
	// insert�� ���õ� �Լ���
	//bool inBoundary();
	// �������� ���۷����� �ϸ�, ������ 8����Ʈ�� ���簡 �Ͼ�� �ʴ´�.
	void selectChild(Node*& nd_, Elem*& elem_, int& flag_ew, int& flag_sn);
	void selectChildpt(Node*& nd_, chk::Point2f& pt_, int& flag_ew, int& flag_sn); // overload

	void calcQuadrantRect(Node*& parent_, Node*& child_, int& flag_ew, int& flag_sn);
	void insert(const int& depth_, Elem* elem_q_, Node* nd_); // non-recursive version. (DFS�� ���� stack �̿�)

															  // search�� ���õ� �Լ���
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
	// ������
	QuadTreeFastPooled();
	QuadTreeFastPooled(
        const vector<chk::Point2f>& pts_in_,
		const int& n_rows_, const int& n_cols_,
		const int& max_depth_,
		const float& eps_,
		const float& dist_thres_,
		ObjectPool<Node>* objpool_node_, ObjectPool<Elem>* objpool_elem_);
	QuadTreeFastPooled( // pts ����, ������ ��� �ʱ�ȭ �ϴ� ����. (multiple ����)
		const int& n_rows_, const int& n_cols_,
        const int& max_depth_,
        const float& eps_,
        const float& dist_thres_,
		ObjectPool<Node>* objpool_node_, ObjectPool<Elem>* objpool_elem_);

	// �Ҹ���. ��� �޸� �Ѽ���.
	~QuadTreeFastPooled();

	// ��� ��� ������ �����ش�.
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

	// ������ �� : new �� malloc�� ���������� �ȵȴ� ... new - delete / malloc - free.
	for (int i = 0; i < node_vector.size(); i++) {
		//node_temp = node_vector[i];
		//delete elem_temp;
		node_vector[i]->doDefault(); // ������ �ʱ�ȭ���ְ�, �ݳ��Ѵ�. �ȱ׷��� ������ ��붧 �̹� �Ҵ�Ǿ��ִ°�ó�� �νĵȴ�.
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
	// printf("tree error�� ���� �̺κ��� ���ɼ��� ����.\n");
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

	// �켱, root���� �ʱ�ȭ �Ѵ�.
	// root = new Node(nullptr, 0, false); // root�� �翬��~ parent�� ����.
	root = objpool_node->getObject();
	root->parent = nullptr; // root�ϱ� �翬�� parent�� ����.
	root->first_child = nullptr;
	root->isvisited = false;
	root->isleaf = false; // root�� ���۵ɶ� �翬�� ������ leaf�̴�.
	root->depth = 0; // ���̴� 0�̴�.
	root->bound.nw = chk::Point2us(0, 0); // root�� �ٿ���� ����.
	root->bound.se = chk::Point2us(n_cols_, n_rows_);

	// parent, first_child�޸� �ʱ�ȭ, isleaf=0, header_elem = -1;
	node_vector.push_back(root);

	// root->first_child = (Node*)malloc(sizeof(Node) * 4);
	root->first_child = objpool_node->getObjectQuadruple();
	node_vector.push_back(root->first_child);
	node_vector.push_back(root->first_child + 1);
	node_vector.push_back(root->first_child + 2);
	node_vector.push_back(root->first_child + 3);

	root->initializeChildren(); // Ȥ�ø𸣴�, 4���� child�� �Ҵ� �ȵ� ���·� �ʱ�ȭ �Ѵ�.

								// �Ķ���� �ʱ�ȭ. 
	MAX_DEPTH = max_depth_; // �ִ� ����.
	width = n_cols_; // �̹��� ������
	height = n_rows_;
	eps = eps_; // ��� approximate params
	scale = 1.0 - eps;
	scale2 = scale*scale;

	// ���־��� ���� �Ҵ� ���� ����.
	dist_thres = dist_thres_*dist_thres_; // ���� �Ÿ��� �̿��Ѵ�.
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

	// �켱, root���� �ʱ�ȭ �Ѵ�.
	// root = new Node(nullptr, 0, false); // root�� �翬��~ parent�� ����.
	root = objpool_node->getObject();

	root->parent = nullptr;
	root->first_child = nullptr;
	root->isvisited = false;
	root->isleaf = false; // root�� ���۵ɶ� �翬�� ������ leaf�̴�.
	root->depth = 0; // ���̴� 0�̴�.
	root->bound.nw = chk::Point2us(0, 0); // root�� �ٿ���� ����.
	root->bound.se = chk::Point2us(n_cols_, n_rows_);

	// parent, first_child�޸� �ʱ�ȭ, isleaf=0, header_elem = -1;
	node_vector.push_back(root);

	// root->first_child = (Node*)malloc(sizeof(Node) * 4);
	// Four children are allocated on the contiguous memories each other.
	root->first_child = objpool_node->getObjectQuadruple();
	node_vector.push_back(root->first_child);
	node_vector.push_back(root->first_child + 1);
	node_vector.push_back(root->first_child + 2);
	node_vector.push_back(root->first_child + 3);

	root->initializeChildren();

	// �Ķ���� �ʱ�ȭ 
	MAX_DEPTH = max_depth_; // �ִ� ����.

	// �̹��� ������
	width  = n_cols_;
	height = n_rows_;

	// ��� approximate params
	eps = eps_;
	scale = 1.0 - eps;
	scale2 = scale*scale;

	// ���־��� ���� �Ҵ� ���� ����.
	dist_thres = dist_thres_*dist_thres_; // ���� �Ÿ���.
	dist_temp  = 0;
	min_dist   = 1e15;

	// elem �ʱ�ȭ
	n_elem = pts_in_.size();
	Elem* elem_temp = nullptr;
	for (int i = 0; i < n_elem; i++) {
		// = new Elem(points_input_[i]);
		elem_temp = objpool_elem->getObject();
		//elem_temp = new Elem();
		elem_temp->pt = pts_in_[i];
		elem_temp->id = i;
		elem_temp->idx_pts = i; // single case ������ id�� idx_pts �� ����.
		elem_temp->next = nullptr;
		elem_vector.push_back(elem_temp);

		// �ش� elem�� Ʈ���� �ִ´�.
		insert(0, elem_temp, root);
	}

};

void QuadTreeFastPooled::selectChild(Node*& nd_, Elem*& elem_, int& flag_ew, int& flag_sn) {
	//Point<ushort> center;
	//center.x = ((nd_->bound.nw.x + nd_->bound.se.x) >> 1);
	//center.y = ((nd_->bound.nw.y + nd_->bound.se.y) >> 1);
	flag_ew = (nd_->bound.nw.x + nd_->bound.se.x) < elem_->pt.x * 2 ? 1 : 0; // �����̸� 1
	flag_sn = (nd_->bound.nw.y + nd_->bound.se.y) < elem_->pt.y * 2 ? 1 : 0; // �����̸� 1
};
void QuadTreeFastPooled::selectChildpt(Node*& nd_, chk::Point2f& pt_, int& flag_ew, int& flag_sn) {
	//Point<ushort> center;
	//center.x = ((nd_->bound.nw.x + nd_->bound.se.x) >> 1);
	//center.y = ((nd_->bound.nw.y + nd_->bound.se.y) >> 1);
	flag_ew = (nd_->bound.nw.x + nd_->bound.se.x) < pt_.x * 2 ? 1 : 0; // �����̸� 1
	flag_sn = (nd_->bound.nw.y + nd_->bound.se.y) < pt_.y * 2 ? 1 : 0; // �����̸� 1
};

void QuadTreeFastPooled::calcQuadrantRect(
	Node*& parent_, Node*& child_, int& flag_ew, int& flag_sn)
{
	// ��и� �� �߽����� ���� �� ����, �ϴ� �� ���̸� ����.
	chk::Point2us center;
	center.x = ((parent_->bound.nw.x + parent_->bound.se.x) >> 1);
	center.y = ((parent_->bound.nw.y + parent_->bound.se.y) >> 1);

	if (flag_ew) { // �����̸�,
		child_->bound.nw.x = center.x;
		child_->bound.se.x = parent_->bound.se.x;
	}
	else { // ����
		child_->bound.nw.x = parent_->bound.nw.x;
		child_->bound.se.x = center.x;
	}

	if (flag_sn) { // �����̸�,
		child_->bound.nw.y = center.y;
		child_->bound.se.y = parent_->bound.se.y;
	}
	else { // �����̸�,
		child_->bound.nw.y = parent_->bound.nw.y;
		child_->bound.se.y = center.y;
	}
};

// ==================== ���� ====================
// - �⺻������, ������ elem_q_->next = NULL�̾�� �Ѵ�.
// - nd_�� reference �� ���� �� ����� ã�ƺ���.
// - child�� �ϳ��� ���ܾ� �ϴ� ��Ȳ�̸�, �Ѳ����� 4���� �������� �Ҵ�����.
// non-max depth -> bridge or leaf. (bridge�� �� �Ȱ���, leaf�� nonmax leaf)
// max depth     -> leaf.
//
//                     [���� �� ]
//                     [�Ǵ�Ʈ��]
//                       /    \
//                      /      \
//                     /        \
//         [�ִ� ���� O]         [�ִ� ���� X]
//          ������ leaf          leaf or bridge
//          /  \                            /   \
//         /    \                          /     \
//  [active O]   [active X]         [leaf]        [bridge]
//   append    activate+append    bridge�� ����,   �׳� �ڽ����� ����.
//                                ���� �ִ� ����
//                                �Բ� �ڽ�����.
//
//      | 00 | 01 |   | HL | HR |
//      |----|----| = |----|----| : LSB(�¿�), MSB(����)
//      | 10 | 11 |   | BL | BR |
//
// [Leaf Ư��]
// - ���� ���� O (id_elem  > -1)
// [bridge Ư��]
// - ���� ���� X (id_elem == -1)

// [��尡 ���� �� ��]
// - �� ��尡 ���� �ɶ��� �׻� leaf�̱� ������ �����Ǵ� ������ �������.
// - leaf�� ���� �� ����, 
//   -> isleaf = 1, id_elem = #, center�� half�� ���ݶ����� �־������.
//
// [leaf -> bridge]
//   -> node�� ����ִ� elem + ���� �� elem�� ���� node�� �ڽĳ��� travel �ؾ��Ѵ�.
//   -> isleaf = 0, id_elem = -1.

// �̰͵� ��� stack������� ���� �� ���� �� ������ ...
void QuadTreeFastPooled::insert(const int& depth_, Elem* elem_q_, Node* nd_) {
	// 1) �ִ� ���̰� �ƴ� ���.
	if (depth_ < MAX_DEPTH) {
		// leaf Ȥ�� bridge�̴�.
		if (nd_->isleaf) {
			// leaf�� ��� -> bridge�� �ٲ��, 
			// ������+������ �� �� ���� ���ϴ� ���� �ڽ������� ����.
			// 1. isleaf = false. ������ bridge�ɰ���.
			nd_->isleaf = false; // bridge ���� �Ϸ�.

								 //2 . ���� node�� �ִ� elem�� �����͸� �̾ƿ���, node�� header�� -1�� �ٲ�.
			Elem* elem_temp = this->elem_vector[nd_->header_elem];
			nd_->header_elem = -1;

			// 3. children�� ������ �Ҵ� (�� �Ϻ��ʱ�ȭ)�� �����Ѵ�. 
			// parent, first_child�޸� �ʱ�ȭ, isleaf=0,  header_elem = -1;
			//nd_->first_child = (Node*)malloc(sizeof(Node) * 4);
			nd_->first_child = objpool_node->getObjectQuadruple();
			node_vector.push_back(nd_->first_child);
			node_vector.push_back(nd_->first_child + 1);
			node_vector.push_back(nd_->first_child + 2);
			node_vector.push_back(nd_->first_child + 3);
			nd_->initializeChildren();

			// 4. elem_temp�� travel.
			int flag_ew, flag_sn, num_child; // �����̸� flag_ew: 1, �����̸� flag_sn: 1
			SELECT_CHILD(nd_, elem_temp, flag_ew, flag_sn)
			num_child = (flag_sn << 1) + flag_ew;
			Node* child = (nd_->first_child + num_child); // ���õ� �ڽ� ����.

			child->isleaf = true; // ���� bridge�����Ƿ�, �ڽ� üũ �ʿ� X. �� leaf�� ������ش�.
			child->depth = depth_ + 1;
			// �ڽ� ��尡 �� �༮�� �𼭸� ���� ������ش�.
			calcQuadrantRect(nd_, child, flag_ew, flag_sn);
			// header_elem�� ���� ������ id�� �ְ�, ���� ����� element 1���� �ø���.
			child->header_elem = elem_temp->id;


			// 5. elem_q_�� travel. ���⼭�� ���ϴ°��� �ڽ��� �ִ��� ������ Ȯ���ؾ��Ѵ�.
            SELECT_CHILD(nd_, elem_q_, flag_ew, flag_sn)
			num_child = (flag_sn << 1) + flag_ew;
			child = (nd_->first_child + num_child);
			// �ڽ��� leaf����, �ƴϸ� �ƹ� �Ҵ��� �Ǿ����� ������ �׽�Ʈ�ؾ���.
			if (!child->isleaf) { // child node�� ���������, �ش� ��带 leaf�� ������ְ�, elem�� ����.
				child->isleaf = true; // leaf�� �������.
				child->depth = depth_ + 1;
				calcQuadrantRect(nd_, child, flag_ew, flag_sn); // �ڽ� ��尡 �� �༮�� �𼭸� ���� ������ش�.
				child->header_elem = elem_q_->id;
			}
			else { // child node�� ä���� ������ (non-maxdepth leaf�϶�) �� �׳� �Ʒ��� ����.
				insert(depth_ + 1, elem_q_, child);
			}
		}
		else {
			// non-maxdepth bridge�� ��� -> �ٷ� �ڽ����� ���� �ȴ�. 
			// ���ϴ� ���� �ڽ��� �ִ� ��� / ���� ���� ����.
			int flag_ew, flag_sn, num_child; // �����̸� flag_ew: 1, �����̸� flag_sn: 1
            SELECT_CHILD(nd_, elem_q_, flag_ew, flag_sn)
			num_child = (flag_sn << 1) + flag_ew;
			Node* child = nd_->first_child + num_child;

			// �ڽ���( bridge �Ǵ� leaf����), �ƴϸ� �ƹ� �Ҵ��� �Ǿ����� ������ �׽�Ʈ�ؾ���.
			// �ش� �ڽĳ�尡 �ʱ�ȭ�Ǿ����� �ʾҴٸ�, depth = 0�̴�. depth =0�� root���� ����.
			if (!child->depth) {
				child->isleaf = true; // leaf�� �������.
				child->depth = depth_ + 1;
				// �ڽ� ��尡 �� �༮�� �𼭸� ���� ������ش�.
				calcQuadrantRect(nd_, child, flag_ew, flag_sn);
				child->header_elem = elem_q_->id;
			}
			else { // child node�� ä���� �������� �׳� �Ʒ��� ����
				insert(depth_ + 1, elem_q_, child);
			}
		}
	}
	// 2) �ִ� ������ ��쿡�� ������ 1�� �̻��� ���Ҹ� ���� leaf�̱� ������, �׳� ���̸�ȴ�.
	else {
		// ���� leaf�� header�̴�.
		// ���� ���� ���ٸ�,���� leaf�� elem�� ������ ���� ã�´�.
		Elem* elem_ptr = this->elem_vector[nd_->header_elem];
		while (elem_ptr->next) elem_ptr = elem_ptr->next;
		elem_ptr->next = elem_q_; // next�� elem_q_�� ����Ǿ���!
	}
};

void QuadTreeFastPooled::insertPublic(const chk::Point2f& pt_, const int& id_, const int& idx_pts_) {
	//Elem* elem_temp = new Elem();
	Elem* elem_temp = objpool_elem->getObject();
	elem_temp->pt = pt_;
	elem_temp->id = id_; // ���������� �ű�� elem�� id��. (in C++, �ش� Ʈ�� �������� elem ��ȣ�̴�.)
	elem_temp->idx_pts = idx_pts_; // �ܺ������� �Ű��� idx_pts. (in C++, salient_ids �� �ش�)
	elem_temp->next = nullptr;
	elem_vector.push_back(elem_temp);

	// insert this element.
	insert(0, elem_temp, root);
};

void QuadTreeFastPooled::findNearestElem(Node*& nd_, Elem*& elem_q_, int& idx_matched, Node*& nd_matched_) {
	Elem* elem_ptr = this->elem_vector[nd_->header_elem]; // ���� leaf�� header�̴�.
														  // ���� ������ �Ÿ��� ���ϰ�, ���� ������ �Ѱ��ش�. ���� ���� nullptr�̸� ����.
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
	Elem* elem_ptr = this->elem_vector[nd_->header_elem]; // ���� leaf�� header�̴�.
														  // ���� ������ �Ÿ��� ���ϰ�, ���� ������ �Ѱ��ش�. ���� ���� nullptr�̸� ����.
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

// Ball-Within-Bound (node ���ο� query ���� ���� min_dist�� �̷���� ���� ��������?)
bool QuadTreeFastPooled::BWBTest(Node*& nd_, Elem*& elem_q_) {
	if (nd_->parent == nullptr) return true; // root node�� �Գ�?
	float d_hori, d_vert;
	// !!!! ����� elem_q�� �����ϴ� ���� ������ �����ؾ��Ѵ�.
	// ���� ��, ������ �� �ε����� ��� ���.
	if (nd_->bound.nw.x == 0) d_hori = nd_->bound.se.x - elem_q_->pt.x;
	else if (nd_->bound.se.x == this->width) d_hori = elem_q_->pt.x - nd_->bound.nw.x;
	else d_hori = nd_->bound.se.x - elem_q_->pt.x < elem_q_->pt.x - nd_->bound.nw.x ? nd_->bound.se.x - elem_q_->pt.x : elem_q_->pt.x - nd_->bound.nw.x;

	// ���� ��, �Ʒ��� �� �ε����� ��� ���.
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

// node ���ο� ���� �����ϴ��� Ȯ��.
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
	// ��
	float min_dist_scale = min_dist*scale2;
	if (elem_q_->pt.x < nd_->bound.nw.x)
		// �»�
		if (elem_q_->pt.y < nd_->bound.nw.y)
			return min_dist_scale >
			(elem_q_->pt.x - nd_->bound.nw.x)*(elem_q_->pt.x - nd_->bound.nw.x)
			+ (elem_q_->pt.y - nd_->bound.nw.y)*(elem_q_->pt.y - nd_->bound.nw.y);
	// ����
		else if (elem_q_->pt.y < nd_->bound.se.y)
			return min_dist_scale >
			(elem_q_->pt.x - nd_->bound.nw.x)*(elem_q_->pt.x - nd_->bound.nw.x);
	// ����
		else
			return min_dist_scale >
			(elem_q_->pt.x - nd_->bound.nw.x)*(elem_q_->pt.x - nd_->bound.nw.x)
			+ (elem_q_->pt.y - nd_->bound.se.y)*(elem_q_->pt.y - nd_->bound.se.y);
	// ��
	else if (elem_q_->pt.x < nd_->bound.se.x)
		// �߻�
		if (elem_q_->pt.y < nd_->bound.nw.y)
			return min_dist_scale >
			(elem_q_->pt.y - nd_->bound.nw.y)*(elem_q_->pt.y - nd_->bound.nw.y);
	// ������ ����.
		else if (elem_q_->pt.y < nd_->bound.se.y)
			return true; // ������ ��ģ��.
						 // ����
		else
			return min_dist_scale >
			(elem_q_->pt.y - nd_->bound.se.y)*(elem_q_->pt.y - nd_->bound.se.y);
	// ��
	else
		// ���
		if (elem_q_->pt.y < nd_->bound.nw.y)
			return min_dist_scale >
			(elem_q_->pt.x - nd_->bound.se.x)*(elem_q_->pt.x - nd_->bound.se.x)
			+ (elem_q_->pt.y - nd_->bound.nw.y)*(elem_q_->pt.y - nd_->bound.nw.y);
	// ����
		else if (elem_q_->pt.y < nd_->bound.se.y)
			return min_dist_scale >
			(elem_q_->pt.x - nd_->bound.se.x)*(elem_q_->pt.x - nd_->bound.se.x);
	// ����
		else
			return min_dist_scale >
			(elem_q_->pt.x - nd_->bound.se.x)*(elem_q_->pt.x - nd_->bound.se.x)
			+ (elem_q_->pt.y - nd_->bound.se.y)*(elem_q_->pt.y - nd_->bound.se.y);
};
//bool QuadTreeFastPooled::BOBTest(Node*& nd_, chk::Point2f& pt_q_) {
//	// �� node�� �ۿ� �ִٰ� �����Ѵ�.
//	// ��
//	float min_dist_scale = min_dist*scale2;
//	if (pt_q_.x < nd_->bound.nw.x)
//		// �»�
//		if (pt_q_.y < nd_->bound.nw.y)
//			return min_dist_scale >
//			(pt_q_.x - nd_->bound.nw.x)*(pt_q_.x - nd_->bound.nw.x)
//			+ (pt_q_.y - nd_->bound.nw.y)*(pt_q_.y - nd_->bound.nw.y);
//	// ����
//		else if (pt_q_.y < nd_->bound.se.y)
//			return min_dist_scale >
//			(pt_q_.x - nd_->bound.nw.x)*(pt_q_.x - nd_->bound.nw.x);
//	// ����
//		else
//			return min_dist_scale >
//			(pt_q_.x - nd_->bound.nw.x)*(pt_q_.x - nd_->bound.nw.x)
//			+ (pt_q_.y - nd_->bound.se.y)*(pt_q_.y - nd_->bound.se.y);
//	// ��
//	else if (pt_q_.x < nd_->bound.se.x)
//		// �߻�
//		if (pt_q_.y < nd_->bound.nw.y)
//			return min_dist_scale >
//			(pt_q_.y - nd_->bound.nw.y)*(pt_q_.y - nd_->bound.nw.y);
//	// ������ ����. ( ���ο� ��ġ�ϴ� ����̸�, �̶��� BOB ������ true�̴�. )
//		else if (pt_q_.y < nd_->bound.se.y)
//			return true;
//	// ����
//		else
//			return min_dist_scale >
//			(pt_q_.y - nd_->bound.se.y)*(pt_q_.y - nd_->bound.se.y);
//	// ��
//	else
//		// ���
//		if (pt_q_.y < nd_->bound.nw.y)
//			return min_dist_scale >
//			(pt_q_.x - nd_->bound.se.x)*(pt_q_.x - nd_->bound.se.x)
//			+ (pt_q_.y - nd_->bound.nw.y)*(pt_q_.y - nd_->bound.nw.y);
//	// ����
//		else if (pt_q_.y < nd_->bound.se.y)
//			return min_dist_scale >
//			(pt_q_.x - nd_->bound.se.x)*(pt_q_.x - nd_->bound.se.x);
//	// ����
//		else
//			return min_dist_scale >
//			(pt_q_.x - nd_->bound.se.x)*(pt_q_.x - nd_->bound.se.x)
//			+ (pt_q_.y - nd_->bound.se.y)*(pt_q_.y - nd_->bound.se.y);
//};

bool QuadTreeFastPooled::BOBTest(Node*& nd_, chk::Point2f& pt_q_) {
    // �� node�� �ۿ� �ִٰ� �����Ѵ�.
    // ��
    float min_dist_scale = min_dist*scale2;
    float close_x = pt_q_.x;
    float close_y = pt_q_.y;

    if (pt_q_.x < nd_->bound.nw.x) close_x = nd_->bound.nw.x;
    else if (pt_q_.x > nd_->bound.se.x) close_x = nd_->bound.se.x;

    if (pt_q_.y < nd_->bound.nw.y) close_y = nd_->bound.nw.y;
    else if (pt_q_.y > nd_->bound.se.y) close_y = nd_->bound.se.y;

    return (((pt_q_.x-close_x)*(pt_q_.x - close_x) + (pt_q_.y - close_y)*(pt_q_.y - close_y)) < min_dist_scale);
};

// stack����� NN search�̴�. ���� ����� ���� ã�� ����.
int QuadTreeFastPooled::searchNNSingleQuery(chk::Point2f& pt_q_, Node*& node_matched_) {
	// root�� stack ó���� �ִ´�.
	stack.push(this->root);

	// �Ÿ� ���� �ʱ�ȭ
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
		// leaf node�̸� ���� ����� ���� ã�´�.
		if (node->isleaf) {
			stack.total_access++;
			findNearestElempt(node, pt_q_, idx_elem_matched, node_matched_);
			if (INBOUND_PT(node, pt_q_) && BWBTest(node, pt_q_)) break;
		}
		else { // leaf node�� �ƴϸ�, ��� ���� �� �� �����Ѵ�.
			   // ������ child�� �������� �ʰų�, BOB �������� ������, �ڽ� ���� �� �ʿ䰡 ����.
			stack.total_access++;
            SELECT_CHILD_PT(node, pt_q_, flag_ew, flag_sn) // ���� ���ɼ��ִ� node�� �����Ѵ�. ���⿡ leaf�� �������� �ִ�.
            
            // child->depth�� �� child�� ���翩�θ� ���Ѵ�.
			child = node->first_child + (!flag_sn << 1) + flag_ew;
			if (child->depth && BOBTest(child, pt_q_)) stack.push(child);
			child = node->first_child + (flag_sn << 1) + !flag_ew;
			if (child->depth && BOBTest(child, pt_q_)) stack.push(child);
			child = node->first_child + (!flag_sn << 1) + !flag_ew;
			if (child->depth && BOBTest(child, pt_q_)) stack.push(child);
			// ���õ� child. ���� �������� �ִ�.
			child = node->first_child + (flag_sn << 1) + flag_ew;
			if (child->depth && BOBTest(child, pt_q_)) stack.push(child);
		}
	}

	stack.clear(); // stack �ʱ�ȭ. (�޸� ���븸 �ʱ�ȭ)
	if (min_dist > dist_thres) { // ������ dist_thres���� �ָ� ������.
        idx_elem_matched = -2; // ��Ī���� �ʾ����Ƿ�, point index�� -2�� �����Ѵ�.
		node_matched_ = this->root; // ��Ī���� �ʾ����Ƿ�, cached node�� root�� �����Ѵ�.
	}
	return idx_elem_matched;
};

int QuadTreeFastPooled::searchNNSingleQueryCached(chk::Point2f& pt_q_, Node*& node_cached_and_matched_) {
	// 0) �Ÿ� ���� �ʱ�ȭ
	min_dist = 1e15f;
	dist_temp = 0.0f;
	stack.total_access++; // node �湮������, counter �ø�.
	int idx_elem_matched = -2; // ��Ī�� ���� index
	Node* node = node_cached_and_matched_; // �켱 cached node�� �ҷ��´�. node_cached_and_matched�� ��Ī ����� �ٽ� ������ش�.

										   // �ϴ�, query�� root ���� (�̹��� ����)�� ��ġ�ϴ������� Ȯ������.
	if (INBOUND_PT(this->root, pt_q_)) {
		// root�� �������� ������ ��Ī�� ���� ���ٴ� �ǹ�. �Ϲ����� searchNN�� ����.
		if (node == this->root) {
			idx_elem_matched = searchNNSingleQuery(pt_q_, node_cached_and_matched_);
			goto flagFinish;
		}
		// ��Ī�� ���� �����ϴ� leaf node�� ������.
		// 1) nd_cached_���� ������ q�� ���� ����� ���� ã�´�. (������ leaf���� �Ѵ�...)
		// ����, q�� cached node�� bound �ȿ� ��ġ�ϰ�, BWB���� �����ϸ�, �׳� ��!
		// ����, inbound �ε�, BWB�� �������� �ʾҴ� ���̶��, parent�� ���ٰ� �ٽ� ����� �����õ�...
		node_cached_and_matched_ = nullptr;
		findNearestElempt(node, pt_q_, idx_elem_matched, node_cached_and_matched_);
		// cached node���� BWB�����ϴ� ��Ī���� �ִٸ�, �ٷ� ����.
		if (INBOUND_PT(node, pt_q_) && BWBTest(node, pt_q_)) {
			goto flagFinish;
		}

		// 2) BWBTest == true �� bridge���� �ö󰣴�.
		while (true) {
			node = node->parent;
			stack.total_access++;
			if (BWBTest(node, pt_q_)) break;
			//if (inBound(node, pt_q_)) break; // PWB Test only.
			if (node->parent == nullptr) break; // root
		}

		// 3) ���� ��ġ�� node���� �Ʒ��� ������ search ����.
		stack.push(node);
		// �Ÿ� ���� �ʱ�ȭ
		int flag_ew, flag_sn;
		Node* child = nullptr;
		while (stack.size > 0) {
			node = stack.top();
			stack.pop();
			// leaf node�̸� ���� ����� ���� ã�´�.
			if (node->isleaf) {
				stack.total_access++;
				findNearestElempt(node, pt_q_, idx_elem_matched, node_cached_and_matched_);
				if (INBOUND_PT(node, pt_q_) && BWBTest(node, pt_q_)) goto flagFinish;
				// if (inBound(node, pt_q_)) goto flagFinish; // PWB Test only.
			}
			// leaf node�� �ƴϸ�, ��� ���� �� �� �����Ѵ�.
			else {
				// ������ node�� BOB �������� ������, �ڽ� ���� �� �ʿ䰡 ����.
				// BOB�� �����ϸ�, �ڽ� ���� ���Ѵ�.
				stack.total_access++;
                SELECT_CHILD_PT(node, pt_q_, flag_ew, flag_sn)
				// ���� ������ child�� �ƴ� children�� ���߿� ����Ǿ�� �ϴ�, ���� stack�� ��.
				// ���� != nullptr && BOB pass �� ���鸸 ����.
				// �ٸ� child��, �ʱ�ȭ �� ���� depth�� ������ >0 �̴�.
				// �ʱ�ȭ ���� ���� ���� 0���� �Ǿ��ִ�. root �����ϰ�� 0�� ����������.
				child = node->first_child + (!flag_sn << 1) + flag_ew;
				if (child->depth && BOBTest(child, pt_q_)) stack.push(child);
				child = node->first_child + (flag_sn << 1) + !flag_ew;
				if (child->depth && BOBTest(child, pt_q_)) stack.push(child);
				child = node->first_child + (!flag_sn << 1) + !flag_ew;
				if (child->depth && BOBTest(child, pt_q_)) stack.push(child);
				// ���� child.
				child = node->first_child + (flag_sn << 1) + flag_ew;
				if (child->depth && BOBTest(child, pt_q_)) stack.push(child);
			}
		}
	}
	else { // ���ο� ��ġ���� ������, ��Ī�� ���� ����!
		node_cached_and_matched_ = this->root;
		idx_elem_matched = -2;
		goto flagFinish;
	}


	// ������ �� ����Ͽ�����, top�� 0���� ������ش�. (������ �����͸� ���� �ʿ�� ����.)
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
		if (INBOUND_PT(this->root, pts[i])) { // �̹��� ���ο� ������ search
			index_matched[i] = searchNNSingleQuery(pts[i], nodes_matched[i]);
		}
		else { // �̹��� ���ΰ� �ƴϸ�, �Ⱒ.
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
		if (INBOUND_PT(this->root, pts[i])) { // �̹��� ���ο� ������ search
			index_matched[i] = searchNNSingleQueryCached(pts[i], nodes_cached_and_matched[i]); // cached node�� �Է�.
		}
		else { // �̹��� ���ΰ� �ƴϸ�, �Ⱒ.
			index_matched[i] = -2;
			nodes_cached_and_matched[i] = this->root;
		}
	}
};

void QuadTreeFastPooled::showAllNodes() {
	// ù��°�� ������ root node�̴�.
	Node* node_temp = this->root;
	node_temp->showNodeSpec();
	for (int i = 1; i < node_vector.size(); i++) {
		node_temp = node_vector[i];
		if (node_temp->parent != nullptr)
			node_temp->showNodeSpec();
	}
};

#endif