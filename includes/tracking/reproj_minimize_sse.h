#ifndef _REPROJ_MINIMIZE_SSE_H_
#define _REPROJ_MINIMIZE_SSE_H_

#include <iostream>
#include <cstdlib>
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"  
#include "Eigen/Dense"

#include "../frame/stereo_frames.h"
#include "../tracking/tracking_reference.h"
#include "../quadtrees/stereo_multiquadtrees_wrapper.h"

#define GET4(ptr,i,j) (*(ptr + 4 * i + j))

using namespace std;
namespace chk {
    class SE3TrackerSSE {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        int MAX_PYR_LVL; //사실 maximum level은 5 이하로 가지 않는다. 그보다 작을 수 있으므로 저장해둔다.
        int ncols[5];
        int nrows[5];

        // camera matrix
        Eigen::Matrix3f K[5];
        Eigen::Matrix3f Kinv[5];

        Mat44 T_nlnr; // left to right extrinsic matrix. (calibrated)

        float fx, fy, cx, cy; // 필요할때 만들어쓰자.
        float fxinv, fyinv;

        float err; // err = err_e + err_ppat
        float err_e;
        float err_p;

        int n_valid_edge;
        int n_valid_point;

        int n_overthres_edge;
        int n_overthres_point;

        // n_overthres / n_valid < 0.6 =  huber thres is is good.
        Params* params;
        StereoFrame* frame_c;
        chk::TrackingReference* track_ref;
        StereoMultiQuadtreesWrapper* tree;

        SE3TrackerSSE(Params* params_,
            const vector<int>& nrows_pyr, const vector<int>& ncols_pyr,
            const vector<Eigen::Matrix3d>& K_pyr_, const Eigen::Matrix4d& T_nlnr_);
        ~SE3TrackerSSE();

        // linking & detaching functions 
        void linkRef(chk::TrackingReference* track_ref_) { track_ref = track_ref_; };
        void linkTree(StereoMultiQuadtreesWrapper* tree_) { tree = tree_; };
        void linkFrameCur(StereoFrame* frame_) { frame_c = frame_; };
        void detachRef() { track_ref = nullptr; };
        void detachTree() { tree = nullptr; };
        void detachFrameCur() { frame_c = nullptr; };

        void calcResidual(const Eigen::Matrix4d& T_ck_, const bool& is_cached, const float& huber_scaler_edge, const float& huber_scaler_point);
        void se3MotionTracking(const Eigen::Matrix4d& T_ck_, const bool& is_cached, const float& huber_scaler_edge, const float& huber_scaler_point, Vec6& delta, const float& lambda);




        // SSE implementation
        // edge warp -> edge matching -> validity test -> generateBufferEdge -> calcEdgeAffineBrightness
        // calcEdgeResidualAndWeightSSE -> calcEdgeHessianAndJacobianSSE (inside, updateSSE)
        // point patch warp -> calc
    private:
        // Hessian and Jacobian
        // We just solve Linear system JtWJ*delta_xi = mJtWr; where mJtWr = -J^t*W*r;
        // JtWJ matrix is guaranteed to be P.S.D and symmetric matrix.
        // Thus, we can efficiently solve this equation by Cholesky decomposition.
        // --> JtWJ.ldlt().solve(mJtWr);
        Mat66 JtWJ; // 6 x 6, = JtWJ_e + JtWJ_ppat
        Vec6 mJtWr; // 6 x 1, = mJtWr_e + mJtWr_ppat

        Mat66 JtWJ_e; // 6 x 6
        Vec6 mJtWr_e; // 6 x 1

        Mat66 JtWJ_p; // 6 x 6
        Vec6 mJtWr_p; // 6 x 1

        // level과 무관하게 하나씩만 가지고 있으면 된다.
        // For se3affinetracker_sse 
        // (1) edge references.
        float* buf_xel_warped; // warped edge reference to left current image.
        float* buf_yel_warped;
        float* buf_zel_warped;
        float* buf_xer_warped; // warped edge reference to right current image.
        float* buf_yer_warped;
        float* buf_zer_warped;

        float* buf_uel_warped; // warped and reprojected edge on left current image.
        float* buf_vel_warped;
        float* buf_uer_warped; // warped and reprojected edge on right current image.
        float* buf_ver_warped;

        float* buf_uel_matched; // matched edge on left current image.
        float* buf_vel_matched;
        float* buf_uer_matched; // matched edge on right current image.
        float* buf_ver_matched;

        float* buf_gxl_matched; // gradient on the matched left pixels.
        float* buf_gyl_matched;
        float* buf_gxr_matched; // gradient on the matched right pixels.
        float* buf_gyr_matched;

        float* buf_res_l_e; // r_e
        float* buf_weight_l_e;
        float* buf_res_r_e;
        float* buf_weight_r_e;

        // (2) point patches
        float* buf_xp_warped;
        float* buf_yp_warped;
        float* buf_zp_warped;

        float* buf_up_warped;
        float* buf_vp_warped;

        float* buf_up_tracked;
        float* buf_vp_tracked;

        float* buf_res_p; // r_p
        float* buf_weight_p;

        float* SSEData; // [4 * 28 ] (for make JtWJ and JtWr)

        void warpAndMatchAndCheckAndGenerateEdgeBuffers(const Eigen::Matrix4d& T_ck_, const bool& is_cached);
        void calcEdgeResidualAndWeightSSE(const float& huber_scaler_edge);
        void calcEdgeHessianAndJacobianSSE();
        void updateEdgeSSE(
            const __m128 &J1, const __m128 &J2, const __m128 &J3, const __m128 &J4, const __m128 &J5, const __m128 &J6,
            const __m128& res, const __m128& weight, float& err);


        void warpPointAndGeneratePointBuffers(const Eigen::Matrix4d& T_ck_);
        void calcPointResidualAndWeightSSE(const float& huber_scaler_point);
        void calcPointHessianAndJacobianSSE();
        void updatePointSSE(
            const __m128 &J1, const __m128 &J2, const __m128 &J3, const __m128 &J4, const __m128 &J5, const __m128 &J6,
            const __m128& res, const __m128& weight, float& err);


        void solveLevenbergMarquardtStepSSE(Vec6& delta, const float& lambda);

    };
    void SE3TrackerSSE::calcResidual(
        const Eigen::Matrix4d& T_ck_, const bool& is_cached, const float& huber_scaler_edge, const float& huber_scaler_point)
    {
        // 이 함수에서 exp(alpha), beta 를 먼저 추정하고, Huber norm을 조절한다.
        if (track_ref == nullptr) throw std::runtime_error("TRACKER: track_ref is not linked\n");
        if (tree == nullptr) throw std::runtime_error("TRACKER: tree is not linked\n");
        if (frame_c == nullptr) throw std::runtime_error("TRACKER: frame_c is not linked\n");

        // initialize
        float huber_thres_edge = 1.5f*huber_scaler_edge;
        float huber_thres_p = 1.5f*huber_scaler_point;

        err = 0;
        err_e = 0;
        err_p = 0;

        n_valid_point = 0;
        n_valid_edge = 0;

        n_overthres_point = 0;
        n_overthres_edge = 0;

        float fx = K[0](0, 0); // intrinsic matrix of the current level lvl.
        float fy = K[0](1, 1);
        float cx = K[0](0, 2);
        float cy = K[0](1, 2);
        int n_cols = frame_c->left->n_cols_pyr[0];
        int n_rows = frame_c->left->n_rows_pyr[0];

        Mat44 T_ck;
        T_ck << T_ck_(0, 0), T_ck_(0, 1), T_ck_(0, 2), T_ck_(0, 3),
            T_ck_(1, 0), T_ck_(1, 1), T_ck_(1, 2), T_ck_(1, 3),
            T_ck_(2, 0), T_ck_(2, 1), T_ck_(2, 2), T_ck_(2, 3),
            T_ck_(3, 0), T_ck_(3, 1), T_ck_(3, 2), T_ck_(3, 3);

        Mat33 R_ck = T_ck.block<3, 3>(0, 0); // rotation from key to cur.
        Vec3 t_ck = T_ck.block<3, 1>(0, 3); // translation from key to cur.
        Mat33 R_rl = T_nlnr.block<3, 3>(0, 0).inverse();
        Vec3 t_rl = -R_rl*T_nlnr.block<3, 1>(0, 3);

        int Nec = track_ref->Nec; // # of edge center points

        Vec3* p_Xe = track_ref->Xe;  // 3-D point edge center points
        Vec3* p_grad_ek = track_ref->grad_e; // gradient.
        char* p_dir = track_ref->bin_e; // edge direction

        int* p_idx_l = track_ref->idx_l;
        int* p_idx_r = track_ref->idx_r;
        Node** p_node_l = track_ref->node_l;
        Node** p_node_r = track_ref->node_r;

        Vec3* p_Xe_max = track_ref->Xe + Nec;

        for (; p_Xe < p_Xe_max; p_Xe++, p_dir++, p_idx_l++, p_idx_r++, p_node_l++, p_node_r++) 
        {
            // [1] warp and project edge centers onto the current level image.
            if ((*p_Xe)(2) < 0.01) continue;

            Vec3 X_warp_l = R_ck*(*p_Xe) + t_ck; // warped point onto left image.
            Vec3 X_warp_r = R_rl*X_warp_l + t_rl;// warped point onto right image.

            float invz = 1.0f / X_warp_l(2);
            float fxiz = fx*invz;
            float fyiz = fy*invz;
            float u_warp_l = fxiz*X_warp_l(0) + cx;
            float v_warp_l = fyiz*X_warp_l(1) + cy;
            invz = 1.0f / X_warp_r(2);
            fxiz = fx*invz;
            fyiz = fy*invz;
            float u_warp_r = fxiz*X_warp_r(0) + cx;
            float v_warp_r = fyiz*X_warp_r(1) + cy;

            // (1-a) matching left and right.
            if (is_cached) {
                tree->qts_left->searchNNCachedSingle(u_warp_l, v_warp_l, *p_dir, *p_idx_l, *p_node_l);
                tree->qts_right->searchNNCachedSingle(u_warp_r, v_warp_r, *p_dir, *p_idx_r, *p_node_r);
            }
            else {
                tree->qts_left->searchNNSingle(u_warp_l, v_warp_l, *p_dir, *p_idx_l, *p_node_l);
                tree->qts_right->searchNNSingle(u_warp_r, v_warp_r, *p_dir, *p_idx_r, *p_node_r);
            }

            // (1-b) if both images are matched, keep going. Otherwise, continue;
            if ((*p_idx_l < 0) || (*p_idx_r < 0)) continue;

            // [2] find matched gradients
            float gx_cl = frame_c->left->ft_edge[0]->grad_edge[*p_idx_l].x;
            float gy_cl = frame_c->left->ft_edge[0]->grad_edge[*p_idx_l].y;
            float igmag_cl = 1.0f / frame_c->left->ft_edge[0]->grad_edge[*p_idx_l].z;
            gx_cl *= igmag_cl;
            gy_cl *= igmag_cl;

            float gx_cr = frame_c->right->ft_edge[0]->grad_edge[*p_idx_r].x;
            float gy_cr = frame_c->right->ft_edge[0]->grad_edge[*p_idx_r].y;
            float igmag_cr = 1.0f / frame_c->right->ft_edge[0]->grad_edge[*p_idx_r].z;
            gx_cr *= igmag_cr;
            gy_cr *= igmag_cr;

            // (2-a) if both gradients render an angle over 3 degrees, continue;
            float dotprod = gx_cl*gx_cr + gy_cl*gy_cr;
            if (dotprod < params->qt.thres_gradtest) continue;

            // (2-b) if both patches are not similar, ignore it. 그리고 깊이가 갑자기 줄어드는경우, occlusion임.
            // 깊이가 줄어들었다는건 어떻게 판단하나? 아 stereo니까 matched pixel 이용해서 triangulate 가능!

            n_valid_edge += 2;

            // All tests are done!!!!
            float ul = frame_c->left->ft_edge[0]->pts_edge[*p_idx_l].x;
            float vl = frame_c->left->ft_edge[0]->pts_edge[*p_idx_l].y;
            float ur = frame_c->right->ft_edge[0]->pts_edge[*p_idx_r].x;
            float vr = frame_c->right->ft_edge[0]->pts_edge[*p_idx_r].y;

            // calculate and push edge residual, Huber weight, Jacobian, and Hessian (따로하자)
            float r_el = gx_cl*(u_warp_l - ul) + gy_cl*(v_warp_l - vl);
            float r_er = gx_cr*(u_warp_r - ur) + gy_cr*(v_warp_r - vr);
            float w_el = fabs(r_el) < huber_thres_edge ? 1.0f : huber_thres_edge / fabs(r_el);
            float w_er = fabs(r_er) < huber_thres_edge ? 1.0f : huber_thres_edge / fabs(r_er);
            err_e += r_el*r_el*w_el;
            err_e += r_er*r_er*w_er;

            if (w_el < 1.0f) ++n_overthres_point;
            if (w_er < 1.0f) ++n_overthres_point;
        }
        // edge 부분에서 한번씩? 튀는 부분이 있는데 왜그런지 확인해보자.
        // [2] with the prior \alpha and \beta, fill out brigtness-compensated residual and Jacobian terms for point patches.
        int Np = track_ref->Np; // # of edge center points

        Vec3* p_Xp = track_ref->Xp;  // 3-D point edge center points
        chk::Vec2* p_track = track_ref->pts_p_tracked;

        Vec3* p_Xp_max = p_Xp + Np;
        for (; p_Xp < p_Xp_max; p_Xp++, p_track++) {
            // [1] warp and project a corner point onto the current level image.
            if ((*p_Xp)(2) < 0.01) continue;

            Vec3 X_warp = R_ck*(*p_Xp) + t_ck; // warped point onto left image.
            float xw = X_warp(0);
            float yw = X_warp(1);
            float izw = 1.0f / X_warp(2);

            float u_warp = fx*xw * izw + cx;
            float v_warp = fy*yw * izw + cy;

            // in image test
            if ((u_warp < 1) && (u_warp > n_cols - 1) && (v_warp < 1) && (v_warp > n_rows - 1)) continue;

            float diff_u = u_warp - (*p_track)(0);
            float diff_v = v_warp - (*p_track)(1);
            float w_up = fabs(diff_u) < huber_thres_p ? 1.0f : huber_thres_p / fabs(diff_u);
            float w_vp = fabs(diff_v) < huber_thres_p ? 1.0f : huber_thres_p / fabs(diff_v);

            if (w_up < 1.0f) ++n_overthres_point;

            err_p += diff_u*diff_u*w_up + diff_v*diff_v*w_vp;
            ++n_valid_point;
        }

        err_e /= n_valid_edge;
        err_p /= n_valid_point;

        err_e = sqrt(err_e);
        err_p = sqrt(err_p);
        err = err_e + err_p;
    };






    void SE3TrackerSSE::se3MotionTracking(const Eigen::Matrix4d& T_ck_, const bool& is_cached, const float& huber_scaler_edge, const float& huber_scaler_point, Vec6& delta, const float& lambda) {
        warpAndMatchAndCheckAndGenerateEdgeBuffers(T_ck_, is_cached);
        calcEdgeResidualAndWeightSSE(huber_scaler_edge);
        calcEdgeHessianAndJacobianSSE();
        warpPointAndGeneratePointBuffers(T_ck_);
        calcPointResidualAndWeightSSE(huber_scaler_point);
        calcPointHessianAndJacobianSSE();

        solveLevenbergMarquardtStepSSE(delta, lambda);
    };


    void SE3TrackerSSE::solveLevenbergMarquardtStepSSE(Vec6& delta, const float& lambda) {
        JtWJ.setZero();
        mJtWr.setZero();
        err = 0;

        JtWJ_e /= (float)(2 * n_valid_edge);
        mJtWr_e /= (float)(2 * n_valid_edge);
        err_e /= (float)(2 * n_valid_edge);

        JtWJ_p /= (float)n_valid_point;
        mJtWr_p /= (float)n_valid_point;
        err_p /= (float)n_valid_point;

        JtWJ.noalias() += (JtWJ_e + JtWJ_p);
        mJtWr.noalias() += (mJtWr_e + mJtWr_p);

        err_e = sqrt(err_e);
        err_p = sqrt(err_p);
        err += err_e + err_p;

        JtWJ(0, 0) *= (1 + lambda);
        JtWJ(1, 1) *= (1 + lambda);
        JtWJ(2, 2) *= (1 + lambda);
        JtWJ(3, 3) *= (1 + lambda);
        JtWJ(4, 4) *= (1 + lambda);
        JtWJ(5, 5) *= (1 + lambda);

        delta = JtWJ.ldlt().solve(mJtWr);
    };


    void SE3TrackerSSE::warpAndMatchAndCheckAndGenerateEdgeBuffers(const Eigen::Matrix4d& T_ck_, const bool& is_cached) {
        // initialize for edge
        n_valid_edge = 0; // same for matched left edge centers.

        float fx = K[0](0, 0); // intrinsic matrix of the current level lvl.
        float fy = K[0](1, 1);
        float cx = K[0](0, 2);
        float cy = K[0](1, 2);
        int n_cols = frame_c->left->n_cols_pyr[0];
        int n_rows = frame_c->left->n_rows_pyr[0];

        Mat44 T_ck;
        T_ck << T_ck_(0, 0), T_ck_(0, 1), T_ck_(0, 2), T_ck_(0, 3),
            T_ck_(1, 0), T_ck_(1, 1), T_ck_(1, 2), T_ck_(1, 3),
            T_ck_(2, 0), T_ck_(2, 1), T_ck_(2, 2), T_ck_(2, 3),
            T_ck_(3, 0), T_ck_(3, 1), T_ck_(3, 2), T_ck_(3, 3);

        Mat33 R_ck = T_ck.block<3, 3>(0, 0); // rotation from key to cur.
        Vec3 t_ck = T_ck.block<3, 1>(0, 3); // translation from key to cur.
        Mat33 R_rl = T_nlnr.block<3, 3>(0, 0).inverse();
        Vec3 t_rl = -R_rl*T_nlnr.block<3, 1>(0, 3);

        int Nec = track_ref->Nec; // # of edge center points
        int Npe = track_ref->Npe[0];

        Vec3* p_Xe = track_ref->Xe;  // 3-D point edge center points
        char* p_dir = track_ref->bin_e; // edge direction

        int* p_idx_l = track_ref->idx_l;
        int* p_idx_r = track_ref->idx_r;
        Node** p_node_l = track_ref->node_l;
        Node** p_node_r = track_ref->node_r;

        Vec3* p_Xe_max = track_ref->Xe + Nec;
        for (; p_Xe < p_Xe_max; p_Xe++, p_dir++, p_idx_l++, p_idx_r++, p_node_l++, p_node_r++)
        {
            // [1] warp and project edge centers onto the current level image.
            if ((*p_Xe)(2) < 0.01) continue;

            Vec3 X_warp_l = R_ck*(*p_Xe) + t_ck; // warped point onto left image.
            Vec3 X_warp_r = R_rl*X_warp_l + t_rl;// warped point onto right image.

            float invz = 1.0f / X_warp_l(2);
            float u_warp_l = fx*X_warp_l(0) * invz + cx;
            float v_warp_l = fy*X_warp_l(1) * invz + cy;
            invz = 1.0f / X_warp_r(2);
            float u_warp_r = fx*X_warp_r(0) * invz + cx;
            float v_warp_r = fy*X_warp_r(1) * invz + cy;

            // (1-a) matching left and right.
            if (is_cached) {
                tree->qts_left->searchNNCachedSingle(u_warp_l, v_warp_l, *p_dir, *p_idx_l, *p_node_l);
                tree->qts_right->searchNNCachedSingle(u_warp_r, v_warp_r, *p_dir, *p_idx_r, *p_node_r);
            }
            else {
                tree->qts_left->searchNNSingle(u_warp_l, v_warp_l, *p_dir, *p_idx_l, *p_node_l);
                tree->qts_right->searchNNSingle(u_warp_r, v_warp_r, *p_dir, *p_idx_r, *p_node_r);
            }

            // (1-b) if both images are matched, keep going. Otherwise, continue;
            if ((*p_idx_l < 0) || (*p_idx_r < 0)) continue;

            // [2] find matched gradients
            float gx_cl = frame_c->left->ft_edge[0]->grad_edge[*p_idx_l].x;
            float gy_cl = frame_c->left->ft_edge[0]->grad_edge[*p_idx_l].y;
            float igmag_cl = 1.0f / frame_c->left->ft_edge[0]->grad_edge[*p_idx_l].z;
            gx_cl *= igmag_cl;
            gy_cl *= igmag_cl;

            float gx_cr = frame_c->right->ft_edge[0]->grad_edge[*p_idx_r].x;
            float gy_cr = frame_c->right->ft_edge[0]->grad_edge[*p_idx_r].y;
            float igmag_cr = 1.0f / frame_c->right->ft_edge[0]->grad_edge[*p_idx_r].z;
            gx_cr *= igmag_cr;
            gy_cr *= igmag_cr;

            // (2-a) if both gradients render an angle over 3 degrees, continue;
            float dotprod = gx_cl*gx_cr + gy_cl*gy_cr;
            if (dotprod < params->qt.thres_gradtest) continue;

            // (2-b) if both patches are not similar, ignore it. 그리고 깊이가 갑자기 줄어드는경우, occlusion임.
            // 깊이가 줄어들었다는건 어떻게 판단하나? 아 stereo니까 matched pixel 이용해서 triangulate 가능!

            float uel_matched = frame_c->left->ft_edge[0]->pts_edge[*p_idx_l].x;
            float vel_matched = frame_c->left->ft_edge[0]->pts_edge[*p_idx_l].y;
            float uer_matched = frame_c->right->ft_edge[0]->pts_edge[*p_idx_r].x;
            float ver_matched = frame_c->right->ft_edge[0]->pts_edge[*p_idx_r].y;

            // update buffers
            *(buf_xel_warped + n_valid_edge) = X_warp_l(0);
            *(buf_yel_warped + n_valid_edge) = X_warp_l(1);
            *(buf_zel_warped + n_valid_edge) = X_warp_l(2);
            *(buf_uel_warped + n_valid_edge) = u_warp_l;
            *(buf_vel_warped + n_valid_edge) = v_warp_l;
            *(buf_uel_matched + n_valid_edge) = uel_matched;
            *(buf_vel_matched + n_valid_edge) = vel_matched;
            *(buf_gxl_matched + n_valid_edge) = gx_cl;
            *(buf_gyl_matched + n_valid_edge) = gy_cl;

            *(buf_xer_warped + n_valid_edge) = X_warp_r(0);
            *(buf_yer_warped + n_valid_edge) = X_warp_r(1);
            *(buf_zer_warped + n_valid_edge) = X_warp_r(2);
            *(buf_uer_warped + n_valid_edge) = u_warp_r;
            *(buf_ver_warped + n_valid_edge) = v_warp_r;
            *(buf_uer_matched + n_valid_edge) = uer_matched;
            *(buf_ver_matched + n_valid_edge) = ver_matched;
            *(buf_gxr_matched + n_valid_edge) = gx_cr;
            *(buf_gyr_matched + n_valid_edge) = gy_cr;

            ++n_valid_edge;
        }
    };

    void SE3TrackerSSE::calcEdgeResidualAndWeightSSE(const float& huber_scaler_edge) {
        n_overthres_edge = 0;

        // SSE version!
        float huber_thres_edge = 1.5f*huber_scaler_edge;

        // fourth multiplier? 
        int n_steps = n_valid_edge / 4;
        int n_remainders = n_valid_edge % 4;

        // left first!
        __m128 u_warp4, v_warp4, u_match4, v_match4, gx4, gy4;
        // cout << "n_step:" << n_steps << ", n_remainder:" << n_remainders << endl;

        int idx = 0;
        for (; idx < n_valid_edge - 3; idx += 4) {
            u_warp4 = _mm_load_ps(buf_uel_warped + idx);
            v_warp4 = _mm_load_ps(buf_vel_warped + idx);
            u_match4 = _mm_load_ps(buf_uel_matched + idx);
            v_match4 = _mm_load_ps(buf_vel_matched + idx);
            gx4 = _mm_load_ps(buf_gxl_matched + idx);
            gy4 = _mm_load_ps(buf_gyl_matched + idx);

            // calculate residual left edge
            __m128 rrrr = _mm_add_ps(_mm_mul_ps(gx4, _mm_sub_ps(u_warp4, u_match4)), _mm_mul_ps(gy4, _mm_sub_ps(v_warp4, v_match4)));
            _mm_store_ps(buf_res_l_e + idx, rrrr);

            // calculate weight left edge (Huber weight)
            float r_el = *(buf_res_l_e + idx);
            *(buf_weight_l_e + idx) = fabs(r_el) < huber_thres_edge ? 1.0f : huber_thres_edge / fabs(r_el);
            if (*(buf_weight_l_e + idx) < 1.0f) ++n_overthres_edge;

            r_el = *(buf_res_l_e + idx + 1);
            *(buf_weight_l_e + idx + 1) = fabs(r_el) < huber_thres_edge ? 1.0f : huber_thres_edge / fabs(r_el);
            if (*(buf_weight_l_e + idx + 1) < 1.0f) ++n_overthres_edge;

            r_el = *(buf_res_l_e + idx + 2);
            *(buf_weight_l_e + idx + 2) = fabs(r_el) < huber_thres_edge ? 1.0f : huber_thres_edge / fabs(r_el);
            if (*(buf_weight_l_e + idx + 2) < 1.0f) ++n_overthres_edge;

            r_el = *(buf_res_l_e + idx + 3);
            *(buf_weight_l_e + idx + 3) = fabs(r_el) < huber_thres_edge ? 1.0f : huber_thres_edge / fabs(r_el);
            if (*(buf_weight_l_e + idx + 3) < 1.0f) ++n_overthres_edge;
        }
        for (; idx < n_valid_edge; ++idx) {
            float u_warp_l = *(buf_uel_warped + idx);
            float v_warp_l = *(buf_vel_warped + idx);
            float gx_cl = *(buf_gxl_matched + idx);
            float gy_cl = *(buf_gyl_matched + idx);

            float ul = *(buf_uel_matched + idx);
            float vl = *(buf_vel_matched + idx);

            // calculate and push edge residual, Huber weight, Jacobian, and Hessian (따로하자)
            float r_el = gx_cl*(u_warp_l - ul) + gy_cl*(v_warp_l - vl);
            float w_el = fabs(r_el) < huber_thres_edge ? 1.0f : huber_thres_edge / fabs(r_el);

            *(buf_res_l_e + idx) = r_el;
            *(buf_weight_l_e + idx) = w_el;

            if (w_el < 1.0f) ++n_overthres_edge;
        }

        //right
        idx = 0;
        for (; idx < n_valid_edge - 3; idx += 4) {
            u_warp4 = _mm_load_ps(buf_uer_warped + idx);
            v_warp4 = _mm_load_ps(buf_ver_warped + idx);
            u_match4 = _mm_load_ps(buf_uer_matched + idx);
            v_match4 = _mm_load_ps(buf_ver_matched + idx);
            gx4 = _mm_load_ps(buf_gxr_matched + idx);
            gy4 = _mm_load_ps(buf_gyr_matched + idx);

            // calculate residual left edge
            __m128 rrrr = _mm_add_ps(_mm_mul_ps(gx4, _mm_sub_ps(u_warp4, u_match4)), _mm_mul_ps(gy4, _mm_sub_ps(v_warp4, v_match4)));
            _mm_store_ps(buf_res_r_e + idx, rrrr);

            // calculate weight left edge (Huber weight)
            float r_er = *(buf_res_r_e + idx);
            *(buf_weight_r_e + idx) = fabs(r_er) < huber_thres_edge ? 1.0f : huber_thres_edge / fabs(r_er);

            r_er = *(buf_res_r_e + idx + 1);
            *(buf_weight_r_e + idx + 1) = fabs(r_er) < huber_thres_edge ? 1.0f : huber_thres_edge / fabs(r_er);

            r_er = *(buf_res_r_e + idx + 2);
            *(buf_weight_r_e + idx + 2) = fabs(r_er) < huber_thres_edge ? 1.0f : huber_thres_edge / fabs(r_er);

            r_er = *(buf_res_r_e + idx + 3);
            *(buf_weight_r_e + idx + 3) = fabs(r_er) < huber_thres_edge ? 1.0f : huber_thres_edge / fabs(r_er);
        }
        for (; idx < n_valid_edge; ++idx) {
            float u_warp_r = *(buf_uer_warped + idx);
            float v_warp_r = *(buf_ver_warped + idx);
            float gx_cr = *(buf_gxr_matched + idx);
            float gy_cr = *(buf_gyr_matched + idx);

            float ur = *(buf_uer_matched + idx);
            float vr = *(buf_ver_matched + idx);

            // calculate and push edge residual, Huber weight, Jacobian, and Hessian (따로하자)
            float r_er = gx_cr*(u_warp_r - ur) + gy_cr*(v_warp_r - vr);
            float w_er = fabs(r_er) < huber_thres_edge ? 1.0f : huber_thres_edge / fabs(r_er);

            *(buf_res_r_e + idx) = r_er;
            *(buf_weight_r_e + idx) = w_er;
        }
    };

    void SE3TrackerSSE::calcEdgeHessianAndJacobianSSE() {
        JtWJ_e.setZero();
        mJtWr_e.setZero();
        err_e = 0;

        __m128 ones = _mm_set1_ps(1.0f);
        __m128 zeros = _mm_set1_ps(0.0f);
        __m128 minusones = _mm_set1_ps(-1.0f);

        __m128 J1, J2, J3, J4, J5, J6; // four data at once.        
        __m128 fx4 = _mm_set1_ps(K[0](0, 0));
        __m128 fy4 = _mm_set1_ps(K[0](1, 1));
        int idx = 0;
        for (; idx < n_valid_edge - 3; idx += 4) {
            __m128 x4 = _mm_load_ps(buf_xel_warped + idx);
            __m128 y4 = _mm_load_ps(buf_yel_warped + idx);
            __m128 invz4 = _mm_rcp_ps(_mm_load_ps(buf_zel_warped + idx));

            __m128 xy4 = _mm_mul_ps(x4, y4);
            __m128 xinvz4 = _mm_mul_ps(x4, invz4);
            __m128 yinvz4 = _mm_mul_ps(y4, invz4);

            __m128 gxfx4 = _mm_mul_ps(_mm_load_ps(buf_gxl_matched + idx), fx4);
            __m128 gyfy4 = _mm_mul_ps(_mm_load_ps(buf_gyl_matched + idx), fy4);

            // J(1) = gx*fx/z ;
            J1 = _mm_mul_ps(gxfx4, invz4);
            // J(2) = gy*fy/z ;
            J2 = _mm_mul_ps(gyfy4, invz4);
            // J(3) = (-gx*fx*x/z - gy*fy*y/z)/z;
            J3 = _mm_mul_ps(_mm_sub_ps(zeros, _mm_add_ps(_mm_mul_ps(gxfx4, xinvz4), _mm_mul_ps(gyfy4, yinvz4))), invz4);
            // J(4) = -( gxfx*xizw*yizw + gyfy*(1.0f + yizw*yizw) );
            J4 = _mm_sub_ps(zeros,
                _mm_add_ps(_mm_mul_ps(_mm_mul_ps(gxfx4, xinvz4), yinvz4), _mm_mul_ps(gyfy4, _mm_add_ps(ones, _mm_mul_ps(yinvz4, yinvz4))))
            );
            // J(5) = gxfx*(1.0f + xizw*xizw) + gyfy*xizw*yizw;
            J5 = _mm_add_ps(_mm_mul_ps(gxfx4, _mm_add_ps(ones, _mm_mul_ps(xinvz4, xinvz4))), _mm_mul_ps(gyfy4, _mm_mul_ps(xinvz4, yinvz4)));
            // J(6) = gyfy*xizw - gxfx*yizw ;
            J6 = _mm_sub_ps(_mm_mul_ps(gyfy4, xinvz4), _mm_mul_ps(gxfx4, yinvz4));

            // update
            updateEdgeSSE(J1, J2, J3, J4, J5, J6, _mm_load_ps(buf_res_l_e + idx), _mm_load_ps(buf_weight_l_e + idx), err_e);
        };
        for (; idx < n_valid_edge; ++idx) {
            float x = *(buf_xel_warped + idx);
            float y = *(buf_yel_warped + idx);
            float iz = 1.0f / (*(buf_zel_warped + idx));

            float xy = x*y;
            float xiz = x*iz;
            float yiz = y*iz;

            float gxfx = *(buf_gxl_matched + idx)*fx;
            float gyfy = *(buf_gyl_matched + idx)*fy;

            Vec6 J;
            J(0) = gxfx*iz;
            J(1) = gyfy*iz;
            J(2) = (-gxfx*xiz - gyfy*yiz)*iz; // 여기가 틀렸었네 ; 
            J(3) = -gxfx*xiz*yiz - gyfy*(1.0f + yiz*yiz);
            J(4) = gxfx*(1.0f + xiz*xiz) + gyfy*xiz*yiz;
            J(5) = -gxfx*yiz + gyfy*xiz;

            float r = *(buf_res_l_e + idx);
            float w = *(buf_weight_l_e + idx);
            JtWJ_e.noalias() += (J*J.transpose())*w;
            mJtWr_e.noalias() -= J*(r*w);
            err_e += r*r*w;
        };

        // right
        idx = 0;
        for (; idx < n_valid_edge - 3; idx += 4) {
            __m128 x4 = _mm_load_ps(buf_xer_warped + idx);
            __m128 y4 = _mm_load_ps(buf_yer_warped + idx);
            __m128 invz4 = _mm_rcp_ps(_mm_load_ps(buf_zer_warped + idx));

            __m128 xy4 = _mm_mul_ps(x4, y4);
            __m128 xinvz4 = _mm_mul_ps(x4, invz4);
            __m128 yinvz4 = _mm_mul_ps(y4, invz4);

            __m128 gxfx4 = _mm_mul_ps(_mm_load_ps(buf_gxr_matched + idx), fx4);
            __m128 gyfy4 = _mm_mul_ps(_mm_load_ps(buf_gyr_matched + idx), fy4);

            // J(1) = gx*fx/z ;
            J1 = _mm_mul_ps(gxfx4, invz4);
            // J(2) = gy*fy/z ;
            J2 = _mm_mul_ps(gyfy4, invz4);
            // J(3) = (-gx*fx*x/z - gy*fy*y/z)/z;
            J3 = _mm_mul_ps(_mm_sub_ps(zeros, _mm_add_ps(_mm_mul_ps(gxfx4, xinvz4), _mm_mul_ps(gyfy4, yinvz4))), invz4);
            // J(4) = -( gxfx*xizw*yizw + gyfy*(1.0f + yizw*yizw) );
            J4 = _mm_sub_ps(zeros,
                _mm_add_ps(_mm_mul_ps(_mm_mul_ps(gxfx4, xinvz4), yinvz4), _mm_mul_ps(gyfy4, _mm_add_ps(ones, _mm_mul_ps(yinvz4, yinvz4))))
            );
            // J(5) = gxfx*(1.0f + xizw*xizw) + gyfy*xizw*yizw;
            J5 = _mm_add_ps(_mm_mul_ps(gxfx4, _mm_add_ps(ones, _mm_mul_ps(xinvz4, xinvz4))), _mm_mul_ps(gyfy4, _mm_mul_ps(xinvz4, yinvz4)));
            // J(6) = gyfy*xizw - gxfx*yizw ;
            J6 = _mm_sub_ps(_mm_mul_ps(gyfy4, xinvz4), _mm_mul_ps(gxfx4, yinvz4));

            // update
            updateEdgeSSE(J1, J2, J3, J4, J5, J6, _mm_load_ps(buf_res_r_e + idx), _mm_load_ps(buf_weight_r_e + idx), err_e);
        };
        for (; idx < n_valid_edge; ++idx) {
            float x = *(buf_xer_warped + idx);
            float y = *(buf_yer_warped + idx);
            float iz = 1.0f / (*(buf_zer_warped + idx));

            float xy = x*y;
            float xiz = x*iz;
            float yiz = y*iz;

            float gxfx = *(buf_gxr_matched + idx)*fx;
            float gyfy = *(buf_gyr_matched + idx)*fy;

            Vec6 J;
            J(0) = gxfx*iz;
            J(1) = gyfy*iz;
            J(2) = (-gxfx*xiz - gyfy*yiz)*iz; // 여기가 틀렸었네 ; 
            J(3) = -gxfx*xiz*yiz - gyfy*(1.0f + yiz*yiz);
            J(4) = gxfx*(1.0f + xiz*xiz) + gyfy*xiz*yiz;
            J(5) = -gxfx*yiz + gyfy*xiz;

            float r = *(buf_res_r_e + idx);
            float w = *(buf_weight_r_e + idx);
            JtWJ_e.noalias() += (J*J.transpose())*w;
            mJtWr_e.noalias() -= J*(r*w);
            err_e += r*r*w;
        };
    };
    void SE3TrackerSSE::updateEdgeSSE(
        const __m128 &J1, const __m128 &J2, const __m128 &J3, const __m128 &J4, const __m128 &J5, const __m128 &J6,
        const __m128& res, const __m128& weight, float& err) 
    {
        //A.noalias() += J * J.transpose() * weight;
        memset(SSEData, 0, sizeof(float) * 4 * 28);

        __m128 J1w = _mm_mul_ps(J1, weight);
        _mm_store_ps(SSEData + 4 * 0, _mm_add_ps(_mm_load_ps(SSEData + 4 * 0), _mm_mul_ps(J1w, J1)));
        _mm_store_ps(SSEData + 4 * 1, _mm_add_ps(_mm_load_ps(SSEData + 4 * 1), _mm_mul_ps(J1w, J2)));
        _mm_store_ps(SSEData + 4 * 2, _mm_add_ps(_mm_load_ps(SSEData + 4 * 2), _mm_mul_ps(J1w, J3)));
        _mm_store_ps(SSEData + 4 * 3, _mm_add_ps(_mm_load_ps(SSEData + 4 * 3), _mm_mul_ps(J1w, J4)));
        _mm_store_ps(SSEData + 4 * 4, _mm_add_ps(_mm_load_ps(SSEData + 4 * 4), _mm_mul_ps(J1w, J5)));
        _mm_store_ps(SSEData + 4 * 5, _mm_add_ps(_mm_load_ps(SSEData + 4 * 5), _mm_mul_ps(J1w, J6)));


        __m128 J2w = _mm_mul_ps(J2, weight);
        _mm_store_ps(SSEData + 4 * 6, _mm_add_ps(_mm_load_ps(SSEData + 4 * 6), _mm_mul_ps(J2w, J2)));
        _mm_store_ps(SSEData + 4 * 7, _mm_add_ps(_mm_load_ps(SSEData + 4 * 7), _mm_mul_ps(J2w, J3)));
        _mm_store_ps(SSEData + 4 * 8, _mm_add_ps(_mm_load_ps(SSEData + 4 * 8), _mm_mul_ps(J2w, J4)));
        _mm_store_ps(SSEData + 4 * 9, _mm_add_ps(_mm_load_ps(SSEData + 4 * 9), _mm_mul_ps(J2w, J5)));
        _mm_store_ps(SSEData + 4 * 10, _mm_add_ps(_mm_load_ps(SSEData + 4 * 10), _mm_mul_ps(J2w, J6)));


        __m128 J3w = _mm_mul_ps(J3, weight);
        _mm_store_ps(SSEData + 4 * 11, _mm_add_ps(_mm_load_ps(SSEData + 4 * 11), _mm_mul_ps(J3w, J3)));
        _mm_store_ps(SSEData + 4 * 12, _mm_add_ps(_mm_load_ps(SSEData + 4 * 12), _mm_mul_ps(J3w, J4)));
        _mm_store_ps(SSEData + 4 * 13, _mm_add_ps(_mm_load_ps(SSEData + 4 * 13), _mm_mul_ps(J3w, J5)));
        _mm_store_ps(SSEData + 4 * 14, _mm_add_ps(_mm_load_ps(SSEData + 4 * 14), _mm_mul_ps(J3w, J6)));

        __m128 J4w = _mm_mul_ps(J4, weight);
        _mm_store_ps(SSEData + 4 * 15, _mm_add_ps(_mm_load_ps(SSEData + 4 * 15), _mm_mul_ps(J4w, J4)));
        _mm_store_ps(SSEData + 4 * 16, _mm_add_ps(_mm_load_ps(SSEData + 4 * 16), _mm_mul_ps(J4w, J5)));
        _mm_store_ps(SSEData + 4 * 17, _mm_add_ps(_mm_load_ps(SSEData + 4 * 17), _mm_mul_ps(J4w, J6)));


        __m128 J5w = _mm_mul_ps(J5, weight);
        _mm_store_ps(SSEData + 4 * 18, _mm_add_ps(_mm_load_ps(SSEData + 4 * 18), _mm_mul_ps(J5w, J5)));
        _mm_store_ps(SSEData + 4 * 19, _mm_add_ps(_mm_load_ps(SSEData + 4 * 19), _mm_mul_ps(J5w, J6)));


        __m128 J6w = _mm_mul_ps(J6, weight);
        _mm_store_ps(SSEData + 4 * 20, _mm_add_ps(_mm_load_ps(SSEData + 4 * 20), _mm_mul_ps(J6w, J6)));


        //b.noalias() -= J * (res * weight);
        __m128 resw = _mm_mul_ps(res, weight);
        _mm_store_ps(SSEData + 4 * 21, _mm_add_ps(_mm_load_ps(SSEData + 4 * 21), _mm_mul_ps(resw, J1)));
        _mm_store_ps(SSEData + 4 * 22, _mm_add_ps(_mm_load_ps(SSEData + 4 * 22), _mm_mul_ps(resw, J2)));
        _mm_store_ps(SSEData + 4 * 23, _mm_add_ps(_mm_load_ps(SSEData + 4 * 23), _mm_mul_ps(resw, J3)));
        _mm_store_ps(SSEData + 4 * 24, _mm_add_ps(_mm_load_ps(SSEData + 4 * 24), _mm_mul_ps(resw, J4)));
        _mm_store_ps(SSEData + 4 * 25, _mm_add_ps(_mm_load_ps(SSEData + 4 * 25), _mm_mul_ps(resw, J5)));
        _mm_store_ps(SSEData + 4 * 26, _mm_add_ps(_mm_load_ps(SSEData + 4 * 26), _mm_mul_ps(resw, J6)));

        //error += res * res * weight;
        _mm_store_ps(SSEData + 4 * 27, _mm_add_ps(_mm_load_ps(SSEData + 4 * 27), _mm_mul_ps(resw, res)));

        // update JtWJ_sse,
        JtWJ_e(0, 0) += (GET4(SSEData, 0, 0) + GET4(SSEData, 0, 1) + GET4(SSEData, 0, 2) + GET4(SSEData, 0, 3));
        JtWJ_e(1, 0) = (JtWJ_e(0, 1) += (GET4(SSEData, 1, 0) + GET4(SSEData, 1, 1) + GET4(SSEData, 1, 2) + GET4(SSEData, 1, 3)));
        JtWJ_e(2, 0) = (JtWJ_e(0, 2) += (GET4(SSEData, 2, 0) + GET4(SSEData, 2, 1) + GET4(SSEData, 2, 2) + GET4(SSEData, 2, 3)));
        JtWJ_e(3, 0) = (JtWJ_e(0, 3) += (GET4(SSEData, 3, 0) + GET4(SSEData, 3, 1) + GET4(SSEData, 3, 2) + GET4(SSEData, 3, 3)));
        JtWJ_e(4, 0) = (JtWJ_e(0, 4) += (GET4(SSEData, 4, 0) + GET4(SSEData, 4, 1) + GET4(SSEData, 4, 2) + GET4(SSEData, 4, 3)));
        JtWJ_e(5, 0) = (JtWJ_e(0, 5) += (GET4(SSEData, 5, 0) + GET4(SSEData, 5, 1) + GET4(SSEData, 5, 2) + GET4(SSEData, 5, 3)));

        JtWJ_e(1, 1) += (GET4(SSEData, 6, 0) + GET4(SSEData, 6, 1) + GET4(SSEData, 6, 2) + GET4(SSEData, 6, 3));
        JtWJ_e(2, 1) = (JtWJ_e(1, 2) += (GET4(SSEData, 7, 0) + GET4(SSEData, 7, 1) + GET4(SSEData, 7, 2) + GET4(SSEData, 7, 3)));
        JtWJ_e(3, 1) = (JtWJ_e(1, 3) += (GET4(SSEData, 8, 0) + GET4(SSEData, 8, 1) + GET4(SSEData, 8, 2) + GET4(SSEData, 8, 3)));
        JtWJ_e(4, 1) = (JtWJ_e(1, 4) += (GET4(SSEData, 9, 0) + GET4(SSEData, 9, 1) + GET4(SSEData, 9, 2) + GET4(SSEData, 9, 3)));
        JtWJ_e(5, 1) = (JtWJ_e(1, 5) += (GET4(SSEData, 10, 0) + GET4(SSEData, 10, 1) + GET4(SSEData, 10, 2) + GET4(SSEData, 10, 3)));

        JtWJ_e(2, 2) += (GET4(SSEData, 11, 0) + GET4(SSEData, 11, 1) + GET4(SSEData, 11, 2) + GET4(SSEData, 11, 3));
        JtWJ_e(3, 2) = (JtWJ_e(2, 3) += (GET4(SSEData, 12, 0) + GET4(SSEData, 12, 1) + GET4(SSEData, 12, 2) + GET4(SSEData, 12, 3)));
        JtWJ_e(4, 2) = (JtWJ_e(2, 4) += (GET4(SSEData, 13, 0) + GET4(SSEData, 13, 1) + GET4(SSEData, 13, 2) + GET4(SSEData, 13, 3)));
        JtWJ_e(5, 2) = (JtWJ_e(2, 5) += (GET4(SSEData, 14, 0) + GET4(SSEData, 14, 1) + GET4(SSEData, 14, 2) + GET4(SSEData, 14, 3)));

        JtWJ_e(3, 3) += (GET4(SSEData, 15, 0) + GET4(SSEData, 15, 1) + GET4(SSEData, 15, 2) + GET4(SSEData, 15, 3));
        JtWJ_e(4, 3) = (JtWJ_e(3, 4) += (GET4(SSEData, 16, 0) + GET4(SSEData, 16, 1) + GET4(SSEData, 16, 2) + GET4(SSEData, 16, 3)));
        JtWJ_e(5, 3) = (JtWJ_e(3, 5) += (GET4(SSEData, 17, 0) + GET4(SSEData, 17, 1) + GET4(SSEData, 17, 2) + GET4(SSEData, 17, 3)));

        JtWJ_e(4, 4) += (GET4(SSEData, 18, 0) + GET4(SSEData, 18, 1) + GET4(SSEData, 18, 2) + GET4(SSEData, 18, 3));
        JtWJ_e(5, 4) = (JtWJ_e(4, 5) += (GET4(SSEData, 19, 0) + GET4(SSEData, 19, 1) + GET4(SSEData, 19, 2) + GET4(SSEData, 19, 3)));

        JtWJ_e(5, 5) += (GET4(SSEData, 20, 0) + GET4(SSEData, 20, 1) + GET4(SSEData, 20, 2) + GET4(SSEData, 20, 3));

        // update  mJtWr_sse
        mJtWr_e(0) -= (GET4(SSEData, 21, 0) + GET4(SSEData, 21, 1) + GET4(SSEData, 21, 2) + GET4(SSEData, 21, 3));
        mJtWr_e(1) -= (GET4(SSEData, 22, 0) + GET4(SSEData, 22, 1) + GET4(SSEData, 22, 2) + GET4(SSEData, 22, 3));
        mJtWr_e(2) -= (GET4(SSEData, 23, 0) + GET4(SSEData, 23, 1) + GET4(SSEData, 23, 2) + GET4(SSEData, 23, 3));
        mJtWr_e(3) -= (GET4(SSEData, 24, 0) + GET4(SSEData, 24, 1) + GET4(SSEData, 24, 2) + GET4(SSEData, 24, 3));
        mJtWr_e(4) -= (GET4(SSEData, 25, 0) + GET4(SSEData, 25, 1) + GET4(SSEData, 25, 2) + GET4(SSEData, 25, 3));
        mJtWr_e(5) -= (GET4(SSEData, 26, 0) + GET4(SSEData, 26, 1) + GET4(SSEData, 26, 2) + GET4(SSEData, 26, 3));

        // update err.
        err += (GET4(SSEData, 27, 0) + GET4(SSEData, 27, 1) + GET4(SSEData, 27, 2) + GET4(SSEData, 27, 3));
    }

    void SE3TrackerSSE::warpPointAndGeneratePointBuffers(const Eigen::Matrix4d& T_ck_) 
    {
        // initialize for edge
        n_valid_point = 0;

        float fx = K[0](0, 0); // intrinsic matrix of the current level lvl.
        float fy = K[0](1, 1);
        float cx = K[0](0, 2);
        float cy = K[0](1, 2);
        int n_cols = frame_c->left->n_cols_pyr[0];
        int n_rows = frame_c->left->n_rows_pyr[0];

        Mat44 T_ck;
        T_ck << T_ck_(0, 0), T_ck_(0, 1), T_ck_(0, 2), T_ck_(0, 3),
            T_ck_(1, 0), T_ck_(1, 1), T_ck_(1, 2), T_ck_(1, 3),
            T_ck_(2, 0), T_ck_(2, 1), T_ck_(2, 2), T_ck_(2, 3),
            T_ck_(3, 0), T_ck_(3, 1), T_ck_(3, 2), T_ck_(3, 3);

        Mat33 R_ck = T_ck.block<3, 3>(0, 0); // rotation from key to cur.
        Vec3 t_ck = T_ck.block<3, 1>(0, 3); // translation from key to cur.

         // with the prior \alpha and \beta, fill out brigtness-compensated residual and Jacobian terms for point patches.
        int Np = track_ref->Np; // # of edge center points

        Vec3* p_Xp = track_ref->Xp;
        chk::Vec2* p_track = track_ref->pts_p_tracked;

        Vec3* p_Xp_max = p_Xp + Np;
        for (; p_Xp < p_Xp_max; p_Xp++, p_track++) {
            // [1] warp and project a corner point onto the current level image.
            if ((*p_Xp)(2) < 0.01) continue;
            Vec3 X_warp = R_ck*(*p_Xp) + t_ck; // warped point onto left image.

            float xw = X_warp(0);
            float yw = X_warp(1);
            float izw = 1.0f / X_warp(2);

            float u_warp = fx*xw * izw + cx;
            float v_warp = fy*yw * izw + cy;

            // in image test
            if ((u_warp < 1) && (u_warp > n_cols - 1) && (v_warp < 1) && (v_warp > n_rows - 1)) continue;


            *(buf_up_warped + n_valid_point) = u_warp;
            *(buf_vp_warped + n_valid_point) = v_warp;
            *(buf_up_tracked + n_valid_point) = (*p_track)(0);
            *(buf_vp_tracked + n_valid_point) = (*p_track)(1);

            *(buf_xp_warped + n_valid_point) = X_warp(0);
            *(buf_yp_warped + n_valid_point) = X_warp(1);
            *(buf_zp_warped + n_valid_point) = X_warp(2);

            ++n_valid_point;
        }
    };

    void SE3TrackerSSE::calcPointResidualAndWeightSSE(const float& huber_scaler_point) {
        n_overthres_point = 0;
        float huber_thres_p = 1.5f*huber_scaler_point;

        __m128 uwarp4, utrack4, vwarp4, vtrack4;

        int idx = 0;
        // form of residual
        // r = [uuuu1,vvvv1,uuuu2,vvvv2, ... , u{n-1},v{n-1}, u{n},v{n}]^t \in R^{2*n_valid_point x 1}
        for (; idx < n_valid_point - 3; idx += 4) {
            int idx2 = 2 * idx;

            // calculate residual left edge
            uwarp4  = _mm_load_ps(buf_up_warped + idx);
            utrack4 = _mm_load_ps(buf_up_tracked + idx);

            vwarp4 = _mm_load_ps(buf_vp_warped + idx);
            vtrack4 = _mm_load_ps(buf_vp_tracked + idx);
            
            __m128 rrrru = _mm_sub_ps(uwarp4, utrack4);
            _mm_store_ps(buf_res_p + idx2, rrrru);

            __m128 rrrrv = _mm_sub_ps(vwarp4, vtrack4);
            _mm_store_ps(buf_res_p + idx2 + 4, rrrrv);

            // calculate weight left edge (Huber weight)
            float r = *(buf_res_p + idx2);
            *(buf_weight_p + idx2) = fabs(r) < huber_thres_p ? 1.0f : huber_thres_p / fabs(r);
            if (*(buf_weight_p + idx2) < 1.0f) ++n_overthres_point;

            r = *(buf_res_p + idx2 + 1);
            *(buf_weight_p + idx2 + 1) = fabs(r) < huber_thres_p ? 1.0f : huber_thres_p / fabs(r);
            if (*(buf_weight_p + idx2 + 1) < 1.0f) ++n_overthres_point;

            r = *(buf_res_p + idx2 + 2);
            *(buf_weight_p + idx2 + 2) = fabs(r) < huber_thres_p ? 1.0f : huber_thres_p / fabs(r);
            if (*(buf_weight_p + idx2 + 2) < 1.0f) ++n_overthres_point;

            r = *(buf_res_p + idx2 + 3);
            *(buf_weight_p + idx2 + 3) = fabs(r) < huber_thres_p ? 1.0f : huber_thres_p / fabs(r);
            if (*(buf_weight_p + idx2 + 3) < 1.0f) ++n_overthres_point;

            r = *(buf_res_p + idx2 + 4);
            *(buf_weight_p + idx2 + 4) = fabs(r) < huber_thres_p ? 1.0f : huber_thres_p / fabs(r);
            if (*(buf_weight_p + idx2 + 4) < 1.0f) ++n_overthres_point;

            r = *(buf_res_p + idx2 + 5);
            *(buf_weight_p + idx2 + 5) = fabs(r) < huber_thres_p ? 1.0f : huber_thres_p / fabs(r);
            if (*(buf_weight_p + idx2 + 5) < 1.0f) ++n_overthres_point;

            r = *(buf_res_p + idx2 + 6);
            *(buf_weight_p + idx2 + 6) = fabs(r) < huber_thres_p ? 1.0f : huber_thres_p / fabs(r);
            if (*(buf_weight_p + idx2 + 6) < 1.0f) ++n_overthres_point;

            r = *(buf_res_p + idx2 + 7);
            *(buf_weight_p + idx2 + 7) = fabs(r) < huber_thres_p ? 1.0f : huber_thres_p / fabs(r);
            if (*(buf_weight_p + idx2 + 7) < 1.0f) ++n_overthres_point;
        };
        for (; idx < n_valid_point; ++idx) {
            int idx2 = 2 * idx;
            float uwarp = *(buf_up_warped + idx);
            float utrack = *(buf_up_tracked + idx);
            float vwarp = *(buf_vp_warped + idx);
            float vtrack = *(buf_vp_tracked + idx);

            // calculate and push edge residual, Huber weight, Jacobian, and Hessian (따로하자)
            float r = uwarp - utrack;
            float w = fabs(r) < huber_thres_p ? 1.0f : huber_thres_p / fabs(r);

            *(buf_res_p + idx2) = r;
            *(buf_weight_p + idx2) = w;

            if (w < 1.0f) ++n_overthres_point;

            r = vwarp - vtrack;
            w = fabs(r) < huber_thres_p ? 1.0f : huber_thres_p / fabs(r);

            *(buf_res_p + idx2 + 1) = r;
            *(buf_weight_p + idx2 + 1) = w;

            if (w < 1.0f) ++n_overthres_point;
        };

      
    };

    void SE3TrackerSSE::calcPointHessianAndJacobianSSE() 
    {
        JtWJ_p.setZero();
        mJtWr_p.setZero();
        err_p = 0;

        __m128 ones = _mm_set1_ps(1.0f);
        __m128 zeros = _mm_set1_ps(0.0f);
        __m128 minusones = _mm_set1_ps(-1.0f);

        __m128 J1, J2, J3, J4, J5, J6; // four data at once.        
        __m128 fx4 = _mm_set1_ps(K[0](0, 0));
        __m128 fy4 = _mm_set1_ps(K[0](1, 1));

        int idx = 0;
        // uuuu vvvv order.
        for (; idx < n_valid_point - 3; idx += 4) {
            int idx2 = 2 * idx;
            __m128 x4 = _mm_load_ps(buf_xp_warped + idx);
            __m128 y4 = _mm_load_ps(buf_yp_warped + idx);
            __m128 invz4 = _mm_rcp_ps(_mm_load_ps(buf_zp_warped + idx));
            __m128 fxinvz4 = _mm_mul_ps(fx4, invz4);
            __m128 fyinvz4 = _mm_mul_ps(fy4, invz4);
            __m128 xinvz4 = _mm_mul_ps(x4, invz4);
            __m128 yinvz4 = _mm_mul_ps(y4, invz4);

            // J(1) = fx/z ;
            J1 = fxinvz4;
            // J(2) = 0;
            J2 = zeros;
            // J(3) = (-fu*x/z)/z;
            J3 = _mm_sub_ps(zeros, _mm_mul_ps( _mm_mul_ps(fx4, xinvz4), invz4));
            // J(4) = (fx*x*y)/z/z
            J4 = _mm_sub_ps(zeros, _mm_mul_ps(fx4, _mm_mul_ps(xinvz4, yinvz4)));
            // J(5) = fx*(1+xiz*xiz);
            J5 = _mm_mul_ps(fx4, _mm_add_ps(ones, _mm_mul_ps(xinvz4, xinvz4)));
            // J(6) = -fx*y/z;
            J6 = _mm_sub_ps(zeros, _mm_mul_ps(fxinvz4, y4));

            // update
            updatePointSSE(J1, J2, J3, J4, J5, J6, _mm_load_ps(buf_res_p + idx2), _mm_load_ps(buf_weight_p + idx2), err_p);

            // J(1) = fx/z ;
            J1 = zeros;
            // J(2) = 0;
            J2 = fyinvz4;
            // J(3) = (-fv*y/z)/z;
            J3 = _mm_sub_ps(zeros, _mm_mul_ps(_mm_mul_ps(fy4, yinvz4), invz4));
            // J(4) = -fy*(1 + y*y/z/z);
            J4 = _mm_sub_ps(zeros, _mm_mul_ps(fy4, _mm_add_ps(ones, _mm_mul_ps(yinvz4, yinvz4))));
            // J(5) = fy*x*y/z/z
            J5 = _mm_mul_ps(fy4, _mm_mul_ps(xinvz4, yinvz4));
            // J(6) = -fy*x/z;
            J6 = _mm_mul_ps(fyinvz4, x4);

            // update
            updatePointSSE(J1, J2, J3, J4, J5, J6, _mm_load_ps(buf_res_p + idx2 + 4), _mm_load_ps(buf_weight_p + idx2 + 4), err_p);
        };
        for (; idx < n_valid_point; ++idx) {
            int idx2 = 2 * idx;
            float x = *(buf_xel_warped + idx);
            float y = *(buf_yel_warped + idx);
            float iz = 1.0f / (*(buf_zel_warped + idx));

            float xy = x*y;
            float xiz = x*iz;
            float yiz = y*iz;

            float fxiz = fx*iz;
            float fyiz = fy*iz;

            Vec6 J;
            J(0) = fxiz;
            J(1) = 0;
            J(2) = (-fxiz*x) *iz;
            J(3) = -fx*xiz*yiz;
            J(4) = fx*(1.0f + xiz*xiz);
            J(5) = -fx*yiz;

            float r = *(buf_res_p + idx2);
            float w = *(buf_weight_p + idx2);
            JtWJ_p.noalias() += (J*J.transpose())*w;
            mJtWr_p.noalias() -= J*(r*w);
            err_p += r*r*w;

            J(0) = 0;
            J(1) = fyiz;
            J(2) = (-fyiz*y) *iz;
            J(3) = -fy*(1 + yiz*yiz);
            J(4) = fy*xiz*yiz;
            J(5) = fy*xiz;

            r = *(buf_res_p + idx2 + 1);
            w = *(buf_weight_p + idx2 + 1);
            JtWJ_p.noalias() += (J*J.transpose())*w;
            mJtWr_p.noalias() -= J*(r*w);
            err_p += r*r*w;
        };
    };
    void SE3TrackerSSE::updatePointSSE(
        const __m128 &J1, const __m128 &J2, const __m128 &J3, const __m128 &J4, const __m128 &J5, const __m128 &J6,
        const __m128& res, const __m128& weight, float& err)
    {
        //A.noalias() += J * J.transpose() * weight;
        memset(SSEData, 0, sizeof(float) * 4 * 28);

        __m128 J1w = _mm_mul_ps(J1, weight);
        _mm_store_ps(SSEData + 4 * 0, _mm_add_ps(_mm_load_ps(SSEData + 4 * 0), _mm_mul_ps(J1w, J1)));
        _mm_store_ps(SSEData + 4 * 1, _mm_add_ps(_mm_load_ps(SSEData + 4 * 1), _mm_mul_ps(J1w, J2)));
        _mm_store_ps(SSEData + 4 * 2, _mm_add_ps(_mm_load_ps(SSEData + 4 * 2), _mm_mul_ps(J1w, J3)));
        _mm_store_ps(SSEData + 4 * 3, _mm_add_ps(_mm_load_ps(SSEData + 4 * 3), _mm_mul_ps(J1w, J4)));
        _mm_store_ps(SSEData + 4 * 4, _mm_add_ps(_mm_load_ps(SSEData + 4 * 4), _mm_mul_ps(J1w, J5)));
        _mm_store_ps(SSEData + 4 * 5, _mm_add_ps(_mm_load_ps(SSEData + 4 * 5), _mm_mul_ps(J1w, J6)));


        __m128 J2w = _mm_mul_ps(J2, weight);
        _mm_store_ps(SSEData + 4 * 6, _mm_add_ps(_mm_load_ps(SSEData + 4 * 6), _mm_mul_ps(J2w, J2)));
        _mm_store_ps(SSEData + 4 * 7, _mm_add_ps(_mm_load_ps(SSEData + 4 * 7), _mm_mul_ps(J2w, J3)));
        _mm_store_ps(SSEData + 4 * 8, _mm_add_ps(_mm_load_ps(SSEData + 4 * 8), _mm_mul_ps(J2w, J4)));
        _mm_store_ps(SSEData + 4 * 9, _mm_add_ps(_mm_load_ps(SSEData + 4 * 9), _mm_mul_ps(J2w, J5)));
        _mm_store_ps(SSEData + 4 * 10, _mm_add_ps(_mm_load_ps(SSEData + 4 * 10), _mm_mul_ps(J2w, J6)));


        __m128 J3w = _mm_mul_ps(J3, weight);
        _mm_store_ps(SSEData + 4 * 11, _mm_add_ps(_mm_load_ps(SSEData + 4 * 11), _mm_mul_ps(J3w, J3)));
        _mm_store_ps(SSEData + 4 * 12, _mm_add_ps(_mm_load_ps(SSEData + 4 * 12), _mm_mul_ps(J3w, J4)));
        _mm_store_ps(SSEData + 4 * 13, _mm_add_ps(_mm_load_ps(SSEData + 4 * 13), _mm_mul_ps(J3w, J5)));
        _mm_store_ps(SSEData + 4 * 14, _mm_add_ps(_mm_load_ps(SSEData + 4 * 14), _mm_mul_ps(J3w, J6)));

        __m128 J4w = _mm_mul_ps(J4, weight);
        _mm_store_ps(SSEData + 4 * 15, _mm_add_ps(_mm_load_ps(SSEData + 4 * 15), _mm_mul_ps(J4w, J4)));
        _mm_store_ps(SSEData + 4 * 16, _mm_add_ps(_mm_load_ps(SSEData + 4 * 16), _mm_mul_ps(J4w, J5)));
        _mm_store_ps(SSEData + 4 * 17, _mm_add_ps(_mm_load_ps(SSEData + 4 * 17), _mm_mul_ps(J4w, J6)));


        __m128 J5w = _mm_mul_ps(J5, weight);
        _mm_store_ps(SSEData + 4 * 18, _mm_add_ps(_mm_load_ps(SSEData + 4 * 18), _mm_mul_ps(J5w, J5)));
        _mm_store_ps(SSEData + 4 * 19, _mm_add_ps(_mm_load_ps(SSEData + 4 * 19), _mm_mul_ps(J5w, J6)));


        __m128 J6w = _mm_mul_ps(J6, weight);
        _mm_store_ps(SSEData + 4 * 20, _mm_add_ps(_mm_load_ps(SSEData + 4 * 20), _mm_mul_ps(J6w, J6)));


        //b.noalias() -= J * (res * weight);
        __m128 resw = _mm_mul_ps(res, weight);
        _mm_store_ps(SSEData + 4 * 21, _mm_add_ps(_mm_load_ps(SSEData + 4 * 21), _mm_mul_ps(resw, J1)));
        _mm_store_ps(SSEData + 4 * 22, _mm_add_ps(_mm_load_ps(SSEData + 4 * 22), _mm_mul_ps(resw, J2)));
        _mm_store_ps(SSEData + 4 * 23, _mm_add_ps(_mm_load_ps(SSEData + 4 * 23), _mm_mul_ps(resw, J3)));
        _mm_store_ps(SSEData + 4 * 24, _mm_add_ps(_mm_load_ps(SSEData + 4 * 24), _mm_mul_ps(resw, J4)));
        _mm_store_ps(SSEData + 4 * 25, _mm_add_ps(_mm_load_ps(SSEData + 4 * 25), _mm_mul_ps(resw, J5)));
        _mm_store_ps(SSEData + 4 * 26, _mm_add_ps(_mm_load_ps(SSEData + 4 * 26), _mm_mul_ps(resw, J6)));

        //error += res * res * weight;
        _mm_store_ps(SSEData + 4 * 27, _mm_add_ps(_mm_load_ps(SSEData + 4 * 27), _mm_mul_ps(resw, res)));

        // update JtWJ_sse,
        JtWJ_p(0, 0) += (GET4(SSEData, 0, 0) + GET4(SSEData, 0, 1) + GET4(SSEData, 0, 2) + GET4(SSEData, 0, 3));
        JtWJ_p(1, 0) = (JtWJ_p(0, 1) += (GET4(SSEData, 1, 0) + GET4(SSEData, 1, 1) + GET4(SSEData, 1, 2) + GET4(SSEData, 1, 3)));
        JtWJ_p(2, 0) = (JtWJ_p(0, 2) += (GET4(SSEData, 2, 0) + GET4(SSEData, 2, 1) + GET4(SSEData, 2, 2) + GET4(SSEData, 2, 3)));
        JtWJ_p(3, 0) = (JtWJ_p(0, 3) += (GET4(SSEData, 3, 0) + GET4(SSEData, 3, 1) + GET4(SSEData, 3, 2) + GET4(SSEData, 3, 3)));
        JtWJ_p(4, 0) = (JtWJ_p(0, 4) += (GET4(SSEData, 4, 0) + GET4(SSEData, 4, 1) + GET4(SSEData, 4, 2) + GET4(SSEData, 4, 3)));
        JtWJ_p(5, 0) = (JtWJ_p(0, 5) += (GET4(SSEData, 5, 0) + GET4(SSEData, 5, 1) + GET4(SSEData, 5, 2) + GET4(SSEData, 5, 3)));

        JtWJ_p(1, 1) += (GET4(SSEData, 6, 0) + GET4(SSEData, 6, 1) + GET4(SSEData, 6, 2) + GET4(SSEData, 6, 3));
        JtWJ_p(2, 1) = (JtWJ_p(1, 2) += (GET4(SSEData, 7, 0) + GET4(SSEData, 7, 1) + GET4(SSEData, 7, 2) + GET4(SSEData, 7, 3)));
        JtWJ_p(3, 1) = (JtWJ_p(1, 3) += (GET4(SSEData, 8, 0) + GET4(SSEData, 8, 1) + GET4(SSEData, 8, 2) + GET4(SSEData, 8, 3)));
        JtWJ_p(4, 1) = (JtWJ_p(1, 4) += (GET4(SSEData, 9, 0) + GET4(SSEData, 9, 1) + GET4(SSEData, 9, 2) + GET4(SSEData, 9, 3)));
        JtWJ_p(5, 1) = (JtWJ_p(1, 5) += (GET4(SSEData, 10, 0) + GET4(SSEData, 10, 1) + GET4(SSEData, 10, 2) + GET4(SSEData, 10, 3)));

        JtWJ_p(2, 2) += (GET4(SSEData, 11, 0) + GET4(SSEData, 11, 1) + GET4(SSEData, 11, 2) + GET4(SSEData, 11, 3));
        JtWJ_p(3, 2) = (JtWJ_p(2, 3) += (GET4(SSEData, 12, 0) + GET4(SSEData, 12, 1) + GET4(SSEData, 12, 2) + GET4(SSEData, 12, 3)));
        JtWJ_p(4, 2) = (JtWJ_p(2, 4) += (GET4(SSEData, 13, 0) + GET4(SSEData, 13, 1) + GET4(SSEData, 13, 2) + GET4(SSEData, 13, 3)));
        JtWJ_p(5, 2) = (JtWJ_p(2, 5) += (GET4(SSEData, 14, 0) + GET4(SSEData, 14, 1) + GET4(SSEData, 14, 2) + GET4(SSEData, 14, 3)));

        JtWJ_p(3, 3) += (GET4(SSEData, 15, 0) + GET4(SSEData, 15, 1) + GET4(SSEData, 15, 2) + GET4(SSEData, 15, 3));
        JtWJ_p(4, 3) = (JtWJ_p(3, 4) += (GET4(SSEData, 16, 0) + GET4(SSEData, 16, 1) + GET4(SSEData, 16, 2) + GET4(SSEData, 16, 3)));
        JtWJ_p(5, 3) = (JtWJ_p(3, 5) += (GET4(SSEData, 17, 0) + GET4(SSEData, 17, 1) + GET4(SSEData, 17, 2) + GET4(SSEData, 17, 3)));

        JtWJ_p(4, 4) += (GET4(SSEData, 18, 0) + GET4(SSEData, 18, 1) + GET4(SSEData, 18, 2) + GET4(SSEData, 18, 3));
        JtWJ_p(5, 4) = (JtWJ_p(4, 5) += (GET4(SSEData, 19, 0) + GET4(SSEData, 19, 1) + GET4(SSEData, 19, 2) + GET4(SSEData, 19, 3)));

        JtWJ_p(5, 5) += (GET4(SSEData, 20, 0) + GET4(SSEData, 20, 1) + GET4(SSEData, 20, 2) + GET4(SSEData, 20, 3));

        // update  mJtWr_sse
        mJtWr_p(0) -= (GET4(SSEData, 21, 0) + GET4(SSEData, 21, 1) + GET4(SSEData, 21, 2) + GET4(SSEData, 21, 3));
        mJtWr_p(1) -= (GET4(SSEData, 22, 0) + GET4(SSEData, 22, 1) + GET4(SSEData, 22, 2) + GET4(SSEData, 22, 3));
        mJtWr_p(2) -= (GET4(SSEData, 23, 0) + GET4(SSEData, 23, 1) + GET4(SSEData, 23, 2) + GET4(SSEData, 23, 3));
        mJtWr_p(3) -= (GET4(SSEData, 24, 0) + GET4(SSEData, 24, 1) + GET4(SSEData, 24, 2) + GET4(SSEData, 24, 3));
        mJtWr_p(4) -= (GET4(SSEData, 25, 0) + GET4(SSEData, 25, 1) + GET4(SSEData, 25, 2) + GET4(SSEData, 25, 3));
        mJtWr_p(5) -= (GET4(SSEData, 26, 0) + GET4(SSEData, 26, 1) + GET4(SSEData, 26, 2) + GET4(SSEData, 26, 3));

        // update err.
        err += (GET4(SSEData, 27, 0) + GET4(SSEData, 27, 1) + GET4(SSEData, 27, 2) + GET4(SSEData, 27, 3));
    };







    SE3TrackerSSE::SE3TrackerSSE(Params* params_,
        const vector<int>& nrows_pyr, const vector<int>& ncols_pyr,
        const vector<Eigen::Matrix3d>& K_pyr_, const Eigen::Matrix4d& T_nlnr_)
    {
        params = params_;
        MAX_PYR_LVL = params->pyr.max_lvl;

        for (int lvl = 0; lvl < MAX_PYR_LVL; lvl++) {
            K[lvl] << K_pyr_[lvl](0, 0), K_pyr_[lvl](0, 1), K_pyr_[lvl](0, 2),
                K_pyr_[lvl](1, 0), K_pyr_[lvl](1, 1), K_pyr_[lvl](1, 2),
                K_pyr_[lvl](2, 0), K_pyr_[lvl](2, 1), K_pyr_[lvl](2, 2);
            Kinv[lvl] = K[lvl].inverse();
        }

        T_nlnr << T_nlnr_(0, 0), T_nlnr_(0, 1), T_nlnr_(0, 2), T_nlnr_(0, 3),
            T_nlnr_(1, 0), T_nlnr_(1, 1), T_nlnr_(1, 2), T_nlnr_(1, 3),
            T_nlnr_(2, 0), T_nlnr_(2, 1), T_nlnr_(2, 2), T_nlnr_(2, 3),
            0.0f, 0.0f, 0.0f, 1.0f;

        int w = ncols_pyr[0];
        int h = nrows_pyr[0];

        buf_xel_warped = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_yel_warped = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_zel_warped = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_xer_warped = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_yer_warped = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_zer_warped = (float*)custom_aligned_malloc(sizeof(float)*w*h);

        buf_uel_warped = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_vel_warped = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_uer_warped = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_ver_warped = (float*)custom_aligned_malloc(sizeof(float)*w*h);

        buf_uel_matched = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_vel_matched = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_uer_matched = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_ver_matched = (float*)custom_aligned_malloc(sizeof(float)*w*h);

        buf_gxl_matched = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_gyl_matched = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_gxr_matched = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_gyr_matched = (float*)custom_aligned_malloc(sizeof(float)*w*h);

        buf_res_l_e = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_weight_l_e = (float*)custom_aligned_malloc(sizeof(float)*w*h);

        buf_res_r_e = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_weight_r_e = (float*)custom_aligned_malloc(sizeof(float)*w*h);

        buf_xp_warped = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_yp_warped = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_zp_warped = (float*)custom_aligned_malloc(sizeof(float)*w*h);

        buf_up_warped = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_vp_warped = (float*)custom_aligned_malloc(sizeof(float)*w*h);

        buf_up_tracked = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_vp_tracked = (float*)custom_aligned_malloc(sizeof(float)*w*h);

        buf_res_p = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_weight_p = (float*)custom_aligned_malloc(sizeof(float)*w*h);

        SSEData = (float*)custom_aligned_malloc(sizeof(float) * 4 * 28);

        // initialize pointers
        detachRef();
        detachTree();
        detachFrameCur();
    };
    SE3TrackerSSE::~SE3TrackerSSE()
    {
        custom_aligned_free((void*)buf_xel_warped);
        custom_aligned_free((void*)buf_yel_warped);
        custom_aligned_free((void*)buf_zel_warped);
        custom_aligned_free((void*)buf_xer_warped);
        custom_aligned_free((void*)buf_yer_warped);
        custom_aligned_free((void*)buf_zer_warped);

        custom_aligned_free((void*)buf_uel_warped);
        custom_aligned_free((void*)buf_vel_warped);
        custom_aligned_free((void*)buf_uer_warped);
        custom_aligned_free((void*)buf_ver_warped);

        custom_aligned_free((void*)buf_uel_matched);
        custom_aligned_free((void*)buf_vel_matched);
        custom_aligned_free((void*)buf_uer_matched);
        custom_aligned_free((void*)buf_ver_matched);

        custom_aligned_free((void*)buf_gxl_matched);
        custom_aligned_free((void*)buf_gyl_matched);
        custom_aligned_free((void*)buf_gxr_matched);
        custom_aligned_free((void*)buf_gyr_matched);

        custom_aligned_free((void*)buf_res_l_e);
        custom_aligned_free((void*)buf_weight_l_e);
        custom_aligned_free((void*)buf_res_r_e);
        custom_aligned_free((void*)buf_weight_r_e);

        custom_aligned_free((void*)buf_xp_warped);
        custom_aligned_free((void*)buf_yp_warped);
        custom_aligned_free((void*)buf_zp_warped);

        custom_aligned_free((void*)buf_up_warped);
        custom_aligned_free((void*)buf_vp_warped);

        custom_aligned_free((void*)buf_up_tracked);
        custom_aligned_free((void*)buf_vp_tracked);

        custom_aligned_free((void*)buf_res_p);
        custom_aligned_free((void*)buf_weight_p);

        cout << "    SE3TrackerSSE tracker is deleted.\n";
    };
};



#endif