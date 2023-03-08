#ifndef _AFFINE_KLT_TRACKER_H_
#define _AFFINE_KLT_TRACKER_H_

#include <iostream>
#include <cstdlib>
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"  
#include "Eigen/Dense"
#include "immintrin.h"

#include "../frame/stereo_frames.h"
#include "../tracking/tracking_reference.h"


using namespace std;
namespace chk {

    /* KLT tracker parametrized by affine transform */
    class AffineKLTTracker {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        int ncols, nrows;
        int n_pts; // # of update candidate points.
        Eigen::Matrix3f K;
        float fx, fy, cx, cy, fxinv, fyinv;

        vector<chk::Point2f> patch;
        int win_sz; // half length of KLT window.
        int M; // # of elements in a patch. (= (2*win_sz+1)*(2*win_sz+1))

        // Hessian and Jacobian
        // We just solve Linear system JtWJ*delta_xi = mJtWr; where mJtWr = -J^t*W*r;
        // JtWJ matrix is guaranteed to be P.S.D and symmetric matrix.
        // Thus, we can efficiently solve this equation by Cholesky decomposition.
        // --> JtWJ.ldlt().solve(mJtWr);
        Mat66 JtWJ;
        Vec6 mJtWr;

        Mat66 JtJ;
        Vec6 JtJinvJt;

        float* errs_ssd; // tracking err (SSD)
        float* errs_ncc; // tracking err (NCC with considering affine transform)
        int* mask;      // tracking pass or fail.





        // For SSE (AVX2)
        Mat66 JtWJ_sse;
        Vec6 mJtWr_sse;

        Mat66 JtWJ_avx;
        Vec6 mJtWr_avx;

        float* upattern;
        float* vpattern;

        float* buf_up_ref;
        float* buf_vp_ref;
        float* buf_up_warp;
        float* buf_vp_warp;

        float* buf_Ik;
        float* buf_du_k;
        float* buf_dv_k;

        float* buf_Ic_warp;
        float* buf_du_c_warp;
        float* buf_dv_c_warp;

        float* buf_residual;
        float* buf_weight;

        float* SSEData; // [4 * 28] (for make JtWJ and JtWr)
        float* AVXData; // [8 * 28] (for make JtWJ and JtWr)
        // JtWJ = [
        // 0, *, *, *, *, *;
        // 1, 6, *, *, *, *;
        // 2, 7,11, *, *, *;
        // 3, 8,12,15, *, *;
        // 4, 9,13,16,18, *;
        // 5,10,14,17,19,20];
        // JtWr = [21,22,23,24,25,26]^t
        // err = [27];
        float* errs_ssd_sse;


        // extrinsically get!                
        Params* params;
        StereoFrame* frame_c;
        chk::TrackingReference* track_ref;

        AffineKLTTracker(Params* params_, const int& nrows_, const int& ncols_, const Eigen::Matrix3d& K_);
        ~AffineKLTTracker();

        void linkRef(chk::TrackingReference* track_ref_) { track_ref = track_ref_; };
        void linkFrameCur(StereoFrame* frame_) { frame_c = frame_; };
        void detachRef() { track_ref = nullptr; };
        void detachFrameCur() { frame_c = nullptr; };

        void fillPriors(const Eigen::Matrix4d& T_ck_); // by using warped information, fill point_params & pts_c_tracked.
        
        
        void trackAllRefPoints();



        inline void update(const Vec6& Jt, const float& r, const float& weight, float& err_ssd);
        void solveGaussNewtonStep(Vec6& delta);
        void trackForwardAdditiveSingle(
            const cv::Mat& I_k, const cv::Mat& I_c, const cv::Mat& du_c, const cv::Mat& dv_c, const chk::Vec2& pt_k,
            chk::Vec2& pt_c_tracked, chk::Vec6& point_params,
            float& err_ssd_, float& err_ncc_, int& mask_);

        void trackInverseCompositionalSingle(
            const cv::Mat& img_k, const cv::Mat& img_c, const cv::Mat& du_k, const cv::Mat& dv_k, const chk::Vec2& pt_k,
            chk::Vec2& pt_c_tracked, chk::Vec6& point_params,
            float& err_ssd_, float& err_ncc_, int& mask_);



        // SSE version
        void generateRefPointsSSE(const chk::Vec2& pt_k_);
        void warpPointsSSE(const chk::Vec6& params_, const chk::Vec2& pt_k_);
        void interpReferenceImageSSE(const cv::Mat& img_k);
        void calcResidualAndWeightSSE(const cv::Mat& img_c, const cv::Mat& du_c, const cv::Mat& dv_c);
        void calcHessianAndJacobianSSE(float& err_ssd_sse_);

        void updateSSE(const __m128 &J1, const __m128 &J2, const __m128 &J3, const __m128 &J4, const __m128 &J5, const __m128 &J6,
            const __m128& res, const __m128& weight, float& err_ssd_sse_);

        void solveGaussNewtonStepSSE(Vec6& delta);
        void trackForwardAdditiveSingleSSE(
            const cv::Mat& I_k, const cv::Mat& I_c, const cv::Mat& du_c, const cv::Mat& dv_c, const chk::Vec2& pt_k,
            const chk::Vec2& pt_k_warped, chk::Vec2& pt_c_tracked, chk::Vec6& point_params,
            float& err_ssd_, float& err_ncc_, int& mask_);



        // AVX2 version
        void generateRefPointsAVX2(const chk::Vec2& pt_k_);
        void warpPointsAVX2(const chk::Vec6& params_, const chk::Vec2& pt_k_);
        void interpReferenceImageAVX2(const cv::Mat& img_k);
        void calcResidualAndWeightAVX2(const cv::Mat& img_c, const cv::Mat& du_c, const cv::Mat& dv_c);
        void calcHessianAndJacobianAVX2(float& err_ssd_);

        void updateAVX2(const __m256 &J1, const __m256 &J2, const __m256 &J3, const __m256 &J4, const __m256 &J5, const __m256 &J6,
            const __m256& res, const __m256& weight, float& err_ssd_);

        void solveGaussNewtonStepAVX2(Vec6& delta);
        void trackForwardAdditiveSingleAVX2(
            const cv::Mat& I_k, const cv::Mat& I_c, const cv::Mat& du_c, const cv::Mat& dv_c, const chk::Vec2& pt_k,
            chk::Vec2& pt_c_tracked, chk::Vec6& point_params,
            float& err_ssd_, float& err_ncc_, int& mask_);
    };


    // SSE version
    /* ================================================================================
    ================================= Implementation ==================================
    =================================================================================== */

    void AffineKLTTracker::generateRefPointsSSE(const chk::Vec2& pt_k_) {
        __m128 uuuu = _mm_set1_ps(pt_k_(0));
        __m128 vvvv = _mm_set1_ps(pt_k_(1));

        __m128 upat, vpat;
        for (int i = 0; i < 20; i++) {
            // reference patch.
            int idx = i * 4;
            upat = _mm_load_ps(upattern + idx);
            vpat = _mm_load_ps(vpattern + idx);
            _mm_store_ps(buf_up_ref + idx, _mm_add_ps(upat, uuuu));
            _mm_store_ps(buf_vp_ref + idx, _mm_add_ps(vpat, vvvv));
        }
    };

    void AffineKLTTracker::warpPointsSSE(const chk::Vec6& params_, const chk::Vec2& pt_k_) {
        __m128 a1a1a1a1 = _mm_set1_ps(params_(0) + 1.0f); // 1 + a 
        __m128 bbbb     = _mm_set1_ps(params_(1)); // b
        __m128 cccc     = _mm_set1_ps(params_(2)); // c
        __m128 d1d1d1d1 = _mm_set1_ps(params_(3) + 1.0f); // 1 + d
        __m128 uplustu4 = _mm_set1_ps(params_(4) + pt_k_(0)); // tx + uk
        __m128 vplustv4 = _mm_set1_ps(params_(5) + pt_k_(1)); // ty + vk

        __m128 upat, vpat;
        for (int i = 0; i < 20; i++) {
            // current warped patch
            int idx = i * 4;
            upat = _mm_load_ps(upattern + idx);
            vpat = _mm_load_ps(vpattern + idx);
            
            _mm_store_ps(buf_up_warp + idx, 
                _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(upat, a1a1a1a1), _mm_mul_ps(vpat, bbbb)), uplustu4));
            _mm_store_ps(buf_vp_warp + idx, 
                _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(upat, cccc), _mm_mul_ps(vpat, d1d1d1d1)), vplustv4));
        }
    };
    void AffineKLTTracker::interpReferenceImageSSE(const cv::Mat& img_k) {
        for (int i = 0; i < 80; i++) {
            *(buf_Ik + i) = improc::interpImageSingle(img_k, *(buf_up_ref + i), *(buf_vp_ref + i));
        }
    };

    void AffineKLTTracker::calcResidualAndWeightSSE(const cv::Mat& img_c, const cv::Mat& du_c, const cv::Mat& dv_c) {

        chk::Point3f interp;
        for (int i = 0; i < 80; i++) {
            improc::interpImageSingle3(img_c,du_c,dv_c, *(buf_up_warp + i), *(buf_vp_warp + i), interp);
            *(buf_Ic_warp + i)   = interp.x;
            *(buf_du_c_warp + i) = interp.y;
            *(buf_dv_c_warp + i) = interp.z;

            *(buf_residual + i) = *(buf_Ic_warp + i) - *(buf_Ik + i); // residual
            float r = *(buf_residual + i);
            *(buf_weight + i) = abs(r) < 6.0f ? 1.0f : 6.0f / abs(r);
        }
    };

    void AffineKLTTracker::calcHessianAndJacobianSSE(float& err_ssd_) {
        __m128 J1, J2, J3, J4, J5, J6; // four data at once.
        __m128 du_warp4, dv_warp4, upattern4, vpattern4;
        for (int i = 0; i < 20; i++) {
            int idx = i * 4;
            du_warp4 = _mm_load_ps(buf_du_c_warp + idx);
            dv_warp4 = _mm_load_ps(buf_dv_c_warp + idx);
            upattern4 = _mm_load_ps(upattern + idx);
            vpattern4 = _mm_load_ps(vpattern + idx);

            // J(1) = du_warp*up;
            J1 = _mm_mul_ps(du_warp4, upattern4);

            // J(2) = du_warp*vp;
            J2 = _mm_mul_ps(du_warp4, vpattern4);

            // J(3) = dv_warp*up;
            J3 = _mm_mul_ps(dv_warp4, upattern4);

            // J(4) = dv_warp*vp;
            J4 = _mm_mul_ps(dv_warp4, vpattern4);

            // J(5) = du_warp;
            J5 = du_warp4;

            // J(6) = dv_warp;
            J6 = dv_warp4;

            // update .
            updateSSE(J1, J2, J3, J4, J5, J6, _mm_load_ps(buf_residual + idx), _mm_load_ps(buf_weight + idx), err_ssd_);
        }
    };

    void AffineKLTTracker::updateSSE(const __m128 &J1, const __m128 &J2, const __m128 &J3, const __m128 &J4, const __m128 &J5, const __m128 &J6,
        const __m128& res, const __m128& weight, float& err_ssd_) {
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

#define GET4(ptr,i,j) (*(ptr + 4 * i + j))
        // update JtWJ_sse,
        JtWJ_sse(0, 0) += (GET4(SSEData, 0, 0) + GET4(SSEData, 0, 1) + GET4(SSEData, 0, 2) + GET4(SSEData, 0, 3));
        JtWJ_sse(1, 0) = (JtWJ_sse(0, 1) += (GET4(SSEData, 1, 0) + GET4(SSEData, 1, 1) + GET4(SSEData, 1, 2) + GET4(SSEData, 1, 3)));
        JtWJ_sse(2, 0) = (JtWJ_sse(0, 2) += (GET4(SSEData, 2, 0) + GET4(SSEData, 2, 1) + GET4(SSEData, 2, 2) + GET4(SSEData, 2, 3)));
        JtWJ_sse(3, 0) = (JtWJ_sse(0, 3) += (GET4(SSEData, 3, 0) + GET4(SSEData, 3, 1) + GET4(SSEData, 3, 2) + GET4(SSEData, 3, 3)));
        JtWJ_sse(4, 0) = (JtWJ_sse(0, 4) += (GET4(SSEData, 4, 0) + GET4(SSEData, 4, 1) + GET4(SSEData, 4, 2) + GET4(SSEData, 4, 3)));
        JtWJ_sse(5, 0) = (JtWJ_sse(0, 5) += (GET4(SSEData, 5, 0) + GET4(SSEData, 5, 1) + GET4(SSEData, 5, 2) + GET4(SSEData, 5, 3)));

        JtWJ_sse(1, 1) += (GET4(SSEData, 6, 0) + GET4(SSEData, 6, 1) + GET4(SSEData, 6, 2) + GET4(SSEData, 6, 3));
        JtWJ_sse(2, 1) = (JtWJ_sse(1, 2) += (GET4(SSEData, 7, 0) + GET4(SSEData, 7, 1) + GET4(SSEData, 7, 2) + GET4(SSEData, 7, 3)));
        JtWJ_sse(3, 1) = (JtWJ_sse(1, 3) += (GET4(SSEData, 8, 0) + GET4(SSEData, 8, 1) + GET4(SSEData, 8, 2) + GET4(SSEData, 8, 3)));
        JtWJ_sse(4, 1) = (JtWJ_sse(1, 4) += (GET4(SSEData, 9, 0) + GET4(SSEData, 9, 1) + GET4(SSEData, 9, 2) + GET4(SSEData, 9, 3)));
        JtWJ_sse(5, 1) = (JtWJ_sse(1, 5) += (GET4(SSEData, 10, 0) + GET4(SSEData, 10, 1) + GET4(SSEData, 10, 2) + GET4(SSEData, 10, 3)));

        JtWJ_sse(2, 2) += (GET4(SSEData, 11, 0) + GET4(SSEData, 11, 1) + GET4(SSEData, 11, 2) + GET4(SSEData, 11, 3));
        JtWJ_sse(3, 2) = (JtWJ_sse(2, 3) += (GET4(SSEData, 12, 0) + GET4(SSEData, 12, 1) + GET4(SSEData, 12, 2) + GET4(SSEData, 12, 3)));
        JtWJ_sse(4, 2) = (JtWJ_sse(2, 4) += (GET4(SSEData, 13, 0) + GET4(SSEData, 13, 1) + GET4(SSEData, 13, 2) + GET4(SSEData, 13, 3)));
        JtWJ_sse(5, 2) = (JtWJ_sse(2, 5) += (GET4(SSEData, 14, 0) + GET4(SSEData, 14, 1) + GET4(SSEData, 14, 2) + GET4(SSEData, 14, 3)));

        JtWJ_sse(3, 3) += (GET4(SSEData, 15, 0) + GET4(SSEData, 15, 1) + GET4(SSEData, 15, 2) + GET4(SSEData, 15, 3));
        JtWJ_sse(4, 3) = (JtWJ_sse(3, 4) += (GET4(SSEData, 16, 0) + GET4(SSEData, 16, 1) + GET4(SSEData, 16, 2) + GET4(SSEData, 16, 3)));
        JtWJ_sse(5, 3) = (JtWJ_sse(3, 5) += (GET4(SSEData, 17, 0) + GET4(SSEData, 17, 1) + GET4(SSEData, 17, 2) + GET4(SSEData, 17, 3)));

        JtWJ_sse(4, 4) += (GET4(SSEData, 18, 0) + GET4(SSEData, 18, 1) + GET4(SSEData, 18, 2) + GET4(SSEData, 18, 3));
        JtWJ_sse(5, 4) = (JtWJ_sse(4, 5) += (GET4(SSEData, 19, 0) + GET4(SSEData, 19, 1) + GET4(SSEData, 19, 2) + GET4(SSEData, 19, 3)));

        JtWJ_sse(5, 5) += (GET4(SSEData, 20, 0) + GET4(SSEData, 20, 1) + GET4(SSEData, 20, 2) + GET4(SSEData, 20, 3));

        // update  mJtWr_sse
        mJtWr_sse(0) -= (GET4(SSEData, 21, 0) + GET4(SSEData, 21, 1) + GET4(SSEData, 21, 2) + GET4(SSEData, 21, 3));
        mJtWr_sse(1) -= (GET4(SSEData, 22, 0) + GET4(SSEData, 22, 1) + GET4(SSEData, 22, 2) + GET4(SSEData, 22, 3));
        mJtWr_sse(2) -= (GET4(SSEData, 23, 0) + GET4(SSEData, 23, 1) + GET4(SSEData, 23, 2) + GET4(SSEData, 23, 3));
        mJtWr_sse(3) -= (GET4(SSEData, 24, 0) + GET4(SSEData, 24, 1) + GET4(SSEData, 24, 2) + GET4(SSEData, 24, 3));
        mJtWr_sse(4) -= (GET4(SSEData, 25, 0) + GET4(SSEData, 25, 1) + GET4(SSEData, 25, 2) + GET4(SSEData, 25, 3));
        mJtWr_sse(5) -= (GET4(SSEData, 26, 0) + GET4(SSEData, 26, 1) + GET4(SSEData, 26, 2) + GET4(SSEData, 26, 3));

        // update err.
        err_ssd_ += (GET4(SSEData, 27, 0) + GET4(SSEData, 27, 1) + GET4(SSEData, 27, 2) + GET4(SSEData, 27, 3));
    };
    void AffineKLTTracker::solveGaussNewtonStepSSE(Vec6& delta)
    {
        delta = JtWJ_sse.ldlt().solve(mJtWr_sse);
    };

    void AffineKLTTracker::trackForwardAdditiveSingleSSE(const cv::Mat& I_k, const cv::Mat& I_c, const cv::Mat& du_c, const cv::Mat& dv_c, const chk::Vec2& pt_k,
        const chk::Vec2& pt_k_warped, chk::Vec2& pt_c_tracked, chk::Vec6& point_params,
        float& err_ssd_, float& err_ncc_, int& mask_)
    {
        // initialize
        err_ssd_ = 0.0f;
        err_ncc_ = 0.0f;
        mask_ = 0;

        // priors: pt_c_tracked, point_params(5:6) = pt_c_tracked - pt_k;
        // (in a case of already tracked, point_params(1:4) are also filled out.)
        // point_params(5:6) -> pt_c_tracked

        if ((point_params(4) + pt_k(0) < 1) || (point_params(4) + pt_k(0) > ncols - 1)
            || (point_params(5) + pt_k(1) < 1) || (point_params(5) + pt_k(1) > nrows - 1)) {
            err_ssd_ = 100.0f;
            err_ncc_ = -1.0f;
            return;
        }

        generateRefPointsSSE(pt_k);
        interpReferenceImageSSE(I_k);

        int MAX_ITER = 50;
        int thres_huber = 5.0;
        float area = 0;
        for (int iter = 0; iter < MAX_ITER; iter++)
        {
            // initialize
            JtWJ_sse.setZero();
            mJtWr_sse.setZero();
            err_ssd_ = 0.0f;
            
            warpPointsSSE(point_params, pt_k);
            calcResidualAndWeightSSE(I_c, du_c, dv_c);
            calcHessianAndJacobianSSE(err_ssd_);

            err_ssd_ = sqrt(err_ssd_ / 80.0f);

            // update tracking point.
            chk::Vec6 delta;
            solveGaussNewtonStepSSE(delta);
            point_params += delta;

            // break condition, point_params(5:6).norm() < 1e-4
            float delta_norm = sqrt(delta(4)*delta(4) + delta(5)*delta(5));
            area = abs((1 + point_params(0))*(1 + point_params(3)) - point_params(1)*point_params(2));
            /*cout << "pt:[" << pt_k(0) << "," << pt_k(1) << "],ptc:[" << point_params(4) + pt_k(0) << "," << point_params(5) + pt_k(1)
            <<"],abcd:["<<point_params(0)<<","<< point_params(1)<<","<< point_params(2)<<","<< point_params(3)
            << "],err:" << err_ssd_ <<"area:"<< area << endl;*/
            if (delta_norm < 1e-3f) break;
        };


        // update point coordinate
        pt_c_tracked(0) = pt_k(0) + point_params(4);
        pt_c_tracked(1) = pt_k(1) + point_params(5);

        // validity test(1)- if the tracked pixel is far from the original projection, reject.
        if ((pt_c_tracked - pt_k_warped).norm() > 4.0f) return;
        else if (err_ssd_ > 20) return; // validity test(2)- ssd > 20, reject it.
        else if ((area > 1.3) || (area < 0.7)) return; // validity test(3) - deformation is too large, reject it.
        
        // all tests are passed.
        mask_ = 1;
    };



    // AVX2 version
    /* ================================================================================
    ================================= Implementation ==================================
    =================================================================================== */
    void AffineKLTTracker::generateRefPointsAVX2(const chk::Vec2& pt_k_) {
        __m256 uuuuuuuu = _mm256_set1_ps(pt_k_(0));
        __m256 vvvvvvvv = _mm256_set1_ps(pt_k_(1));

        __m256 upat, vpat;
        for (int i = 0; i < 10; i++) {
            // reference patch.
            int idx = i * 8;
            upat = _mm256_load_ps(upattern + idx);
            vpat = _mm256_load_ps(vpattern + idx);
            _mm256_store_ps(buf_up_ref + idx, _mm256_add_ps(upat, uuuuuuuu));
            _mm256_store_ps(buf_vp_ref + idx, _mm256_add_ps(vpat, vvvvvvvv));
        }
    };
    void AffineKLTTracker::interpReferenceImageAVX2(const cv::Mat& img_k) {
        for (int i = 0; i < 80; i++) {
            *(buf_Ik + i) = improc::interpImageSingle(img_k, *(buf_up_ref + i), *(buf_vp_ref + i));
        }
    };

    void AffineKLTTracker::warpPointsAVX2(const chk::Vec6& params_, const chk::Vec2& pt_k_) {
 
        __m256 a1a1a1a1a1a1a1a1 = _mm256_set1_ps(params_(0) + 1.0f); // 1 + a 
        __m256 bbbbbbbb = _mm256_set1_ps(params_(1)); // b 
        __m256 cccccccc = _mm256_set1_ps(params_(2)); // c 
        __m256 d1d1d1d1d1d1d1d1 = _mm256_set1_ps(params_(3) + 1.0f); // 1 + d
        __m256 uplustu8 = _mm256_set1_ps(params_(4) + pt_k_(0)); // tu + uk
        __m256 vplustv8 = _mm256_set1_ps(params_(5) + pt_k_(1)); // tv + vk

        __m256 upat, vpat;
        for (int i = 0; i < 10; i++) {
            // current warped patch
            int idx = i * 8;
            _mm256_store_ps(buf_up_warp + idx,
                _mm256_add_ps(
                    _mm256_add_ps(_mm256_mul_ps(upat, a1a1a1a1a1a1a1a1), _mm256_mul_ps(vpat, bbbbbbbb)), uplustu8));
            _mm256_store_ps(buf_vp_warp + idx,
                _mm256_add_ps(
                    _mm256_add_ps(_mm256_mul_ps(upat, cccccccc), _mm256_mul_ps(vpat, d1d1d1d1d1d1d1d1)), vplustv8));
        }
    };
    
    void AffineKLTTracker::calcResidualAndWeightAVX2(const cv::Mat& img_c, const cv::Mat& du_c, const cv::Mat& dv_c) {

        for (int i = 0; i < 80; i++) {
            *(buf_Ic_warp + i) = improc::interpImageSingle(img_c, *(buf_up_warp + i), *(buf_vp_warp + i));
            *(buf_du_c_warp + i) = improc::interpImageSingle(du_c, *(buf_up_warp + i), *(buf_vp_warp + i));
            *(buf_dv_c_warp + i) = improc::interpImageSingle(dv_c, *(buf_up_warp + i), *(buf_vp_warp + i));

            *(buf_residual + i) = *(buf_Ic_warp + i) - *(buf_Ik + i); // residual
            float r = *(buf_residual + i);
            *(buf_weight + i) = abs(r) < 6.0f ? 1.0f : 6.0f / abs(r);
        }
    };

    void AffineKLTTracker::calcHessianAndJacobianAVX2(float& err_ssd_) {
        __m256 J1, J2, J3, J4, J5, J6; // four data at once.
        __m256 du_warp8, dv_warp8, upattern8, vpattern8;
        for (int i = 0; i < 10; i++) {
            int idx = i * 8;
            du_warp8 = _mm256_load_ps(buf_du_c_warp + idx);
            dv_warp8 = _mm256_load_ps(buf_dv_c_warp + idx);
            upattern8 = _mm256_load_ps(upattern + idx);
            vpattern8 = _mm256_load_ps(vpattern + idx);

            // J(1) = du_warp*up;
            J1 = _mm256_mul_ps(du_warp8, upattern8);

            // J(2) = du_warp*vp;
            J2 = _mm256_mul_ps(du_warp8, vpattern8);

            // J(3) = dv_warp*up;
            J3 = _mm256_mul_ps(dv_warp8, upattern8);

            // J(4) = dv_warp*vp;
            J4 = _mm256_mul_ps(dv_warp8, vpattern8);

            // J(5) = du_warp;
            J5 = du_warp8;

            // J(6) = dv_warp;
            J6 = dv_warp8;

            // update .
            updateAVX2(J1, J2, J3, J4, J5, J6, _mm256_load_ps(buf_residual + idx), _mm256_load_ps(buf_weight + idx), err_ssd_);
        }
    };

    void AffineKLTTracker::updateAVX2(const __m256 &J1, const __m256 &J2, const __m256 &J3, const __m256 &J4, const __m256 &J5, const __m256 &J6,
        const __m256& res, const __m256& weight, float& err_ssd_) {
        //A.noalias() += J * J.transpose() * weight;
        memset(AVXData, 0, sizeof(float) * 8 * 28);

        __m256 J1w = _mm256_mul_ps(J1, weight);
        _mm256_store_ps(AVXData + 8 * 0, _mm256_add_ps(_mm256_load_ps(AVXData + 8 * 0), _mm256_mul_ps(J1w, J1)));
        _mm256_store_ps(AVXData + 8 * 1, _mm256_add_ps(_mm256_load_ps(AVXData + 8 * 1), _mm256_mul_ps(J1w, J2)));
        _mm256_store_ps(AVXData + 8 * 2, _mm256_add_ps(_mm256_load_ps(AVXData + 8 * 2), _mm256_mul_ps(J1w, J3)));
        _mm256_store_ps(AVXData + 8 * 3, _mm256_add_ps(_mm256_load_ps(AVXData + 8 * 3), _mm256_mul_ps(J1w, J4)));
        _mm256_store_ps(AVXData + 8 * 4, _mm256_add_ps(_mm256_load_ps(AVXData + 8 * 4), _mm256_mul_ps(J1w, J5)));
        _mm256_store_ps(AVXData + 8 * 5, _mm256_add_ps(_mm256_load_ps(AVXData + 8 * 5), _mm256_mul_ps(J1w, J6)));

        __m256 J2w = _mm256_mul_ps(J2, weight);
        _mm256_store_ps(AVXData + 8 * 6, _mm256_add_ps(_mm256_load_ps(AVXData + 8 * 6), _mm256_mul_ps(J2w, J2)));
        _mm256_store_ps(AVXData + 8 * 7, _mm256_add_ps(_mm256_load_ps(AVXData + 8 * 7), _mm256_mul_ps(J2w, J3)));
        _mm256_store_ps(AVXData + 8 * 8, _mm256_add_ps(_mm256_load_ps(AVXData + 8 * 8), _mm256_mul_ps(J2w, J4)));
        _mm256_store_ps(AVXData + 8 * 9, _mm256_add_ps(_mm256_load_ps(AVXData + 8 * 9), _mm256_mul_ps(J2w, J5)));
        _mm256_store_ps(AVXData + 8 * 10, _mm256_add_ps(_mm256_load_ps(AVXData + 8 * 10), _mm256_mul_ps(J2w, J6)));

        __m256 J3w = _mm256_mul_ps(J3, weight);
        _mm256_store_ps(AVXData + 8 * 11, _mm256_add_ps(_mm256_load_ps(AVXData + 8 * 11), _mm256_mul_ps(J3w, J3)));
        _mm256_store_ps(AVXData + 8 * 12, _mm256_add_ps(_mm256_load_ps(AVXData + 8 * 12), _mm256_mul_ps(J3w, J4)));
        _mm256_store_ps(AVXData + 8 * 13, _mm256_add_ps(_mm256_load_ps(AVXData + 8 * 13), _mm256_mul_ps(J3w, J5)));
        _mm256_store_ps(AVXData + 8 * 14, _mm256_add_ps(_mm256_load_ps(AVXData + 8 * 14), _mm256_mul_ps(J3w, J6)));

        __m256 J4w = _mm256_mul_ps(J4, weight);
        _mm256_store_ps(AVXData + 8 * 15, _mm256_add_ps(_mm256_load_ps(AVXData + 8 * 15), _mm256_mul_ps(J4w, J4)));
        _mm256_store_ps(AVXData + 8 * 16, _mm256_add_ps(_mm256_load_ps(AVXData + 8 * 16), _mm256_mul_ps(J4w, J5)));
        _mm256_store_ps(AVXData + 8 * 17, _mm256_add_ps(_mm256_load_ps(AVXData + 8 * 17), _mm256_mul_ps(J4w, J6)));

        __m256 J5w = _mm256_mul_ps(J5, weight);
        _mm256_store_ps(AVXData + 8 * 18, _mm256_add_ps(_mm256_load_ps(AVXData + 8 * 18), _mm256_mul_ps(J5w, J5)));
        _mm256_store_ps(AVXData + 8 * 19, _mm256_add_ps(_mm256_load_ps(AVXData + 8 * 19), _mm256_mul_ps(J5w, J6)));

        __m256 J6w = _mm256_mul_ps(J6, weight);
        _mm256_store_ps(AVXData + 8 * 20, _mm256_add_ps(_mm256_load_ps(AVXData + 8 * 20), _mm256_mul_ps(J6w, J6)));

        //b.noalias() -= J * (res * weight);
        __m256 resw = _mm256_mul_ps(res, weight);
        _mm256_store_ps(AVXData + 8 * 21, _mm256_add_ps(_mm256_load_ps(AVXData + 8 * 21), _mm256_mul_ps(resw, J1)));
        _mm256_store_ps(AVXData + 8 * 22, _mm256_add_ps(_mm256_load_ps(AVXData + 8 * 22), _mm256_mul_ps(resw, J2)));
        _mm256_store_ps(AVXData + 8 * 23, _mm256_add_ps(_mm256_load_ps(AVXData + 8 * 23), _mm256_mul_ps(resw, J3)));
        _mm256_store_ps(AVXData + 8 * 24, _mm256_add_ps(_mm256_load_ps(AVXData + 8 * 24), _mm256_mul_ps(resw, J4)));
        _mm256_store_ps(AVXData + 8 * 25, _mm256_add_ps(_mm256_load_ps(AVXData + 8 * 25), _mm256_mul_ps(resw, J5)));
        _mm256_store_ps(AVXData + 8 * 26, _mm256_add_ps(_mm256_load_ps(AVXData + 8 * 26), _mm256_mul_ps(resw, J6)));

        //error += res * res * weight;
        _mm256_store_ps(AVXData + 8 * 27, _mm256_add_ps(_mm256_load_ps(AVXData + 8 * 27), _mm256_mul_ps(resw, res)));

#define GET8(ptr,i,j) (*(ptr + 8 * i + j))
        // update JtWJ_avx,
        JtWJ_avx(0, 0) += (GET8(AVXData, 0, 0) + GET8(AVXData, 0, 1) + GET8(AVXData, 0, 2) + GET8(AVXData, 0, 3) + GET8(AVXData, 0, 4) + GET8(AVXData, 0, 5) + GET8(AVXData, 0, 6) + GET8(AVXData, 0, 7));
        JtWJ_avx(1, 0) = (JtWJ_avx(0, 1) += (GET8(AVXData, 1, 0) + GET8(AVXData, 1, 1) + GET8(AVXData, 1, 2) + GET8(AVXData, 1, 3) + GET8(AVXData, 1, 4) + GET8(AVXData, 1, 5) + GET8(AVXData, 1, 6) + GET8(AVXData, 1, 7)));
        JtWJ_avx(2, 0) = (JtWJ_avx(0, 2) += (GET8(AVXData, 2, 0) + GET8(AVXData, 2, 1) + GET8(AVXData, 2, 2) + GET8(AVXData, 2, 3) + GET8(AVXData, 2, 4) + GET8(AVXData, 2, 5) + GET8(AVXData, 2, 6) + GET8(AVXData, 2, 7)));
        JtWJ_avx(3, 0) = (JtWJ_avx(0, 3) += (GET8(AVXData, 3, 0) + GET8(AVXData, 3, 1) + GET8(AVXData, 3, 2) + GET8(AVXData, 3, 3) + GET8(AVXData, 3, 4) + GET8(AVXData, 3, 5) + GET8(AVXData, 3, 6) + GET8(AVXData, 3, 7)));
        JtWJ_avx(4, 0) = (JtWJ_avx(0, 4) += (GET8(AVXData, 4, 0) + GET8(AVXData, 4, 1) + GET8(AVXData, 4, 2) + GET8(AVXData, 4, 3) + GET8(AVXData, 4, 4) + GET8(AVXData, 4, 5) + GET8(AVXData, 4, 6) + GET8(AVXData, 4, 7)));
        JtWJ_avx(5, 0) = (JtWJ_avx(0, 5) += (GET8(AVXData, 5, 0) + GET8(AVXData, 5, 1) + GET8(AVXData, 5, 2) + GET8(AVXData, 5, 3) + GET8(AVXData, 5, 4) + GET8(AVXData, 5, 5) + GET8(AVXData, 5, 6) + GET8(AVXData, 5, 7)));

        JtWJ_avx(1, 1) += (GET8(AVXData, 6, 0) + GET8(AVXData, 6, 1) + GET8(AVXData, 6, 2) + GET8(AVXData, 6, 3) + GET8(AVXData, 6, 4) + GET8(AVXData, 6, 5) + GET8(AVXData, 6, 6) + GET8(AVXData, 6, 7));
        JtWJ_avx(2, 1) = (JtWJ_avx(1, 2) += (GET8(AVXData, 7, 0) + GET8(AVXData, 7, 1) + GET8(AVXData, 7, 2) + GET8(AVXData, 7, 3) + GET8(AVXData, 7, 4) + GET8(AVXData, 7, 5) + GET8(AVXData, 7, 6) + GET8(AVXData, 7, 7)));
        JtWJ_avx(3, 1) = (JtWJ_avx(1, 3) += (GET8(AVXData, 8, 0) + GET8(AVXData, 8, 1) + GET8(AVXData, 8, 2) + GET8(AVXData, 8, 3) + GET8(AVXData, 8, 4) + GET8(AVXData, 8, 5) + GET8(AVXData, 8, 6) + GET8(AVXData, 8, 7)));
        JtWJ_avx(4, 1) = (JtWJ_avx(1, 4) += (GET8(AVXData, 9, 0) + GET8(AVXData, 9, 1) + GET8(AVXData, 9, 2) + GET8(AVXData, 9, 3) + GET8(AVXData, 9, 4) + GET8(AVXData, 9, 5) + GET8(AVXData, 9, 6) + GET8(AVXData, 9, 7)));
        JtWJ_avx(5, 1) = (JtWJ_avx(1, 5) += (GET8(AVXData, 10, 0) + GET8(AVXData, 10, 1) + GET8(AVXData, 10, 2) + GET8(AVXData, 10, 3) + GET8(AVXData, 10, 4) + GET8(AVXData, 10, 5) + GET8(AVXData, 10, 6) + GET8(AVXData, 10, 7)));

        JtWJ_avx(2, 2) += (GET8(AVXData, 11, 0) + GET8(AVXData, 11, 1) + GET8(AVXData, 11, 2) + GET8(AVXData, 11, 3) + GET8(AVXData, 11, 4) + GET8(AVXData, 11, 5) + GET8(AVXData, 11, 6) + GET8(AVXData, 11, 7));
        JtWJ_avx(3, 2) = (JtWJ_avx(2, 3) += (GET8(AVXData, 12, 0) + GET8(AVXData, 12, 1) + GET8(AVXData, 12, 2) + GET8(AVXData, 12, 3) + GET8(AVXData, 12, 4) + GET8(AVXData, 12, 5) + GET8(AVXData, 12, 6) + GET8(AVXData, 12, 7)));
        JtWJ_avx(4, 2) = (JtWJ_avx(2, 4) += (GET8(AVXData, 13, 0) + GET8(AVXData, 13, 1) + GET8(AVXData, 13, 2) + GET8(AVXData, 13, 3) + GET8(AVXData, 13, 4) + GET8(AVXData, 13, 5) + GET8(AVXData, 13, 6) + GET8(AVXData, 13, 7)));
        JtWJ_avx(5, 2) = (JtWJ_avx(2, 5) += (GET8(AVXData, 14, 0) + GET8(AVXData, 14, 1) + GET8(AVXData, 14, 2) + GET8(AVXData, 14, 3) + GET8(AVXData, 14, 4) + GET8(AVXData, 14, 5) + GET8(AVXData, 14, 6) + GET8(AVXData, 14, 7)));

        JtWJ_avx(3, 3) += (GET8(AVXData, 15, 0) + GET8(AVXData, 15, 1) + GET8(AVXData, 15, 2) + GET8(AVXData, 15, 3) + GET8(AVXData, 15, 4) + GET8(AVXData, 15, 5) + GET8(AVXData, 15, 6) + GET8(AVXData, 15, 7));
        JtWJ_avx(4, 3) = (JtWJ_avx(3, 4) += (GET8(AVXData, 16, 0) + GET8(AVXData, 16, 1) + GET8(AVXData, 16, 2) + GET8(AVXData, 16, 3) + GET8(AVXData, 16, 4) + GET8(AVXData, 16, 5) + GET8(AVXData, 16, 6) + GET8(AVXData, 16, 7)));
        JtWJ_avx(5, 3) = (JtWJ_avx(3, 5) += (GET8(AVXData, 17, 0) + GET8(AVXData, 17, 1) + GET8(AVXData, 17, 2) + GET8(AVXData, 17, 3) + GET8(AVXData, 17, 4) + GET8(AVXData, 17, 5) + GET8(AVXData, 17, 6) + GET8(AVXData, 17, 7)));

        JtWJ_avx(4, 4) += (GET8(AVXData, 18, 0) + GET8(AVXData, 18, 1) + GET8(AVXData, 18, 2) + GET8(AVXData, 18, 3) + GET8(AVXData, 18, 4) + GET8(AVXData, 18, 5) + GET8(AVXData, 18, 6) + GET8(AVXData, 18, 7));
        JtWJ_avx(5, 4) = (JtWJ_avx(4, 5) += (GET8(AVXData, 19, 0) + GET8(AVXData, 19, 1) + GET8(AVXData, 19, 2) + GET8(AVXData, 19, 3) + GET8(AVXData, 19, 4) + GET8(AVXData, 19, 5) + GET8(AVXData, 19, 6) + GET8(AVXData, 19, 7)));

        JtWJ_avx(5, 5) += (GET8(AVXData, 20, 0) + GET8(AVXData, 20, 1) + GET8(AVXData, 20, 2) + GET8(AVXData, 20, 3) + GET8(AVXData, 20, 4) + GET8(AVXData, 20, 5) + GET8(AVXData, 20, 6) + GET8(AVXData, 20, 7));

        // update  mJtWr_sse
        mJtWr_avx(0) -= (GET8(AVXData, 21, 0) + GET8(AVXData, 21, 1) + GET8(AVXData, 21, 2) + GET8(AVXData, 21, 3) + GET8(AVXData, 21, 4) + GET8(AVXData, 21, 5) + GET8(AVXData, 21, 6) + GET8(AVXData, 21, 7));
        mJtWr_avx(1) -= (GET8(AVXData, 22, 0) + GET8(AVXData, 22, 1) + GET8(AVXData, 22, 2) + GET8(AVXData, 22, 3) + GET8(AVXData, 22, 4) + GET8(AVXData, 22, 5) + GET8(AVXData, 22, 6) + GET8(AVXData, 22, 7));
        mJtWr_avx(2) -= (GET8(AVXData, 23, 0) + GET8(AVXData, 23, 1) + GET8(AVXData, 23, 2) + GET8(AVXData, 23, 3) + GET8(AVXData, 23, 4) + GET8(AVXData, 23, 5) + GET8(AVXData, 23, 6) + GET8(AVXData, 23, 7));
        mJtWr_avx(3) -= (GET8(AVXData, 24, 0) + GET8(AVXData, 24, 1) + GET8(AVXData, 24, 2) + GET8(AVXData, 24, 3) + GET8(AVXData, 24, 4) + GET8(AVXData, 24, 5) + GET8(AVXData, 24, 6) + GET8(AVXData, 24, 7));
        mJtWr_avx(4) -= (GET8(AVXData, 25, 0) + GET8(AVXData, 25, 1) + GET8(AVXData, 25, 2) + GET8(AVXData, 25, 3) + GET8(AVXData, 25, 4) + GET8(AVXData, 25, 5) + GET8(AVXData, 25, 6) + GET8(AVXData, 25, 7));
        mJtWr_avx(5) -= (GET8(AVXData, 26, 0) + GET8(AVXData, 26, 1) + GET8(AVXData, 26, 2) + GET8(AVXData, 26, 3) + GET8(AVXData, 26, 4) + GET8(AVXData, 26, 5) + GET8(AVXData, 26, 6) + GET8(AVXData, 26, 7));

        // update err.
        err_ssd_ += (GET8(AVXData, 27, 0) + GET8(AVXData, 27, 1) + GET8(AVXData, 27, 2) + GET8(AVXData, 27, 3) + GET8(AVXData, 27, 4) + GET8(AVXData, 27, 5) + GET8(AVXData, 27, 6) + GET8(AVXData, 27, 7));
    };
    void AffineKLTTracker::solveGaussNewtonStepAVX2(Vec6& delta)
    {
        delta = JtWJ_avx.ldlt().solve(mJtWr_avx);
    };

    void AffineKLTTracker::trackForwardAdditiveSingleAVX2(const cv::Mat& I_k, const cv::Mat& I_c, const cv::Mat& du_c, const cv::Mat& dv_c, const chk::Vec2& pt_k,
        chk::Vec2& pt_c_tracked, chk::Vec6& point_params,
        float& err_ssd_, float& err_ncc_, int& mask_)
    {
        // initialize
        err_ssd_ = 0.0f;
        err_ncc_ = 0.0f;
        mask_ = false;

        // priors: pt_c_tracked, point_params(5:6) = pt_c_tracked - pt_k;
        // (in a case of already tracked, point_params(1:4) are also filled out.)
        // point_params(5:6) -> pt_c_tracked

        if ((point_params(4) + pt_k(0) < 1) || (point_params(4) + pt_k(0) > ncols - 1)
            || (point_params(5) + pt_k(1) < 1) || (point_params(5) + pt_k(1) > nrows - 1)) {
            err_ssd_ = 100.0f;
            err_ncc_ = -1.0f;
            return;
        }

        generateRefPointsAVX2(pt_k);
        interpReferenceImageAVX2(I_k);

        int MAX_ITER = 50;
        int thres_huber = 5.0;
        for (int iter = 0; iter < MAX_ITER; iter++)
        {
            // initialize
            JtWJ_avx.setZero();
            mJtWr_avx.setZero();
            err_ssd_ = 0;

            warpPointsAVX2(point_params, pt_k);
            calcResidualAndWeightAVX2(I_c, du_c, dv_c);
            calcHessianAndJacobianAVX2(err_ssd_);

            err_ssd_ = sqrt(err_ssd_ / (float)80);

            // update tracking point.
            chk::Vec6 delta;
            solveGaussNewtonStepAVX2(delta);
            point_params += delta;

            // break condition, point_params(5:6).norm() < 1e-4
            float delta_norm = sqrt(delta(4)*delta(4) + delta(5)*delta(5));
            float area = abs((1 + point_params(0))*(1 + point_params(3)) - point_params(1)*point_params(2));
            cout << "pt:[" << pt_k(0) << "," << pt_k(1) << "],ptc:[" << point_params(4) + pt_k(0) << "," << point_params(5) + pt_k(1)
            <<"],abcd:["<<point_params(0)<<","<< point_params(1)<<","<< point_params(2)<<","<< point_params(3)
            << "],err:" << err_ssd_ <<"area:"<< area << endl;
            if (delta_norm < 1e-3f) break;
        };


        // update point coordinate
        pt_c_tracked(0) = pt_k(0) + point_params(4);
        pt_c_tracked(1) = pt_k(1) + point_params(5);
        mask_ = true;
    };




    /* ================================================================================
    ================================= Implementation ==================================
    =================================================================================== */
    AffineKLTTracker::AffineKLTTracker(Params* params_, const int& nrows_, const int& ncols_, const Eigen::Matrix3d& K_) {
        params = params_;
        nrows = nrows_;
        ncols = ncols_;
        K << K_(0, 0), K_(0, 1), K_(0, 2),
            K_(1, 0), K_(1, 1), K_(1, 2),
            K_(2, 0), K_(2, 1), K_(2, 2);

        fx = K(0, 0);
        fy = K(1, 1);
        cx = K(0, 2);
        cy = K(1, 2);
        fxinv = 1.0f / fx;
        fyinv = 1.0f / fy;

        errs_ssd = (float*)custom_aligned_malloc(sizeof(float)*ncols*nrows);
        errs_ncc = (float*)custom_aligned_malloc(sizeof(float)*ncols*nrows);
        mask = (int*)custom_aligned_malloc(sizeof(int)*ncols*nrows);

        win_sz = 6;
        M = (2 * win_sz + 1)*(2 * win_sz + 1);
        patch.reserve(M);
        for (int u = -win_sz; u < win_sz + 1; u++)
            for (int v = -win_sz; v < win_sz + 1; v++)
                patch.push_back(chk::Point2f((float)u, (float)v));


        // For SSE. Use only 80 pixels.
        upattern    = (float*)custom_aligned_malloc(sizeof(float) * 88);
        vpattern    = (float*)custom_aligned_malloc(sizeof(float) * 88);
        buf_up_ref  = (float*)custom_aligned_malloc(sizeof(float) * 88);
        buf_vp_ref  = (float*)custom_aligned_malloc(sizeof(float) * 88);
        buf_up_warp = (float*)custom_aligned_malloc(sizeof(float) * 88);
        buf_vp_warp = (float*)custom_aligned_malloc(sizeof(float) * 88);

        buf_Ik      = (float*)custom_aligned_malloc(sizeof(float) * 88);
        buf_du_k = (float*)custom_aligned_malloc(sizeof(float) * 88);
        buf_dv_k = (float*)custom_aligned_malloc(sizeof(float) * 88);

        buf_Ic_warp = (float*)custom_aligned_malloc(sizeof(float) * 88);
        buf_du_c_warp = (float*)custom_aligned_malloc(sizeof(float) * 88);
        buf_dv_c_warp = (float*)custom_aligned_malloc(sizeof(float) * 88);

        buf_residual = (float*)custom_aligned_malloc(sizeof(float) * 88);
        buf_weight   = (float*)custom_aligned_malloc(sizeof(float) * 88);

        SSEData = (float*)custom_aligned_malloc(sizeof(float) * 4 * 28);
        AVXData = (float*)custom_aligned_malloc(sizeof(float) * 8 * 28);

        errs_ssd_sse = (float*)custom_aligned_malloc(sizeof(float) * ncols*nrows);

        int cnt = 0;
        for (int i = -win_sz + 1; i < win_sz; i += 2) {
            for (int j = -win_sz + 1; j < win_sz; j += 2) {
                *(upattern + cnt) = j;
                *(vpattern + cnt) = i;
                ++cnt;
                //cout << "[" << j << "," << i << "]" << endl;
            }
        }
        for (int i = -win_sz; i < win_sz + 1; i += 2) {
            for (int j = -win_sz; j < win_sz + 1; j += 2) {
                *(upattern + cnt) = j;
                *(vpattern + cnt) = i;
                ++cnt;
                //cout << "[" << j << "," << i << "]" << endl;
            }
        }
       
        cout << "length of up vp : " << cnt << endl; 
        
    };
    AffineKLTTracker::~AffineKLTTracker() {
        params = nullptr;
        frame_c = nullptr;
        track_ref = nullptr;
        if (errs_ssd != nullptr) custom_aligned_free((void*)errs_ssd);
        if (errs_ncc != nullptr) custom_aligned_free((void*)errs_ncc);
        if (mask != nullptr) custom_aligned_free((void*)mask);
        if (upattern != nullptr) custom_aligned_free((void*)upattern);
        if (vpattern != nullptr) custom_aligned_free((void*)vpattern);

        if (buf_up_ref != nullptr) custom_aligned_free((void*)buf_up_ref);
        if (buf_vp_ref != nullptr) custom_aligned_free((void*)buf_vp_ref);
        if (buf_up_warp != nullptr) custom_aligned_free((void*)buf_up_warp);
        if (buf_vp_warp != nullptr) custom_aligned_free((void*)buf_vp_warp);

        if (buf_Ik != nullptr) custom_aligned_free((void*)buf_Ik);
        if (buf_du_k != nullptr) custom_aligned_free((void*)buf_du_k);
        if (buf_dv_k != nullptr) custom_aligned_free((void*)buf_dv_k);

        if (buf_Ic_warp != nullptr) custom_aligned_free((void*)buf_Ic_warp);
        if (buf_du_c_warp != nullptr) custom_aligned_free((void*)buf_du_c_warp);
        if (buf_dv_c_warp != nullptr) custom_aligned_free((void*)buf_dv_c_warp);

        if (buf_residual != nullptr) custom_aligned_free((void*)buf_residual);
        if (buf_weight != nullptr) custom_aligned_free((void*)buf_weight);

        if (SSEData != nullptr) custom_aligned_free((void*)SSEData);
        if (AVXData != nullptr) custom_aligned_free((void*)AVXData);

        if (errs_ssd_sse != nullptr) custom_aligned_free((void*)errs_ssd_sse);
    };

    void AffineKLTTracker::fillPriors(const Eigen::Matrix4d& T_ck_)
    {
        int Np = track_ref->Np;
        chk::Vec2* p_pts_p = track_ref->pts_p;
        chk::Vec3* p_Xp = track_ref->Xp;
        chk::Vec3* p_Xp_max = track_ref->Xp + Np;
        Eigen::Matrix3f R_ck;
        Eigen::Vector3f t_ck;
        R_ck << T_ck_(0, 0), T_ck_(0, 1), T_ck_(0, 2),
            T_ck_(1, 0), T_ck_(1, 1), T_ck_(1, 2),
            T_ck_(2, 0), T_ck_(2, 1), T_ck_(2, 2);
        t_ck << T_ck_(0, 3), T_ck_(1, 3), T_ck_(2, 3);

        chk::Vec2* p_track = track_ref->pts_p_tracked;
        chk::Vec2* p_warped = track_ref->pts_p_warped;
        chk::Vec6* p_point_params = track_ref->point_params;
        for (; p_Xp < p_Xp_max; p_Xp++, p_track++, p_pts_p++, p_point_params++) {
            if ((*p_Xp)(2) < 0.01f) continue;

            // warp and reproject
            chk::Vec3 Xpwarp = R_ck*(*p_Xp) + t_ck;
            float uw = fx*(Xpwarp(0) / Xpwarp(2)) + cx;
            float vw = fy*(Xpwarp(1) / Xpwarp(2)) + cy;

            (*p_track)(0) = uw; // pt_k
            (*p_track)(1) = vw;
            (*p_warped)(0) = uw;
            (*p_warped)(1) = vw;

            (*p_point_params)(4) = uw - (*p_pts_p)(0); // pt_c_tracked - pt_k.
            (*p_point_params)(5) = vw - (*p_pts_p)(1);
        }
    };

    void AffineKLTTracker::trackForwardAdditiveSingle(
        const cv::Mat& img_k, const cv::Mat& img_c, const cv::Mat& du_c, const cv::Mat& dv_c, const chk::Vec2& pt_k,
        chk::Vec2& pt_c_tracked, chk::Vec6& point_params,
        float& err_ssd_, float& err_ncc_, int& mask_)
    {
        // initialize
        err_ssd_ = 0.0f;
        err_ncc_ = 0.0f;
        mask_ = false;

        // priors: pt_c_tracked, point_params(5:6) = pt_c_tracked - pt_k;
        // (in a case of already tracked, point_params(1:4) are also filled out.)
        // point_params(5:6) -> pt_c_tracked

        if ((point_params(4) + pt_k(0) < 1) || (point_params(4) + pt_k(0) > ncols - 1)
            || (point_params(5) + pt_k(1) < 1) || (point_params(5) + pt_k(1) > nrows - 1)) {
            err_ssd_ = 100.0f;
            err_ncc_ = -1.0f;
            return;
        }

        int MAX_ITER = 50;
        int thres_huber = 5.0;
        for (int iter = 0; iter < MAX_ITER; iter++)
        {
            // initialize
            JtWJ.setZero();
            mJtWr.setZero();

            float a = point_params(0);
            float b = point_params(1);
            float c = point_params(2);
            float d = point_params(3);
            float tu = point_params(4);
            float tv = point_params(5);

            // fill J & r for this pixel point.
            chk::Vec6 J;
            int cnt = 0;
            for (int j = 0; j < M; j++)
            {
                float uj = patch[j].x; // j-th element of patch pixels.
                float vj = patch[j].y; // j-th element of patch pixels.
                float u_warp = (1 + a) * uj + b * vj + tu + pt_k(0);
                float v_warp = c * uj + (1 + d) * vj + tv + pt_k(1);

                // in image test
                if ((u_warp < 1) || (u_warp > ncols - 1) || (v_warp < 1) || (v_warp > nrows - 1)) continue;

                // calculate and push edge residual, Huber weight, Jacobian, and Hessian
                float Ic_warp = improc::interpImageSingle(img_c, u_warp, v_warp);
                float dx_warp = improc::interpImageSingle(du_c, u_warp, v_warp);
                float dy_warp = improc::interpImageSingle(dv_c, u_warp, v_warp);

                float Ik = improc::interpImageSingle(img_k, uj + pt_k(0), vj + pt_k(1));
                float r = Ic_warp - Ik;
                float w = fabs(r) < thres_huber ? 1.0f : thres_huber / fabs(r);


                // fill out Jacobian of this point.
                J(0) = dx_warp*uj;
                J(1) = dx_warp*vj;
                J(2) = dy_warp*uj;
                J(3) = dy_warp*vj;
                J(4) = dx_warp;
                J(5) = dy_warp;

                update(J, r, w, err_ssd_);
                ++cnt;
            }

            err_ssd_ = sqrt(err_ssd_ / (float)cnt);

            // update tracking point.
            chk::Vec6 delta;
            solveGaussNewtonStep(delta);
            point_params += delta;

            // validity test (1)- if the tracked pixel is far from the original projection, reject.
            if (1) 1;

            // break condition, point_params(5:6).norm() < 1e-4
            float delta_norm = sqrt(delta(4)*delta(4) + delta(5)*delta(5));
            float area = abs((1 + point_params(0))*(1 + point_params(3)) - point_params(1)*point_params(2));
            /*cout << "pt:[" << pt_k(0) << "," << pt_k(1) << "],ptc:[" << point_params(4) + pt_k(0) << "," << point_params(5) + pt_k(1)
            <<"],abcd:["<<point_params(0)<<","<< point_params(1)<<","<< point_params(2)<<","<< point_params(3)
            << "],err:" << err_ssd_ <<"area:"<< area << endl;*/
            if (delta_norm < 1e-3f) break;
        };


        // update point coordinate
        pt_c_tracked(0) = pt_k(0) + point_params(4);
        pt_c_tracked(1) = pt_k(1) + point_params(5);
        mask_ = true;
    };

    void AffineKLTTracker::trackInverseCompositionalSingle(
        const cv::Mat& img_k, const cv::Mat& img_c, const cv::Mat& du_k, const cv::Mat& dv_k, const chk::Vec2& pt_k,
        chk::Vec2& pt_c_tracked, chk::Vec6& point_params,
        float& err_ssd_, float& err_ncc_, int& mask_)
    {

    };

    /**
    * Left point patch: Fill out Hessian (approximated), Jacobian, and residual vector.
    **/
    inline void AffineKLTTracker::update(const Vec6& Jt, const float& r, const float& weight, float& err_ssd)
    {
        // cout << "Jt.transpose()*Jt:\n" << Jt*(Jt.transpose()) << endl;
        JtWJ.noalias() += (Jt*Jt.transpose())*weight;
        mJtWr.noalias() -= Jt*(r*weight);
        err_ssd += r*r*weight;
    }

    void AffineKLTTracker::solveGaussNewtonStep(Vec6& delta)
    {
        delta = JtWJ.ldlt().solve(mJtWr);
    };


    void AffineKLTTracker::trackAllRefPoints() {
        int Np = track_ref->Np;
        chk::Vec3* p_Xp = track_ref->Xp;
        chk::Vec3* p_Xp_max = track_ref->Xp + Np;
        chk::Vec2* p_pts_k = track_ref->pts_p; // pts_k
        chk::Vec2* p_pts_k_warp = track_ref->pts_p_warped;
        chk::Vec2* p_track = track_ref->pts_p_tracked; // pts_c_tracked (until now, pts_k_warped)
        chk::Vec6* p_point_params = track_ref->point_params; // 6x1 motion parameters. [1+a,b;c,1+d], [tx,ty]'

        float* p_err_sdd = errs_ssd;
        float* p_err_ncc = errs_ncc;
        int* p_mask = mask;

        cv::Mat& img_k = track_ref->keyframe->left->img[0];
        cv::Mat& img_c = frame_c->left->img[0];
        cv::Mat& du_c = frame_c->left->du[0];
        cv::Mat& dv_c = frame_c->left->dv[0];
        for (; p_Xp < p_Xp_max; p_Xp++, p_track++, p_pts_k++, p_pts_k_warp++, p_point_params++,
            p_err_sdd++, p_err_ncc++, p_mask++)
        {
            if ((*p_Xp)(2) < 0.01f) {
                *p_err_sdd = 1000;
                *p_err_ncc = 1000;
                *p_mask = 0;
                continue; // only for points with depths.
            }
            trackForwardAdditiveSingleSSE(
                img_k, img_c, du_c, dv_c, *p_pts_k, *p_pts_k_warp, *p_track, *p_point_params, *p_err_sdd, *p_err_ncc, *p_mask);
        }
    };
};
#endif