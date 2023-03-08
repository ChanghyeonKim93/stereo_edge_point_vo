#ifndef _STEREOEPVO_H_
#define _STEREOEPVO_H_

#include <iostream>
#include <exception>
#include <vector>
#include <cmath>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "Eigen/Dense"

#include "utils/Lie.h"
#include "utils/timer_windows.h"

#include "image/load_stereo.h"
#include "Params.h"

#include "image/image_proc.h"
#include "depth_reconstruction/depth_reconstruction.h"

#include "frame/stereo_frames.h"

#include "quadtrees/stereo_multiquadtrees_wrapper.h"
#include "quadtrees\CommonStruct.h"

#include "tracking/tracking_reference.h"
#include "tracking/motion_tracker.h"

using namespace std;
class StereoEPVO {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        vector<string>               leftnames;
    vector<string>               rightnames;
    vector<double>               time_cam;

    StereoEPVO(const string& leftDir, const string& rightDir,
        const string& timeDir, const string& calibDir);
    ~StereoEPVO();
    void run(); // 알고리즘 구동부. TODO: thread로 묶기.

private:
    Params*                       params; // 온~갖 파라미터들!
    StereoCameras*                cams; // camera 구조체

    StereoFrame*                  frame_k; // frames (key)
    StereoFrame*                  frame_c; // frames (cur)
    StereoMultiQuadtreesWrapper*  tree; // quadtree를 저장하는 공간
    chk::TrackingReference*       track_ref; // point and informs for tracking. // keyframe일때만 activate.

    chk::SE3AffineBrightTracker*  se3affine_tracker;

    int max_kfs_num;
    vector<StereoFrame*> kfs;// keyframe들을 저장하는 공간

                             // 이미지 번호
    int n_key;
    int n_cur;
    int n_start;
    int n_final;

    // 자세관련.
    Eigen::VectorXd xi_ck;
    Eigen::Vector2d ab_ck;

    vector<Eigen::VectorXd> xi_save;
    vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> ab_save;
    vector<double> Nec_save;
    vector<double> Np_save;
    vector<int> iter_save;

    // 알고리즘 구동상태 flag들
    bool flag_e_exist, flag_e_match, flag_e_gradtest;
    bool flag_p_exist, flag_p_inimage, flag_p_track;
    bool edge_on, point_on, edge_on_reproj, point_on_reproj;

    // visualization 용
    struct PointsKey {

    };
    vector<PointsKey> points_key;

    vector<int> idx_e_recon;
    vector<int> idx_p_recon;
};


/* ================================================================================
================================= Implementation ==================================
=================================================================================== */
void StereoEPVO::run() {
    // 자주 사용되는 저장공간을 선 할당한다.
    cout << "\n\n==================================================\n";
    cout << "START algorithm..." << endl;
    bool flag_vis = true;
    cv::Mat img_l, img_r;

    Eigen::Matrix4d T_ck = Eigen::Matrix4d::Identity();
    chk::Vec8 xi_ck;
    xi_ck.setZero();

    // load first stereo images
    img_l = cv::Mat::zeros(cams->left->n_rows, cams->left->n_cols, CV_8UC1); // 0.2 ms
    img_r = cv::Mat::zeros(cams->left->n_rows, cams->left->n_cols, CV_8UC1); // 0.2 ms
    load_stereo::imreadStereo(leftnames[n_key], rightnames[n_key], img_l, img_r);// 16 ~ 20 ms
    cams->undistortImages(img_l, img_r, img_l, img_r); // 0.25 ms


                                                       // initialize Keyframe
    frame_k->framePyrConstruct(img_l, img_r, n_key, Eigen::Matrix4d::Identity(), true, params); // 7~11 ms
    depthrecon::depthReconStatic(frame_k); // 초기화 안된 상태에서 3 ms / 300 pts
    depthrecon::depthReconStaticPoint(frame_k); // 초기화 안된 상태에서 3 ms / 300 pts
    track_ref->initializeAndRenewReference(frame_k); // 0.005 ms 매우빠름. 동적할당 문제있다.
    se3affine_tracker->linkRef(track_ref); // link track_ref to tracker;


    if (flag_vis) {
        for (int lvl = 0; lvl < 3; lvl++) {
            cv::namedWindow("img");
            cv::Point2f pt_temp(0, 0);

            cv::Mat img_draw;
            frame_k->left->img[lvl].convertTo(img_draw, CV_8UC1);
            cv::cvtColor(img_draw, img_draw, CV_GRAY2RGB);
            cv::Scalar circle_edge(0, 0, 255);
            cv::Scalar circle_pts(255, 0, 0);

            int Nec = track_ref->Nec;
            int Npe = track_ref->Npe[lvl];
            for (int i = 0; i < Nec*Npe; i++) {
                pt_temp.x = (*(track_ref->pts_epat[lvl] + i))[0];
                pt_temp.y = (*(track_ref->pts_epat[lvl] + i))[1];
                cv::circle(img_draw, pt_temp, 0, circle_edge, 1);
            }

            int Np = track_ref->Np;
            int Npp = track_ref->Npp[lvl];
            for (int i = 0; i < Np*Npp; i++) {
                pt_temp.x = (*(track_ref->pts_ppat[lvl] + i))[0];
                pt_temp.y = (*(track_ref->pts_ppat[lvl] + i))[1];
                cv::circle(img_draw, pt_temp, 0, circle_pts, 1);
            }
            cout << " # of total patch pixels  - edge: " << Nec*Npe << ", point: " << Np*Npp << endl;
            cv::imshow("img", img_draw);
            cv::waitKey(0);
            cv::destroyWindow("img");
        }

    }
    // iterative part starts.
    for (n_cur = n_start + 1; n_cur < n_final; n_cur++) {
        cout << "===========================================\n";
        cout << "[n_cur: " << n_cur << "]\n";
        load_stereo::imreadStereo(leftnames[n_cur], rightnames[n_cur], img_l, img_r); // 이미지를 읽어온다. 16 ~ 20 ms
        cams->undistortImages(img_l, img_r, img_l, img_r); // 0.5 ms 이내.
        frame_c->framePyrConstruct(img_l, img_r, n_cur, Eigen::Matrix4d::Identity(), true, params); // 방컴: 8~9 ms, 학교컴: 12~14 ms
                                                                                                    // findSalient가 가장 많이 소모. (대략 3~5 ms)

        se3affine_tracker->linkFrameCur(frame_c); // link frame_c to tracker;

        if (flag_vis) {
            cv::Point2f pt_temp(0, 0);

            cv::Mat img_draw;
            cv::cvtColor(frame_c->left->edge[0], img_draw, CV_GRAY2RGB);
            cv::Scalar circle_centers(0, 255, 0);
            cv::Scalar circle_pts(0, 0, 255);
            cout << "# center: " << frame_c->left->pts_centers.size() << endl;
            cout << "# points: " << frame_c->left->ft_point[0]->pts.size() << endl;
            for (int i = 0; i < frame_c->left->pts_centers.size(); i++) {
                chk::chk2cv(frame_c->left->pts_centers[i], pt_temp);
                cv::circle(img_draw, pt_temp, 2, circle_centers, 2);
            }
            for (int i = 0; i < frame_c->left->ft_point[0]->pts.size(); i++) {
                chk::chk2cv(frame_c->left->ft_point[0]->pts[i], pt_temp);
                cv::circle(img_draw, pt_temp, 2, circle_pts, 2);
            }
            cv::imshow("img", img_draw);
            cv::waitKey(0);
        }

        Eigen::Matrix4d T_ck_temp = T_ck;

        // lvl 3까지 point & edge joint tracking 하고, 그 이상부터는 residual minimization
        // Coarse-to-fine update!
        for (int lvl = params->pyr.max_lvl - 1; lvl > -1; --lvl) {

            int n_cols = frame_k->left->img[lvl].cols;
            int n_rows = frame_k->left->img[lvl].rows;

            // 현재이미지에서 quatrees를 생성한다. (reference edge가 있는지없는지도 확인해야한다).
            tree->multiQuadtreeBuild( // At lvl 2, under 1.5 ms is consumed. (dorm com: [5]0.3 [4]0.5 [3]0.7 ms)
                frame_c->left->ft_edge[lvl]->pts_edge, frame_c->left->ft_edge[lvl]->bins_edge,
                frame_c->right->ft_edge[lvl]->pts_edge, frame_c->right->ft_edge[lvl]->bins_edge,
                n_rows, n_cols,
                params->qt.max_depth, params->qt.eps, params->qt.thres_dist);

            se3affine_tracker->linkTree(tree); // link tree to tracker;

                                               // compute the residual and adjust the huber threshold
            Eigen::Matrix<float, 6, 1> xi_temp = xi_ck.block<6, 1>(0, 0);
            Lie::se3Exp(xi_temp, T_ck);

            float huber_scaler_edge = 1.0f;
            float huber_scaler_point = 1.0f;
            se3affine_tracker->calcResidual(T_ck, lvl, huber_scaler_edge, huber_scaler_point);
            while ((se3affine_tracker->n_overthres_point / se3affine_tracker->n_valid_pointpatch) > 0.6 && huber_scaler_point < 5) {
                float saturated_ratio = (se3affine_tracker->n_overthres_point / se3affine_tracker->n_valid_pointpatch);
                cout << "saturated ratio: " << saturated_ratio << endl;
                huber_scaler_point *= 1.5f;
                se3affine_tracker->calcResidual(T_ck, lvl, huber_scaler_edge, huber_scaler_point);
            }
            cout << " huber thres (point): " << huber_scaler_point << endl;



            // iteratively motion update. 
            double dt_lvl = 0;
            bool is_cached = false;
            int iter = 0;
            float lambda_lowlimit = 0.001; // for L-M method. (damping coefficient)
            float lambda_maxlimit = 1000; // for L-M method. (damping coefficient)

            float lambda = 0.01; // initiali damping ratio
            float res_old = 1e9f, res_new = 0.0f;
            // Levenberg-Marquardt optimization.
            for (; iter < params->iter.MAX_ITER; iter++) {
                chk::Vec8 delta_step;
                delta_step.setZero();

                tic();
                Eigen::Matrix<float, 6, 1> xi_temp = xi_ck.block<6, 1>(0, 0);
                Lie::se3Exp(xi_ck.block<6, 1>(0, 0), T_ck);
                se3affine_tracker->fillHJr(T_ck, lvl, is_cached); // render A and b
                se3affine_tracker->solveLMstep(delta_step, lambda);     // solve A*x = b

                xi_ck.noalias() += delta_step;
                double dt = toc(0);

                // calculate new residual after this update step
                // Vec6 resNew = calcRes(lvl, refToNew_new, aff_g2l_new, setting_coarseCutoffTH * levelCutoffRepeat);

                // decide whether to accept this step
                res_new = se3affine_tracker->err;
                bool is_decresing = res_new < res_old;
                if (is_decresing) {
                    // decrease lambda
                    // TODO: 사실 decreasing일때만 update 해야한다.
                    res_old = res_new;
                    lambda *= 0.5;
                    if (lambda < lambda_lowlimit) lambda = lambda_lowlimit; // lower bound clipping
                }
                else {
                    // increase lambda in LM (bad convergence)
                    lambda *= 4;
                    if (lambda > lambda_maxlimit) lambda = lambda_maxlimit; // upper bound clipping
                }

                dt_lvl += dt;
                cout << "lv" << lvl << ",itr:" << iter << ",lam:" << lambda << " // err: " << se3affine_tracker->err
                    << ", err_e: " << se3affine_tracker->err_e << ", err_p: " << se3affine_tracker->err_ppat
                    << " //  t_ck:[" << T_ck(0, 3) << "," << T_ck(1, 3) << "," << T_ck(2, 3) << "]"
                    << ", dt: " << dt << " [ms], norm: " << delta_step.block<6, 1>(0, 0).norm() << "\n";
                if (!is_cached) is_cached = true;

                //  break condition? (when incremental step is sufficiently small.)
                if (delta_step.block<6, 1>(0, 0).norm() < 1e-4) break;
            }
            cout << " --- lvl statistics: iter: " << iter << " / dt_lvl : " << dt_lvl << " [ms]\n";


            if (flag_vis) {
                Eigen::Matrix3d K = frame_k->K_pyr[lvl];
                Eigen::Matrix3d Kinv = frame_k->Kinv_pyr[lvl]; // 현재 level의 intrinsic matrix
                float fx = K(0, 0);
                float fy = K(1, 1);
                float cx = K(0, 2);
                float cy = K(1, 2);
                cv::namedWindow("match");
                cv::Mat img_draw;
                frame_c->left->img[lvl].convertTo(img_draw, CV_8UC1);
                cv::cvtColor(img_draw, img_draw, CV_GRAY2RGB);
                cv::Scalar circle_green(0, 255, 0);
                cv::Scalar circle_red(0, 0, 255);
                cv::Scalar circle_blue(255, 0, 0);

                chk::Mat44 T_ck_temp;
                T_ck_temp << T_ck(0, 0), T_ck(0, 1), T_ck(0, 2), T_ck(0, 3),
                    T_ck(1, 0), T_ck(1, 1), T_ck(1, 2), T_ck(1, 3),
                    T_ck(2, 0), T_ck(2, 1), T_ck(2, 2), T_ck(2, 3),
                    T_ck(3, 0), T_ck(3, 1), T_ck(3, 2), T_ck(3, 3);
                chk::Mat33 R_ck = T_ck_temp.block<3, 3>(0, 0);
                chk::Vec3 t_ck = T_ck_temp.block<3, 1>(0, 3);

                chk::Vec2* p_pts_e = track_ref->pts_e;
                chk::Vec3* p_Xe = track_ref->Xe;
                int* idx_l = track_ref->idx_l;
                chk::Vec3* p_Xemax = track_ref->Xe + track_ref->Nec;

                for (; p_Xe< p_Xemax; p_Xe++, p_pts_e++, idx_l++) {
                    if ((*p_Xe)[2] > 0.001f) {
                        chk::Vec3 X_warp_l = R_ck*(*p_Xe) + t_ck; // warped point onto left image.
                        float iz = 1.0f / X_warp_l(2);
                        float u_warp_l = fx*X_warp_l(0) * iz + cx;
                        float v_warp_l = fy*X_warp_l(1) * iz + cy;

                        chk::Point2f pt_matched = frame_c->left->ft_edge[lvl]->pts_edge[*idx_l];

                        cv::line(img_draw, cv::Point2f(u_warp_l, v_warp_l), cv::Point2f(pt_matched.x, pt_matched.y), cv::Scalar(0, 0, 0), 1);
                        cv::circle(img_draw, cv::Point2f(u_warp_l, v_warp_l), 1, circle_green, 1);
                        cv::circle(img_draw, cv::Point2f(pt_matched.x, pt_matched.y), 3, circle_red, 1);
                    }
                }


                chk::Vec3* p_Xppat = track_ref->Xppat[lvl];
                chk::Vec3* p_Xppatmax = track_ref->Xppat[lvl] + track_ref->Np*track_ref->Npp[lvl];
                for (; p_Xppat < p_Xppatmax; p_Xppat++) {
                    if ((*p_Xppat)(2) > 0.001f) {
                        chk::Vec3 Xp_warp = R_ck*(*p_Xppat) + t_ck; // warped point onto left image.
                        float iz = 1.0f / Xp_warp(2);
                        float u_warp = fx*Xp_warp(0) * iz + cx;
                        float v_warp = fy*Xp_warp(1) * iz + cy;

                        cv::circle(img_draw, cv::Point2f(u_warp, v_warp), 1, circle_blue, 1);
                    }
                }
                cv::imshow("match", img_draw);
                cv::waitKey(0);
                cv::destroyWindow("match");
            }

            // delete tree on current level.
            se3affine_tracker->detachTree();
            tree->multiQuadtreeDelete();
            cout << "\n";
        } // end iterative optimization
          // Track point (affine KLT)

          // Reprojection error minimization

          // Update keyframe depth 

          // keyframe change?


        se3affine_tracker->detachFrameCur();
    };
};

StereoEPVO::StereoEPVO(const string& leftDir, const string& rightDir,
    const string& timeDir, const string& calibDir)
    : n_start(0), n_final(0)
{
    // 생성자에서는 기본적인 데이터 구조체들만을 만들거나 일부만 채워넣는다.
    // 이미지 불러오기.
    cams = new StereoCameras();
    load_stereo::loadStereoInforms(calibDir, cams);
    load_stereo::loadStereoImages(leftDir, rightDir, timeDir,
        leftnames, rightnames, time_cam);
    cout << "# of images: " << leftnames.size() << endl;

    n_start = 110;
    n_final = leftnames.size();
    n_final = 130;

    n_key = n_start;
    n_cur = -1;

    cout << "img. start num: " << n_start << " / final num: " << n_final << endl;

    // 파라미터 불러오기 
    int n_rows = cams->left->n_rows;
    int n_cols = cams->left->n_cols;
    params = new Params(n_cols, n_rows);

    // 저장공간.
    xi_save.reserve(n_final);
    ab_save.reserve(n_final);
    Nec_save.reserve(n_final);
    Np_save.reserve(n_final);
    iter_save.reserve(n_final);
    points_key.reserve(0); // visualization을 위한 keyframe 정보들이 저장되는 공간.

    max_kfs_num = 10; // default = 10;

                      // keyframe과 current frame class를 만든다. 다 비어있는 상태임.
    frame_k = new StereoFrame(-1, Eigen::Matrix4d::Identity(), params->pyr.max_lvl, cams, params); // keyframe
    frame_c = new StereoFrame(-1, Eigen::Matrix4d::Identity(), params->pyr.max_lvl, cams, params); // current frame

                                                                                                   // recon indicator 초기화
    idx_e_recon.reserve(2000);
    idx_p_recon.reserve(2000);

    // reference points and information 관련
    track_ref = new chk::TrackingReference(-1, params->pyr.max_lvl, frame_k->left->n_rows_pyr, frame_k->left->n_cols_pyr); // tracking에 쓰이는 모든 reference 정보가 들어있다.
    se3affine_tracker = new chk::SE3AffineBrightTracker(params, frame_k->left->n_rows_pyr, frame_k->left->n_cols_pyr, frame_k->K_pyr, cams->T_nlnr);
    // quadtrees wrapper
    tree = new StereoMultiQuadtreesWrapper(); // it uses idx_l idx_r node_l node_r in reference.
};

StereoEPVO::~StereoEPVO() {
    cout << "\n\n==================================================\n";
    cout << "SEPVO is deleted." << endl;
    cout << " L frame_key: ";
    delete frame_k;
    cout << " L frame_cur: ";
    delete frame_c;
    delete cams;
    delete tree;
    delete track_ref;
    delete se3affine_tracker;
};
#endif