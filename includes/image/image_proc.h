#ifndef _IMAGE_PROC_H_
#define _IMAGE_PROC_H_
#include <iostream>
#include <exception>
#include <vector>
#include <cmath>
#include "opencv2/opencv.hpp"  
#include "Eigen/Dense"

#include "immintrin.h"
#include "xmmintrin.h"

#include "../utils/timer_windows.h"
#include "../quadtrees/CommonStruct.h"

using namespace std;

namespace improc {

	void pyrDownNormal(const cv::Mat& source, cv::Mat& dest) {
		const float* source_ptr = source.ptr<float>(0);
		float* dest_ptr = dest.ptr<float>(0);
		int width = source.cols;
		int height = source.rows;
		// normal
		const float* s;
		int wh = width*height;
		for (int v = 0; v < wh; v += 2 * width) {
			for (int u = 0; u < width; u += 2) {
				s = source_ptr + u + v;
				*dest_ptr = (s[0] + s[1] + s[width] + s[width + 1])*0.25f;
				++dest_ptr;
			}
		}
	};

	void pyrDownSSE(const cv::Mat& source, cv::Mat& dest) {
		const float* source_ptr = source.ptr<float>(0);
		float* dest_ptr = dest.ptr<float>(0);
		int width = source.cols;
		int height = source.rows;

		// SSE
		if (width % 8 == 0) {
			__m128 p025 = _mm_setr_ps(0.25f, 0.25f, 0.25f, 0.25f);
			const float* maxY = source_ptr + width*height; // 맨 끝 포인터
			for (const float* y = source_ptr; y < maxY; y += width * 2) { // 두줄씩 내려감.
				const float* maxX = y + width; // 해당 줄 맨 끝 포인터
				// _mm_prefetch((char*)maxX + 640, _MM_HINT_T0); // 별 효과가 없다.
				for (const float* x = y; x < maxX; x += 8) {
					__m128 top_left = _mm_load_ps((float*)x);
					__m128 bot_left = _mm_load_ps((float*)x + width);
					__m128 left = _mm_add_ps(top_left, bot_left);
					__m128 top_right = _mm_load_ps((float*)x + 4);
					__m128 bot_right = _mm_load_ps((float*)x + width + 4);
					__m128 right = _mm_add_ps(top_right, bot_right);

					__m128 sumA = _mm_shuffle_ps(left, right, _MM_SHUFFLE(2, 0, 2, 0));
					__m128 sumB = _mm_shuffle_ps(left, right, _MM_SHUFFLE(3, 1, 3, 1));

					__m128 sum = _mm_add_ps(sumA, sumB);
					sum = _mm_mul_ps(sum, p025);

					_mm_store_ps(dest_ptr, sum);
					dest_ptr += 4;
				}
			}
		}
		else {
			pyrDownNormal(source, dest);
		}
	};
	void pyrDownAVX(const cv::Mat& source, cv::Mat& dest) {
		const float* source_ptr = source.ptr<float>(0);
		float* dest_ptr = dest.ptr<float>(0);
		int width = source.cols;
		int height = source.rows;

		// AVX
		if (width % 16 == 0) {
			__m256 p025 = _mm256_setr_ps(0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f);
			const float* maxY = source_ptr + width*height; // 맨 끝 포인터
			for (const float* y = source_ptr; y < maxY; y += width * 2) { // 두줄씩 내려감.
				const float* maxX = y + width; // 해당 줄 맨 끝 포인터
				for (const float* x = y; x < maxX; x += 16) {
					__m256 top_left = _mm256_load_ps((float*)x);
					__m256 bot_left = _mm256_load_ps((float*)x + width);
					__m256 left = _mm256_add_ps(top_left, bot_left);
					__m256 top_right = _mm256_load_ps((float*)x + 8);
					__m256 bot_right = _mm256_load_ps((float*)x + width + 8);
					__m256 right = _mm256_add_ps(top_right, bot_right);

					__m256 sumA = _mm256_shuffle_ps(left, right, _MM_SHUFFLE(2, 0, 2, 0));
					__m256 sumB = _mm256_shuffle_ps(left, right, _MM_SHUFFLE(3, 1, 3, 1));

					__m256 sum = _mm256_add_ps(sumA, sumB);
					sum = _mm256_mul_ps(sum, p025);

					_mm256_store_ps(dest_ptr, sum);
					dest_ptr += 8;
				}
			}
		}
	};

	void imagePyramid(const cv::Mat& img, vector<cv::Mat>& img_pyr) {
		// img needs to be 'CV_32FC1'.
		if (img_pyr.size() > 0) { // 뭔가 있다.
			size_t max_lvl = img_pyr.size();
			img.copyTo(img_pyr[0]);
			for (size_t lvl = 1; lvl < max_lvl; lvl++) {
				pyrDownSSE(img_pyr[lvl - 1], img_pyr[lvl]);
			}
		}
		else { // 아무것도 없다?
			throw std::runtime_error("In 'imagePyramid', img_pyr needs to be initialized!\n");
		}
	};

	void interpImage(const cv::Mat& img, const vector<chk::Point2f>& pts, vector<float>& brightness, vector<int>& valid_vec) {
		brightness.resize(0);
		valid_vec.resize(0);

		const float* img_ptr = img.ptr<float>(0);
		size_t n_pts = pts.size();
		int n_cols = img.cols;
		int n_rows = img.rows;

		int is_valid = -1; // valid pixel ?

		float u_cur, v_cur; // float-precision coordinates
		int u_0, v_0; // truncated coordinates
		int v_0n_cols, v_0n_colsu_0;

		float I1, I2, I3, I4;
		float ax, ay, axay;

		// I1 ax / 1-ax I2 
		// ay   (v,u)
		//  
		// 1-ay
		// I3           I4

		for (size_t i = 0; i < n_pts; i++) {
			u_cur = pts[i].x;
			v_cur = pts[i].y;
			u_0 = (int)floor(u_cur);
			v_0 = (int)floor(v_cur);
			is_valid = 1;
			brightness.push_back(0);
			valid_vec.push_back(0);

			if (u_cur >= 0 && u_cur < n_cols - 1)
				ax = u_cur - (float)u_0;
			else if (u_cur == n_cols - 1) {
				u_0 = (n_cols - 1) - 1;
				ax = 0;
			}
			else if (u_cur > -1 && u_cur < 0) {
				u_0 = 1;
				ax = 1;
			}
			else is_valid = 0;
			if (v_cur >= 0 && v_cur < n_rows - 1)
				ay = v_cur - (float)v_0;
			else if (v_cur == n_rows - 1) {
				v_0 = (n_rows - 1) - 1;
				ay = 0;
			}
			else if (v_cur > -1 && v_cur < 0) {
				v_0 = 1;
				ay = 1;
			}
			else is_valid = 0;

			axay = ax*ay;
			if (is_valid) {
				v_0n_cols = v_0*n_cols;
				v_0n_colsu_0 = v_0n_cols + u_0;
				I1 = img_ptr[v_0n_colsu_0];
				I2 = img_ptr[v_0n_colsu_0 + 1];
				I3 = img_ptr[v_0n_colsu_0 + n_cols];
				I4 = img_ptr[v_0n_colsu_0 + n_cols + 1];

				brightness[i] += axay*(I1 - I2 - I3 + I4);
				brightness[i] += ax*(-I1 + I2);
				brightness[i] += ay*(-I1 + I3);
				brightness[i] += I1;
				valid_vec[i] = 1;
			}
			else {
				brightness[i] = -2;
				valid_vec[i] = 0;
			}
		}
	};

	void interpImage(const cv::Mat& img, const vector<chk::Point2f>& pts, chk::Point2f& pt_offset, float* brightness, int* valid_vec) {
		
		const float* img_ptr = img.ptr<float>(0);
		size_t n_pts = pts.size();
		int n_cols = img.cols;
		int n_rows = img.rows;

		int is_valid = -1; // valid pixel ?

		float u_cur, v_cur; // float-precision coordinates
		int u_0, v_0; // truncated coordinates
		int v_0n_cols, v_0n_colsu_0;

		float I1, I2, I3, I4;
		float ax, ay, axay;

		// I1 ax / 1-ax I2 
		// ay   (v,u)
		//  
		// 1-ay
		// I3           I4

		for (size_t i = 0; i < n_pts; i++) {
			u_cur = pts[i].x + pt_offset.x;
			v_cur = pts[i].y + pt_offset.y;
			u_0 = (int)floor(u_cur);
			v_0 = (int)floor(v_cur);
			is_valid = 1;
			brightness[i] = 0;
			valid_vec[i] = 0;

			if (u_cur >= 0 && u_cur < n_cols - 1)
				ax = u_cur - (float)u_0;
			else if (u_cur == n_cols - 1) {
				u_0 = (n_cols - 1) - 1;
				ax = 0;
			}
			else if (u_cur > -1 && u_cur < 0) {
				u_0 = 1;
				ax = 1;
			}
			else is_valid = 0;
			if (v_cur >= 0 && v_cur < n_rows - 1)
				ay = v_cur - (float)v_0;
			else if (v_cur == n_rows - 1) {
				v_0 = (n_rows - 1) - 1;
				ay = 0;
			}
			else if (v_cur > -1 && v_cur < 0) {
				v_0 = 1;
				ay = 1;
			}
			else is_valid = 0;

			axay = ax*ay;
			if (is_valid) {
				v_0n_cols = v_0*n_cols;
				v_0n_colsu_0 = v_0n_cols + u_0;
				I1 = img_ptr[v_0n_colsu_0];
				I2 = img_ptr[v_0n_colsu_0 + 1];
				I3 = img_ptr[v_0n_colsu_0 + n_cols];
				I4 = img_ptr[v_0n_colsu_0 + n_cols + 1];

				brightness[i] += axay*(I1 - I2 - I3 + I4);
				brightness[i] += ax*(-I1 + I2);
				brightness[i] += ay*(-I1 + I3);
				brightness[i] += I1;
				valid_vec[i] = 1;
			}
			else {
				brightness[i] = -2;
				valid_vec[i] = 0;
			}
		}
	};

    float interpImageSingle(const cv::Mat& img, const float& u, const float& v) {

        float* img_ptr = (float*)img.ptr<float>(0);
        int n_cols = img.cols;
        int n_rows = img.rows;

        // I1 ax / 1-ax I2 
        // ay   (v,u)
        //  
        // 1-ay
        // I3           I4
        float ax, ay;
        int u0 = (int)u; // truncated coordinates
        int v0 = (int)v;
        // cout << "u0 v0: " << u0 << ", " << v0 << ", ncols, nrows: " << n_cols << ", " << n_rows << endl;
        if ((u0 > 0) && (u0 < n_cols - 1))
            ax = u - (float)u0;
        else return -1;
        if ((v0 >= 0) && (v0 < n_rows - 1))
            ay = v - (float)v0;
        else return -1;

        float axay = ax*ay;
        int v0cols = v0*n_cols;
        int v0colsu0 = v0cols + u0;

        float I00, I01, I10, I11;
        I00 = img_ptr[v0colsu0];
        I01 = img_ptr[v0colsu0 + 1];
        I10 = img_ptr[v0colsu0 + n_cols];
        I11 = img_ptr[v0colsu0 + n_cols + 1];

        float res = ax*(I01 - I00) + ay*(I10 - I00) + axay*(-I01 + I00 + I11 - I10) + I00;
        
        return res;
    };

    void interpImageSingle3(const cv::Mat& img, const cv::Mat& du, const cv::Mat& dv, const float& u, const float& v, chk::Point3f& interp_) 
    {
        float* img_ptr = (float*)img.ptr<float>(0);
        float* du_ptr = (float*)du.ptr<float>(0);
        float* dv_ptr = (float*)dv.ptr<float>(0);

        int n_cols = img.cols;
        int n_rows = img.rows;

        // I1 ax / 1-ax I2 
        // ay   (v,u)
        //  
        // 1-ay
        // I3           I4
        float ax, ay;
        int u0 = (int)u; // truncated coordinates
        int v0 = (int)v;
        // cout << "u0 v0: " << u0 << ", " << v0 << ", ncols, nrows: " << n_cols << ", " << n_rows << endl;
        if ((u0 > 0) && (u0 < n_cols - 1))
            ax = u - (float)u0;
        else {
            interp_.x = -1;
            interp_.y = -1;
            interp_.z = -1;
            return;
        }
        if ((v0 >= 0) && (v0 < n_rows - 1))
            ay = v - (float)v0;
        else {
            interp_.x = -1;
            interp_.y = -1;
            interp_.z = -1;
            return;
        }
        float axay = ax*ay;
        int v0cols = v0*n_cols;
        int v0colsu0 = v0cols + u0;

        float I00, I01, I10, I11;
        float du00, du01, du10, du11;
        float dv00, dv01, dv10, dv11;

        I00 = img_ptr[v0colsu0];
        I01 = img_ptr[v0colsu0 + 1];
        I10 = img_ptr[v0colsu0 + n_cols];
        I11 = img_ptr[v0colsu0 + n_cols + 1];

        du00 = du_ptr[v0colsu0];
        du01 = du_ptr[v0colsu0 + 1];
        du10 = du_ptr[v0colsu0 + n_cols];
        du11 = du_ptr[v0colsu0 + n_cols + 1];

        dv00 = dv_ptr[v0colsu0];
        dv01 = dv_ptr[v0colsu0 + 1];
        dv10 = dv_ptr[v0colsu0 + n_cols];
        dv11 = dv_ptr[v0colsu0 + n_cols + 1];

        float res_img = ax*(I01 - I00) + ay*(I10 - I00) + axay*(-I01 + I00 + I11 - I10) + I00;
        float res_du  = ax*(du01 - du00) + ay*(du10 - du00) + axay*(-du01 + du00 + du11 - du10) + du00;
        float res_dv  = ax*(dv01 - dv00) + ay*(dv10 - dv00) + axay*(-dv01 + dv00 + dv11 - dv10) + dv00;

        interp_.x = res_img;
        interp_.y = res_du;
        interp_.z = res_dv;
    };

	void sampleImage(const cv::Mat& img, const vector<chk::Point2f>& pts, chk::Point2f& pt_offset, float* brightness,int* valid_vec) {
	
		const float* img_ptr = img.ptr<float>(0);
		size_t n_pts = pts.size();

		int is_valid = -1; // valid pixel ?

		int u_cur, v_cur; // 
		for (size_t i = 0; i < n_pts; i++) {
			u_cur = (int)(pts[i].x + pt_offset.x);
			v_cur = (int)(pts[i].y + pt_offset.y);
			
			is_valid = 1;
			if ((u_cur > 0) && (u_cur < img.cols) && ( v_cur > 0) && (v_cur < img.rows)) {
				brightness[i] = *(img_ptr+v_cur*img.cols +u_cur);
				valid_vec[i] = 1;
			}
			else {
				brightness[i] = -2;
				valid_vec[i] = 0;
			}
		}
	};

    void diffImage(const cv::Mat& img, cv::Mat& dimg, bool flag_dx, bool flag_dy) {// central difference.
        const float* img_ptr = img.ptr<float>(0);
        float* dimg_ptr = dimg.ptr<float>(0);

        // p + n_rows*n_cols - 1; (맨 끝 픽셀)
        // p + v*n_cols;          (v 행 맨 첫 픽셀)
        // p + (v+1)*n_cols - 1;  (v 행 맨 끝 픽셀)
        if ((flag_dx == true) & (flag_dy == false)) { // dx
            int n_cols = img.cols;
            int n_rows = img.rows;
            
            for (int v = 0; v < n_rows; v++) {
                float* p_dimg = dimg_ptr + v*n_cols;
                float* p_dimg_rowendm1 = dimg_ptr + (v + 1)*n_cols - 1;
                const float* p_img_next = img_ptr + v*n_cols + 2;
                const float* p_img_prev = img_ptr + v*n_cols;

                // first and last pixel in this row.
                *p_dimg_rowendm1 = 0;
                *p_dimg = 0;
                ++p_dimg;
                for (; p_dimg < p_dimg_rowendm1; p_dimg++, p_img_next++, p_img_prev++)
                {
                    *p_dimg = 0.5f*(*p_img_next - *p_img_prev);
                }
            }
        }
        else if ((flag_dx == false) & (flag_dy == true)) { // dy
            int n_cols = img.cols;
            int n_rows = img.rows;

            float* p_dimg     = dimg_ptr;
            float* p_dimg_end = dimg_ptr + n_cols*(n_rows - 1);
            const float* p_img_next = img_ptr + 2*n_cols;
            const float* p_img_prev = img_ptr;
            for (; p_dimg < dimg_ptr + n_cols; p_dimg++) {
                *p_dimg = 0;
            }
            for (; p_dimg < p_dimg_end;) 
            {
                float* p_dimg_rowmax = p_dimg + n_cols;
                for (; p_dimg < p_dimg_rowmax; p_dimg++, p_img_next++, p_img_prev++) 
                {
                    *p_dimg = 0.5f*(*p_img_next - *p_img_prev);
                }
            }
            for (; p_dimg < dimg_ptr + n_rows*n_cols; p_dimg++) {
                *p_dimg = 0;
            }
        }
        else { // error.
            throw std::runtime_error("Neither flag_dx nor flag_dy is true!\n");
        }
    };


    
	float calcZNCC(float* a, float* b, int len) {
		float mean_l = 0.0f;
		float mean_r = 0.0f;
		float numer = 0.0f;
		float denom_l = 0.0f;
		float denom_r = 0.0f;

		// calculate means
		for (int i = 0; i < len; i++) {
			mean_l += a[i];
			mean_r += b[i];
		}
		float invn_elem = 1.0f / ((float)len + 0.0001f);

		mean_l *= invn_elem;
		mean_r *= invn_elem;

		//calculate costs
		float l_minus_mean;
		float r_minus_mean;
		for (int i = 0; i < len; i++) {
			l_minus_mean = a[i] - mean_l;
			r_minus_mean = b[i] - mean_r;
			numer += l_minus_mean*r_minus_mean;
			denom_l += l_minus_mean*l_minus_mean;
			denom_r += r_minus_mean*r_minus_mean;
		}
		return numer / sqrt(denom_l*denom_r + 0.001);
	};

	void calcNCCstrip(const cv::Mat& img_l, const cv::Mat& img_r, const int& ul, const int& vl,
		const int& u_min, const int& u_max, const int& win_sz, const int& fat_sz,
		vector<float>& scores) 
	{
		int len = (2 * win_sz + 1)*(2 * fat_sz + 1);
		int n_cols = img_l.cols;
		int n_rows = img_l.rows;

		scores.resize(0);
		const float* ptr_l = img_l.ptr<float>(0);
		const float* ptr_r = img_r.ptr<float>(0);
		float invN = 1.0f / ((float)len);
		for (int u_r = u_min; u_r < u_max + 1; u_r++) {
			float mean_l = 0.0f;
			float mean_r = 0.0f;
			float numer = 0.0f;
			float denom_l = 0.0f;
			float denom_r = 0.0f;
			for (int f = -fat_sz; f < fat_sz + 1; f++) {
				for (int w = -win_sz; w < win_sz + 1; w++) {
					mean_l += *(ptr_l + (vl + f)*n_cols + ul + w);
					mean_r += *(ptr_r + (vl + f)*n_cols + u_r + w);
				}
			}
			mean_l *= invN;
			mean_r *= invN;
			//calculate costs
			float l_minus_mean;
			float r_minus_mean;
			for (int f = -fat_sz; f < fat_sz + 1; f++) {
				for (int w = -win_sz; w < win_sz + 1; w++) {
					l_minus_mean = *(ptr_l + (vl + f)*n_cols + ul + w) - mean_l;
					r_minus_mean = *(ptr_r + (vl + f)*n_cols + u_r + w) - mean_r;
					numer += l_minus_mean*r_minus_mean;
					denom_l += l_minus_mean*l_minus_mean;
					denom_r += r_minus_mean*r_minus_mean;
				}
			}
			scores.push_back(numer / sqrt(denom_l*denom_r + 0.001f));
		}
	};

	void calcNCCstrip_pre(const cv::Mat& img_l, const cv::Mat& img_r, const int& ul, const int& vl,
		const int& u_min, const int& u_max, const int& win_sz, const int& fat_sz,
		vector<float>& scores) 
	{
		int win_len = win_sz * 2 + 1;
		int fat_len = fat_sz * 2 + 1;
		float* l_minus_means = new float[win_len*fat_len];
		int len = win_len * fat_len;
		int n_cols = img_l.cols;
		int n_rows = img_l.rows;

		scores.resize(0);
		float* ptr_l = (float*)img_l.ptr<float>(0);
		float* ptr_r = (float*)img_r.ptr<float>(0);
		float* ptr_l_temp = nullptr;
		float* ptr_r_temp = nullptr;
		float invN = 1.0f / ((float)len);

		float mean_l = 0.0f;
		for (int f = -fat_sz; f < fat_sz + 1; f++) {
			int vlfncolsuwin = (vl + f)*n_cols + ul - win_sz;
			ptr_l_temp = ptr_l + vlfncolsuwin;
			for (int w = -win_sz; w < win_sz + 1; w++)
				mean_l += *(ptr_l_temp++);
		}
		mean_l *= invN;
		float denom_l = 0.0f;
		int cnt = 0;
		for (int f = -fat_sz; f < fat_sz + 1; f++) {
			int vlfncolsuwin = (vl + f)*n_cols + ul - win_sz;
			ptr_l_temp = ptr_l + vlfncolsuwin;
			for (int w = -win_sz; w < win_sz + 1; w++) {
				l_minus_means[cnt] = *(ptr_l_temp++) - mean_l;
				denom_l += l_minus_means[cnt] * l_minus_means[cnt];
				++cnt;
			}
		}

		for (int u_r = u_min; u_r < u_max + 1; u_r++) {
			float mean_r = 0.0f;
			float numer = 0.0f;
			float denom_r = 0.0f;
			for (int f = -fat_sz; f < fat_sz + 1; f++) 
				for (int w = -win_sz; w < win_sz + 1; w++) 
					mean_r += *(ptr_r + (vl + f)*n_cols + u_r + w);
		
			mean_r *= invN;
			//calculate costs
			cnt = 0;
			float r_minus_mean;
			for (int f = -fat_sz; f < fat_sz + 1; f++) {
				int vlfncolsuwin = (vl + f)*n_cols + u_r - win_sz;
				ptr_r_temp = ptr_r + vlfncolsuwin;
				for (int w = -win_sz; w < win_sz + 1; w++) {
					r_minus_mean = *(ptr_r_temp++) - mean_r;
					numer += l_minus_means[cnt] * r_minus_mean;
					denom_r += r_minus_mean*r_minus_mean;
					++cnt;
				}
			}
			scores.push_back(numer / sqrt(denom_l*denom_r + 0.001f));
		}
		delete[] l_minus_means;
	};

	void calcNCCstrip_fast(const cv::Mat& img_l, const cv::Mat& img_r, const int& ul, const int& vl,
		const int& u_min, const int& u_max, const int& win_sz, const int& fat_sz,
		vector<float>& scores)
	{
		int win_len = win_sz * 2 + 1;
		int fat_len = fat_sz * 2 + 1;
		float* pat_a = new float[win_len*fat_len];
		int len = win_len * fat_len;
		int n_cols = img_l.cols;
		int n_rows = img_l.rows;

		scores.resize(0);
		float* ptr_l = (float*)img_l.ptr<float>(0);
		float* ptr_r = (float*)img_r.ptr<float>(0);

		float sa = 0.0f;
		float sa2 = 0.0f;
		float den_a = 0.0f;
		float invN = 1.0f / ((float)len);

		int cnt = 0;
		float* pa = nullptr;
		for (int f = -fat_sz; f < fat_sz + 1; f++) {
			int vcolsuw = (vl + f)*n_cols + ul - win_sz;
			pa = ptr_l + vcolsuw;
			for (int i = 0; i < win_len; i++) {
				*(pat_a + cnt) = *pa;
				sa += *pa;
				sa2 += (*pa)*(*pa);
				++pa;
				++cnt;
			}
		}
		den_a = sa2 - sa*sa*invN;
		for (int u_r = u_min; u_r < u_max + 1; u_r++)
		{
			float sb = 0.0f;
			float sb2 = 0.0f;
			float sab = 0.0f;
			float den_b = 0.0f;
			float* pb = nullptr;
			pa = pat_a;
			for (int f = -fat_sz; f < fat_sz + 1; f++) {
				int vcolsuw = (vl + f)*n_cols + u_r - win_sz;
				pb = ptr_r + vcolsuw;
				for (int i = 0; i < win_len; i++) {
					sb += *pb;
					sab += (*pa)*(*pb);
					sb2 += (*pb)*(*pb);
					++pb;
					++pa;
				}
			}
			den_b = sb2 - sb*sb*invN;

			scores.push_back((sab - sa*sb*invN) / sqrt(den_a*den_b + 0.0001f));
		}
		delete[] pat_a;
	};

	void calcNCCstrip_fast2(const cv::Mat& img_l, const cv::Mat& img_r, const int& ul, const int& vl,
		const int& u_min, const int& u_max, const int& win_sz, const int& fat_sz,
		vector<float>& scores)
	{
		int win_len = win_sz * 2 + 1;
		int fat_len = fat_sz * 2 + 1;
		float* pat_a = new float[win_len*fat_len];
		int len = win_len * fat_len;
		int n_cols = img_l.cols;
		int n_rows = img_l.rows;

		scores.resize(0);
		float* ptr_l = (float*)img_l.ptr<float>(0);
		float* ptr_r = (float*)img_r.ptr<float>(0);

		float sa = 0.0f;
		float sa2 = 0.0f;
		float den_a = 0.0f;
		float invN = 1.0f / ((float)len);

		int cnt = 0;
		float* pa = nullptr;
		for (int f = -fat_sz; f < fat_sz + 1; f++) {
			int vcolsuw = (vl + f)*n_cols + ul - win_sz;
			pa = ptr_l + vcolsuw;
			for (int i = 0; i < win_len; i++) {
				*(pat_a + cnt) = *pa;
				sa += *pa;
				sa2 += (*pa)*(*pa);
				++pa;
				++cnt;
			}
		}
		den_a = sa2 - sa*sa*invN;
		float sb = 0.0f; // 업데이트 가능
		float sb2 = 0.0f; // 업데이트 가능
		for (int u_r = u_min; u_r < u_max + 1; u_r++)
		{
			float sab = 0.0f;
			float den_b = 0.0f;
			float* pb = nullptr;
			pa = pat_a;
			if (u_r == u_min) {
				for (int f = -fat_sz; f < fat_sz + 1; f++) {
					int vcolsuw = (vl + f)*n_cols + u_r - win_sz;
					pb = ptr_r + vcolsuw;
					for (int i = 0; i < win_len; i++) {
						sb += *pb;
						sb2 += (*pb)*(*pb);
						sab += (*pa)*(*pb);
						++pa;
						++pb;
					}
				}
			}
			else {
				for (int f = -fat_sz; f < fat_sz + 1; f++) {
					int vcolsuw = (vl + f)*n_cols + u_r - win_sz - 1;
					pb = ptr_r + vcolsuw;
					sb -= *pb;
					sb += *(pb + win_len);
					sb2 -= (*pb)*(*pb);
					sb2 += (*(pb + win_len))*(*(pb + win_len));
				}
				for (int f = -fat_sz; f < fat_sz + 1; f++) {
					int vcolsuw = (vl + f)*n_cols + u_r - win_sz;
					pb = ptr_r + vcolsuw;
					for (int i = 0; i < win_len; i++) {
						sab += (*pb)*(*pa);
						++pb;
						++pa;
					}
				}
			}

			den_b = sb2 - sb*sb*invN;
			scores.push_back((sab - sa*sb*invN) / sqrt(den_a*den_b + 0.0001f));
		}

        delete[] pat_a;
	};



    void calcNCCstripDirectional_fast2(
        const cv::Mat& img_k, const cv::Mat& img_c,
        const chk::Point2f& pt_k, const chk::Point2f& pt_start, 
        const chk::Point2f& l_k, const chk::Point2f& l_c,
        const int& n_search, const int& win_sz, const int& fat_sz,
        vector<float>& scores)
    {        
        bool flag_debug = false;
        cv::Mat img_cu, img_ku;
        if (flag_debug) {            
            img_k.convertTo(img_ku, CV_8UC1);
            img_c.convertTo(img_cu, CV_8UC1);
        }
        

        scores.resize(0);
        chk::Point2f ln_k(-l_k.y, l_k.x);
        chk::Point2f ln_c(-l_c.y, l_c.x);
        int win_len = win_sz * 2 + 1;
        int fat_len = fat_sz * 2 + 1;
        int N = win_len * fat_len; // # of elements in a patch.
        float invN = 1.0f / ((float)N);

        int n_cols = img_k.cols;
        int n_rows = img_k.rows;

        float* ptr_k = (float*)img_k.ptr<float>(0);
        float* ptr_c = (float*)img_c.ptr<float>(0);

        // a: keyframe patch, b: current frame patch
        // 진행방향: -l_nc방향. l_nc*(-fat_sz+f) [f: 0~2*fat_sz]
        float* pat_a   = new float[win_len*fat_len];
        float* strip_b = new float[(win_sz * 2 + n_search)*fat_len];

        float sa    = 0.0f; // sum of Ik
        float sa2   = 0.0f; // sum of Ik^2
        float den_a = 0.0f; // denominator ( sqrt( sum((Ik-mk)^2) ))

        // Make patch_a 
        int cnt = 0;
        for (int w = -win_sz; w < win_sz + 1; w++) {
            chk::Point2f pt_strip_center = (chk::Point2f)pt_k + ((chk::Point2f)l_k)*w;
            for (int f = 0; f < fat_len; f++) {
                chk::Point2f pt_now = pt_strip_center + ln_k*(f-fat_sz);
                float Ik_interp = interpImageSingle(img_k, pt_now.x, pt_now.y);
                *(pat_a + cnt) = Ik_interp;
                sa += Ik_interp;
                sa2 += Ik_interp*Ik_interp;
                ++cnt;

                if (flag_debug) cv::circle(img_ku, cv::Point2f(pt_now.x,pt_now.y),1,cv::Scalar(0,0,255),1);

                
            }
        }
        den_a = sa2 - sa*sa*invN;
        if (den_a < 0.001f) {
            throw std::runtime_error("den_a == 0!!\n");
        }
        else {
            // Make strip_b
            chk::Point2f pt_fat_start = (chk::Point2f)pt_start + ((chk::Point2f)l_c)*(-win_sz) + ln_c*(-fat_sz);
            cnt = 0;
            for (int w = 0; w < n_search + 2 * win_sz; w++) {
                for (int f = 0; f < fat_len; f++) {
                    chk::Point2f pt_now = pt_fat_start + ln_c*f;
                    float Ic_interp = interpImageSingle(img_c, pt_now.x, pt_now.y);
                    *(strip_b + cnt) = Ic_interp;

                    if (flag_debug) cv::circle(img_cu, cv::Point2f(pt_now.x, pt_now.y), 1, cv::Scalar(0, 0, 255), 1);

                    ++cnt;
                }
                pt_fat_start += l_c;
            }

            if (flag_debug) {
                cv::namedWindow("temporal_k");
                cv::namedWindow("temporal_c");
                cv::imshow("temporal_k", img_ku);
                cv::imshow("temporal_c", img_cu);
                cv::waitKey(0);
            }

            // score 계산
            float sb = 0.0f; // 업데이트 가능
            float sb2 = 0.0f; // 업데이트 
            float* pb_patchstart = strip_b;
            for (int i = 0; i < n_search; i++)
            {
                float sab = 0.0f;
                float den_b = 0.0f;
                float* pa = pat_a;
                if (!i) {
                    float* pb = strip_b;
                    for (int j = 0; j < N; j++) {
                        sb += *pb;
                        sb2 += (*pb)*(*pb);
                        sab += (*pa)*(*pb);
                        ++pa;
                        ++pb;
                    }
                }
                else {
                    float* pb = pb_patchstart; // 이전 patch에서의 시작점.
                    for (int f = 0; f < fat_len; f++) {
                        sb -= *pb;
                        sb2 -= (*pb)*(*pb);

                        sb += *(pb + N);
                        sb2 += (*(pb + N))*(*(pb + N));
                        ++pb;
                    }

                    pb_patchstart += fat_len; // 이번 패치 시작! 
                    pb = pb_patchstart;
                    for (int j = 0; j < N; j++) {
                        sab += (*pb)*(*pa);
                        ++pa;
                        ++pb;
                    }
                }

                den_b = sb2 - sb*sb*invN;
                if (den_b < 0.01f) scores.push_back(-1);
                else scores.push_back((sab - sa*sb*invN) / sqrt(den_a*den_b));
            }
        }
        delete[] pat_a;
        delete[] strip_b;
     };
};
#endif