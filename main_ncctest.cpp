//#include <iostream>
//#include <exception>
//#include "SEPVO.h"
//// https://stackoverflow.com/questions/42718167/sse-optimization-of-sum-of-squared-differences
//
//using namespace std;
//
//
//int main() {
//	string leftDir = "F:\\#연구\\#DATASET\\euroc\\V1_01_easy\\mav0\\cam0\\data";
//	string rightDir = "F:/#연구/#DATASET/euroc/V1_01_easy/mav0/cam1/data";
//	string timeDir = "F:/#연구/#DATASET/euroc/cpp_informs/V101.txt";
//	string calibDir = "F:/#연구/#DATASET/euroc/cpp_informs/EuRoC.yaml";
//
//	/*string leftDir  = "D:/research/DATASET/euroc/V1_01_easy/mav0/cam0/data";
//	string rightDir = "D:/research/DATASET/euroc/V1_01_easy/mav0/cam1/data";
//	string timeDir  = "D:/research/DATASET/euroc/cpp_informs/V101.txt";
//	string calibDir = "D:/research/DATASET/euroc/cpp_informs/EuRoC.yaml";*/
//
//	StereoEPVO* stereoVO = new StereoEPVO(leftDir, rightDir, timeDir, calibDir);
//	cv::Mat img_l = cv::imread(stereoVO->leftnames[0], CV_LOAD_IMAGE_GRAYSCALE);
//	cv::Mat img_r = cv::imread(stereoVO->rightnames[0], CV_LOAD_IMAGE_GRAYSCALE);
//
//	cv::namedWindow("img", CV_WINDOW_AUTOSIZE);
//	cv::imshow("img", img_l);
//	cv::waitKey(5);
//
//	cv::Mat img_lf;
//	cv::Mat img_rf;
//	stereoVO->cams->undistortImages(img_l, img_r, img_l, img_r);
//
//	img_l.convertTo(img_lf, CV_32FC1);
//	img_r.convertTo(img_rf, CV_32FC1);
//
//	int n_iter = 10000;
//
//	int win_sz = 7;
//	int fat_sz = 3;
//	int len = (2 * win_sz + 1)*(2 * fat_sz + 1);
//	cv::Point2f pt(600, 240);
//	int u_max = pt.x - 10;
//	int u_min = win_sz + 1;
//	int u_l = pt.x;
//	int v_l = pt.y;
//
//	// 일반적인 계산
//	vector<float> score_normal;
//	vector<float> score_pre;
//	vector<float> score_fast;
//	vector<float> score_fast2;
//
//	score_normal.reserve(u_max - u_min + 1);
//	score_pre.reserve(u_max - u_min + 1);
//	score_fast.reserve(u_max - u_min + 1);
//	score_fast2.reserve(u_max - u_min + 1);
//
//	int n_cols = img_lf.cols;
//	int n_rows = img_lf.rows;
//	tic();
//	for (int iter = 0; iter < n_iter; iter++)
//	{
//		score_normal.resize(0);
//		float* ptr_l = img_lf.ptr<float>(0);
//		float* ptr_r = img_rf.ptr<float>(0);
//		float invn_elem = 1.0f / ((float)len);
//		for (int u_r = u_min; u_r < u_max; u_r++) {
//
//			float mean_l = 0.0f;
//			float mean_r = 0.0f;
//			float numer = 0.0f;
//			float denom_l = 0.0f;
//			float denom_r = 0.0f;
//			for (int f = -fat_sz; f < fat_sz + 1; f++) {
//				for (int w = -win_sz; w < win_sz + 1; w++) {
//					mean_l += *(ptr_l + (v_l + f)*n_cols + u_l + w);
//					mean_r += *(ptr_r + (v_l + f)*n_cols + u_r + w);
//				}
//			}
//			mean_l *= invn_elem;
//			mean_r *= invn_elem;
//
//			//calculate costs
//			float l_minus_mean;
//			float r_minus_mean;
//			for (int f = -fat_sz; f < fat_sz + 1; f++) {
//				for (int w = -win_sz; w < win_sz + 1; w++) {
//					l_minus_mean = *(ptr_l + (v_l + f)*n_cols + u_l + w) - mean_l;
//					r_minus_mean = *(ptr_r + (v_l + f)*n_cols + u_r + w) - mean_r;
//					numer += l_minus_mean*r_minus_mean;
//					denom_l += l_minus_mean*l_minus_mean;
//					denom_r += r_minus_mean*r_minus_mean;
//				}
//			}
//			score_normal.push_back(numer / sqrt(denom_l*denom_r + 0.0001f));
//		}
//	}
//
//	cout << "normal: " << toc(0) << endl;
//
//	tic();
//	int win_len = win_sz * 2 + 1;
//	int fat_len = fat_sz * 2 + 1;
//	float* l_minus_means = new float[win_len*fat_len];
//
//	for (int iter = 0; iter < n_iter; iter++)
//	{
//		score_pre.resize(0);
//		float* ptr_l = img_lf.ptr<float>(0);
//		float* ptr_r = img_rf.ptr<float>(0);
//		float* ptr_l_temp = nullptr;
//		float* ptr_r_temp = nullptr;
//		float invn_elem = 1.0f / ((float)len);
//
//		float mean_l = 0.0f;
//		for (int f = -fat_sz; f < fat_sz + 1; f++) {
//			int vlfncolsuwin = (v_l + f)*n_cols + u_l - win_sz;
//			ptr_l_temp = ptr_l + vlfncolsuwin;
//			for (int w = -win_sz; w < win_sz + 1; w++)
//				mean_l += *(ptr_l_temp++);
//		}
//		mean_l *= invn_elem;
//
//		float denom_l = 0.0f;
//		int cnt = 0;
//		for (int f = -fat_sz; f < fat_sz + 1; f++) {
//			int vlfncolsuwin = (v_l + f)*n_cols + u_l - win_sz;
//			ptr_l_temp = ptr_l + vlfncolsuwin;
//			for (int w = -win_sz; w < win_sz + 1; w++) {
//				l_minus_means[cnt] = *(ptr_l_temp++) - mean_l;
//				denom_l += l_minus_means[cnt] * l_minus_means[cnt];
//				++cnt;
//			}
//		}
//
//
//		for (int u_r = u_min; u_r < u_max; u_r++)
//		{
//
//			float mean_r = 0.0f;
//			float numer = 0.0f;
//			float denom_r = 0.0f;
//			for (int f = -fat_sz; f < fat_sz + 1; f++)
//			{
//				int vlfncolsuwin = (v_l + f)*n_cols + u_r - win_sz;
//				ptr_r_temp = ptr_r + vlfncolsuwin;
//				for (int w = -win_sz; w < win_sz + 1; w++)
//					mean_r += *(ptr_r_temp++);
//			}
//
//			mean_r *= invn_elem;
//
//			//calculate costs
//			cnt = 0;
//			float r_minus_mean;
//			for (int f = -fat_sz; f < fat_sz + 1; f++) {
//				int vlfncolsuwin = (v_l + f)*n_cols + u_r - win_sz;
//				ptr_r_temp = ptr_r + vlfncolsuwin;
//				for (int w = -win_sz; w < win_sz + 1; w++) {
//					r_minus_mean = *(ptr_r_temp++) - mean_r;
//					numer += l_minus_means[cnt] * r_minus_mean;
//					denom_r += r_minus_mean*r_minus_mean;
//					++cnt;
//				}
//			}
//			score_pre.push_back(numer / sqrt(denom_l*denom_r + 0.0001f));
//		}
//	}
//	cout << "pre-calculate: " << toc(0) << endl;
//
//	tic();
//	float* pat_a = new float[win_len*fat_len];
//	float* ptr_l = img_lf.ptr<float>(0);
//	float* ptr_r = img_rf.ptr<float>(0);
//
//	for (int iter = 0; iter < n_iter; iter++)
//	{
//		score_fast.resize(0);
//
//		// (1) sa, sa2, pat_a, sasa, den_a
//		float sa = 0.0f;
//		float sa2 = 0.0f;
//		float den_a = 0.0f;
//		float invN = 1.0f / ((float)len);
//
//		int cnt = 0;
//		float* pa = nullptr;
//		for (int f = -fat_sz; f < fat_sz + 1; f++) {
//			int vcolsuw = (v_l + f)*n_cols + u_l - win_sz;
//			pa = ptr_l + vcolsuw;
//			for (int i = 0; i < win_len; i++) {
//				*(pat_a + cnt) = *pa;
//				sa += *pa;
//				sa2 += (*pa)*(*pa);
//				++pa;
//				++cnt;
//			}
//		}
//		den_a = sa2 - sa*sa*invN;
//
//		for (int u_r = u_min; u_r < u_max; u_r++)
//		{
//			float sb = 0.0f;
//			float sb2 = 0.0f;
//			float sab = 0.0f;
//			float den_b = 0.0f;
//			float* pb = nullptr;
//			pa = pat_a;
//			for (int f = -fat_sz; f < fat_sz + 1; f++) {
//				int vcolsuw = (v_l + f)*n_cols + u_r - win_sz;
//				pb = ptr_r + vcolsuw;
//				for (int i = 0; i < win_len; i++) {
//					sb += *pb;
//					sab += (*pa)*(*pb);
//					sb2 += (*pb)*(*pb);
//					++pb;
//					++pa;
//				}
//			}
//			den_b = sb2 - sb*sb*invN;
//
//			score_fast.push_back((sab - sa*sb*invN) / sqrt(den_a*den_b + 0.0001f));
//		}
//	}
//	cout << "pre-calculate & another approach: " << toc(0) << endl;
//
//	tic();
//	for (int iter = 0; iter < n_iter; iter++)
//	{
//		score_fast2.resize(0);
//
//		// (1) sa, sa2, pat_a, sasa, den_a
//		float sa = 0.0f;
//		float sa2 = 0.0f;
//		float den_a = 0.0f;
//		float invN = 1.0f / ((float)len);
//
//		int cnt = 0;
//		float* pa = nullptr;
//		for (int f = -fat_sz; f < fat_sz + 1; f++) {
//			int vcolsuw = (v_l + f)*n_cols + u_l - win_sz;
//			pa = ptr_l + vcolsuw;
//			for (int i = 0; i < win_len; i++) {
//				*(pat_a + cnt) = *pa;
//				sa += *pa;
//				sa2 += (*pa)*(*pa);
//				++pa;
//				++cnt;
//			}
//		}
//		den_a = sa2 - sa*sa*invN;
//		float sb = 0.0f; // 업데이트 가능
//		float sb2 = 0.0f; // 업데이트 가능
//		for (int u_r = u_min; u_r < u_max; u_r++)
//		{
//			float sab = 0.0f;
//			float den_b = 0.0f;
//			float* pb = nullptr;
//			pa = pat_a;
//			if (u_r == u_min) {
//				for (int f = -fat_sz; f < fat_sz + 1; f++) {
//					int vcolsuw = (v_l + f)*n_cols + u_r - win_sz;
//					pb = ptr_r + vcolsuw;
//					for (int i = 0; i < win_len; i++) {
//						sb += *pb;
//						sb2 += (*pb)*(*pb);
//						sab += (*pa)*(*pb);
//						++pa;
//						++pb;
//					}
//				}
//			}
//			else {
//				for (int f = -fat_sz; f < fat_sz + 1; f++) {
//					int vcolsuw = (v_l + f)*n_cols + u_r - win_sz - 1;
//					pb = ptr_r + vcolsuw;
//					sb -= *pb;
//					sb += *(pb + win_len);
//					sb2 -= (*pb)*(*pb);
//					sb2 += (*(pb + win_len))*(*(pb + win_len));
//				}
//				for (int f = -fat_sz; f < fat_sz + 1; f++) {
//					int vcolsuw = (v_l + f)*n_cols + u_r - win_sz;
//					pb = ptr_r + vcolsuw;
//					for (int i = 0; i < win_len; i++) {
//						sab += (*pb)*(*pa);
//						++pb;
//						++pa;
//					}
//				}
//			}
//
//			den_b = sb2 - sb*sb*invN;
//			score_fast2.push_back((sab - sa*sb*invN) / sqrt(den_a*den_b + 0.0001f));
//		}
//	}
//	delete[] pat_a;
//	cout << "pre-calculate & another approach2: " << toc(0) << endl;
//
//	for (int i = 0; i < score_fast.size(); i++) {
//	cout << score_normal.at(i)<<", "<< score_pre.at(i)<<", "<< score_fast.at(i)<<", "<< score_fast2.at(i) << "\n";
//	}
//	cout << endl;
//
//	if (0) {
//		try {
//			stereoVO->run(); // algorithm runs.
//
//			delete stereoVO;
//			printf("All allocations are returned...\n");
//		}
//		catch (std::exception& e) {
//			std::cout << "\n[Exception]: " << e.what();
//			delete stereoVO;
//		}
//	}
//	else {
//		delete stereoVO;
//	}
//
//	printf("End of the program...\n");
//	return 0;
//}