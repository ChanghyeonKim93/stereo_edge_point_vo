//#include <iostream>
//#include <exception>
//#include "SEPVO.h"
//
//using namespace std;
//int main() {
//	string leftDir = "F:\\#연구\\#DATASET\\euroc\\V1_01_easy\\mav0\\cam0\\data";
//	string rightDir = "F:/#연구/#DATASET/euroc/V1_01_easy/mav0/cam1/data";
//	string timeDir = "F:/#연구/#DATASET/euroc/cpp_informs/V101.txt";
//	string calibDir = "F:/#연구/#DATASET/euroc/cpp_informs/EuRoC.yaml";
//
//	/*string leftDir = "D:/research/DATASET/euroc/V1_01_easy/mav0/cam0/data";
//	string rightDir = "D:/research/DATASET/euroc/V1_01_easy/mav0/cam1/data";
//	string timeDir = "D:/research/DATASET/euroc/cpp_informs/V101.txt";
//	string calibDir = "D:/research/DATASET/euroc/cpp_informs/EuRoC.yaml";*/
//
//	StereoEPVO* stereoVO = new StereoEPVO(leftDir, rightDir, timeDir, calibDir);
//	
//	cv::Mat img_l, img_r;
//	img_l = cv::imread(stereoVO->leftnames[50]);
//	img_r = cv::imread(stereoVO->rightnames[50]);
//
//	img_l.convertTo(img_l, CV_32FC1);
//	img_r.convertTo(img_r, CV_32FC1);
//
//	int n_cols = img_l.cols;
//	int n_rows = img_l.rows;
//
//	float* pL = img_l.ptr<float>(0);
//	float* pR = img_r.ptr<float>(0);
//	float* pRes = new float[n_cols*n_rows];
//
//	int nCacheline = 64 / sizeof(float);
//	cout << "# of fetched elements at once: " << nCacheline << endl;
//
//	tic();
//	int vncols = 0;
//	int ind = 0;
//	for (int v = 0; v < n_rows; v++) {
//		vncols = v*n_cols;
//		for (int u = 0; u < n_rows; u+= nCacheline) {
//			ind = vncols + u;
//			for (int j = 0; j < nCacheline; j++) {
//				*(pRes + ind) = (*(pL+ind))*(*(pR + ind));
//				++ind;
//			}
//		}
//	}
//	cout << "Cache: " << toc(0) << " [ms]\n";
//
//	tic();
//	for (int v = 0; v < n_rows; v+=1) {
//		vncols = v*n_cols;
//		ind = vncols;
//		for (int u = 0; u < n_rows; u +=1) {
//			*(pRes + ind) = (*(pL + ind))*(*(pR + ind));
//			++ind;;
//		}
//	}
//	cout << "Normal: " << toc(0) << " [ms]\n";
//
//
//	// 결론. 무의미! 속도 더느려지네 ;
//
//	delete stereoVO;
//	printf("End of the program...\n");
//	return 0;
//}