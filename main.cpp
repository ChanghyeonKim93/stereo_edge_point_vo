#include <iostream>
#include <exception>
#include "includes/SEPVO.h"
#define EIGEN_MAX_ALIGN_BYTES 32
#include "includes/custom_memory.h"

using namespace std;
int main() {
	string leftDir = "F:\\#楷备\\#DATASET\\euroc\\V1_01_easy\\mav0\\cam0\\data";
	string rightDir = "F:/#楷备/#DATASET/euroc/V1_01_easy/mav0/cam1/data";
	string timeDir = "F:/#楷备/#DATASET/euroc/cpp_informs/V101.txt";
	string calibDir = "F:/#楷备/#DATASET/euroc/cpp_informs/EuRoC.yaml";

	/*string leftDir  = "D:/research/DATASET/euroc/V1_01_easy/mav0/cam0/data";
	string rightDir = "D:/research/DATASET/euroc/V1_01_easy/mav0/cam1/data";
	string timeDir  = "D:/research/DATASET/euroc/cpp_informs/V101.txt";
	string calibDir = "D:/research/DATASET/euroc/cpp_informs/EuRoC.yaml";*/

	StereoEPVO* stereoVO = new StereoEPVO(leftDir, rightDir, timeDir, calibDir);
	try {
		stereoVO->run(); // algorithm runs.

		delete stereoVO;
		printf("All allocations are returned...\n");
	}
	catch (std::exception& e) {
		std::cout << "\n[Exception]: " << e.what();
		delete stereoVO;
	}

	printf("End of the program...\n");
	return 0;
}