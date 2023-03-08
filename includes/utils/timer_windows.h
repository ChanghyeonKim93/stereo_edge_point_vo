#ifndef _TIMER_WINDOWS_H_
#define _TIMER_WINDOWS_H_
#include <chrono>
#include <iostream>

auto start = std::chrono::high_resolution_clock::now();
auto finish = std::chrono::high_resolution_clock::now();
auto gap = finish - start;

void tic() {
	start = std::chrono::high_resolution_clock::now();
}
double toc(bool flag_verbose) {
	finish = std::chrono::high_resolution_clock::now();
	gap = finish - start;
	if (flag_verbose) {
		std::cout << "exec time: " << (double)(gap / std::chrono::microseconds(1)) / 1000.0 << "[ms]" << std::endl;
	}
	return (double)(gap / std::chrono::microseconds(1)) / 1000.0;
}
#endif