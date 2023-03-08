#ifndef _DEPTH_FILTER_
#define _DEPTH_FILTER_
#include <iostream>
#include <vector>
#include <math.h>
#define pi 3.141592

class DepthFilter {
public:
	DepthFilter();
	~DepthFilter();
	void updateDF(float& invd, float& std_invd);
public:
	float a() { return a_; };
	float b() { return b_; };
	float mu() { return mu_; };
	float sig() { return sig_; };
	float zmin() { return z_min_; };
	float zmax() { return z_max_; };

	void set_mu(float in) { mu_ = in; };
	void set_sig(float in) { sig_ = in; };
	void set_zmin(float in) { z_min_ = in; };
	void set_zmax(float in) { z_max_ = in; };

private:
	float a_;
	float b_;
	float mu_;
	float sig_;
	float z_min_;
	float z_max_;
};
	

/* ================================================================================
================================= Implementation ==================================
=================================================================================== */
DepthFilter::DepthFilter() {
	// initialize
	a_ = 0.5f;
	b_ = 0.5f;
	mu_ = 0.0f;
	sig_ = 3.0f;
	z_min_ = 0.05f; // [m]
	z_max_ = 20.0f; // [m]
};
DepthFilter::~DepthFilter() {

};

void DepthFilter::updateDF(float& invd, float& std_invd) {
	// update limits
	if (z_min_ > invd)		z_min_ = invd;
	if (z_max_ < invd)		z_max_ = invd;

	float m = (sig_*sig_*invd + std_invd*std_invd* mu_) / (sig_*sig_ + std_invd*std_invd);
	float s = std_invd*sig_ / sqrt(sig_*sig_ + std_invd*std_invd);
	float C1 = 1.0 / sqrt(2 * pi*(sig_*sig_ + std_invd*std_invd))*exp(-(invd - mu_)*(invd - mu_) / (std_invd*std_invd + sig_*sig_)*0.5);
	float C2 = b_ / ((a_ + b_) * (z_max_ - z_min_));
	float invC1C2 = 1.0/(C1 + C2);
	C1 *= invC1C2;
	C2 *= invC1C2;

	float A = C1*(a_ + 1) / (a_ + b_ + 1) + C2*a_ / (a_ + b_ + 1);
	float B = 1 / A*(C1*(a_ + 1)*(a_ + 2) / (a_ + b_ + 1) / (a_ + b_ + 2) + C2*(a_ + 1)*a_/ (a_ + b_ + 1) / (a_ + b_ + 2));

	float a_new = A / (A - B)*(B - 1);
	float b_new = 1 / (B - A)*(A - 1)*(B - 1);
	float mu_new = C1*m + C2*mu_;
	float sig_new = sqrt(C1*(m*m + s*s) + C2*(mu_*mu_ + sig_*sig_) - mu_new*mu_new);

	// fill out
	a_ = a_new;
	b_ = b_new;
	mu_ = mu_new;
	sig_ = sig_new;
};
#endif