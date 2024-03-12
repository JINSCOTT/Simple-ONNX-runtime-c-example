#pragma once
// based on https://raw.githubusercontent.com/tobybreckon/python-examples-cv/master/kalman_tracking_live.py by Toby Breckon 

#include "opencv2/video/tracking.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/cvdef.h"
#include <iostream>
#include <deque>
#include "utils.h"
class KalmanFilter2D {
public:
	void Init(float StartX, float StartY, float StartW, float StartH, int _Col, int _Row);
	// Predict new
	cv::Point Predict();
	// Correct position
	void Correct(float x, float y, float w, float h);
	// Is object close to predicted position?
	bool CloseProximity(cv::Point Pos);
	void DrawTrack(cv::Mat& frame);
	void SetColor(cv::Scalar _NewColor);
	void SetDrawSize(int _NewSize);
	int HistorySize();
	cv::Point pos();
private:
	std::deque<cv::Point> TrackHistory;	// Push back pop front
	cv::KalmanFilter kalman = cv::KalmanFilter(4,2,0) ;
	cv::Scalar Color = cv::Scalar(0, 255, 0);
	int DrawSize = 1;
	int Cols = 640;
	int Rows = 480;

};

