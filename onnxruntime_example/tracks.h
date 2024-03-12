#pragma once



#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui.hpp"
#include "kalman.h"
#include "labels.h"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>



class Track {
public:
	Track(int _label, float x, float y, float w, float h, float Cols, float ROws, int _TrackNum) ;
	// A weak method to check if it is the same object
	bool isSame(int _label, float x,float y,float w,float h);

	cv::Point Predict();
	bool isFound();
	bool isDead();
	void SetLostLimit(int cnt);
	void Draw(cv::Mat& frame);
	void SetColor(cv::Scalar _NewColor);
	void SetDrawSize(int _NewSize);
	int TrackNumber();
protected:
	unsigned int label = 100;
	KalmanFilter2D km;
	cv::Scalar Color = cv::Scalar(0, 255, 0);
	int DrawSize = 1;
	bool _isFound = false;
	unsigned int LostCounter = 0;
	unsigned int LostLimit = 30;
	int TrackNum = 0;
};

class TrackManager {
public:
	TrackManager();
	void SetCols(int val);
	void SetRows(int val);
	void SetMLCols(int val);
	void SetMLRows(int val);
	void ProcessInput(float* mlres, int size);
	void Draw(cv::Mat &Frame);
private:
	std::vector<Track*> Tracks;
	// Dimension of original video frame and machine learning model
	float Cols = 640, Rows = 480, MLCols = 640, MLRows = 640;
	int TrackNum = 0;
	std::mt19937 mt;

};