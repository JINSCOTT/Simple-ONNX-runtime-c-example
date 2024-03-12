#include "kalman.h"

void KalmanFilter2D::Init(float StartX, float StartY, float StartW, float StartH, int _Col, int _Row) {
	this->Cols = _Col;
	this->Rows = _Row;
	// Initate kalman filter
	kalman.statePre = (cv::Mat_<float>(4, 1) << StartX + StartW / 2, StartY + StartH / 2, 0, 0);
	kalman.statePost = (cv::Mat_<float>(4, 1) << StartX + StartW / 2, StartY + StartH / 2, 0, 0);
	kalman.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);

	kalman.processNoiseCov = (cv::Mat_<float>(4, 4) << 0.3f, 0, 0, 0, 0, 0.3f, 0, 0, 0, 0, 0.3f, 0, 0, 0, 0, 0.3f);
	kalman.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 1, 0, 0);
	kalman.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F);
	
	TrackHistory.push_back(cv::Point(StartX + StartW / 2, StartY + StartH / 2));
}
void KalmanFilter2D::Correct(float x, float y, float w, float h) {
	kalman.correct(Center(x, y, w, h));
}
cv::Point KalmanFilter2D::Predict() {

	// Don't predict if out of sight
	if (TrackHistory.back().x > Cols || TrackHistory.back().x < 0 || TrackHistory.back().y > Rows || TrackHistory.back().y < 0) {
		return TrackHistory.back();
	}
	cv::Mat Predictions = cv::Mat_<float>(2,1);
	try {
		Predictions = kalman.predict();
	}
	catch (cv::Exception& e) {
		std::cout << e.what() << " " << e.what() << "\n";
	}
	TrackHistory.push_back(cv::Point(Predictions.at<float>(0) , Predictions.at<float>(1) ));
	return TrackHistory.back();
}

bool KalmanFilter2D::CloseProximity(cv::Point Pos) {
	// This is not a good way to guess they are the same thing
	if (cv::norm(Pos - TrackHistory.back()) < Cols / 5) {
		std::cout << "Is close\n";
		return true;
	}
	else return false;
}
void KalmanFilter2D::DrawTrack(cv::Mat& frame) {
	for (int i = 0; i < TrackHistory.size() - 1; i++) {
		cv::line(frame, TrackHistory[i], TrackHistory[i + 1], this->Color, DrawSize);
	}
}

void KalmanFilter2D::SetColor(cv::Scalar _NewColor) {
	this->Color = _NewColor;
}
void KalmanFilter2D::SetDrawSize(int _NewSize) {
	this->DrawSize = _NewSize;
}

int KalmanFilter2D::HistorySize() {
	return TrackHistory.size();
}

cv::Point KalmanFilter2D::pos() {
	return TrackHistory.back();
}