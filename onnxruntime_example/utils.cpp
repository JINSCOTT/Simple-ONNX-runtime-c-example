#include "utils.h"

cv::Mat Center(float x, float y, float w, float h) {
	
	return (cv::Mat_<float>(2, 1) << x + w / 2, y + h / 2);
}