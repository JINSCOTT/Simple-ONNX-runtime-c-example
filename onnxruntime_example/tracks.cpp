#include "tracks.h"


Track::Track(int _label, float x, float y, float w, float h, float maxX, float maxY, int _TrackNum) {
	this->label = _label;
	// Init kalman
	this->_isFound = true;
	this->TrackNum = _TrackNum;
	this->km.Init(x, y, w, h, maxX, maxY);
}

bool Track::isSame(int _label, float x, float y, float w, float h) {
	if (this->label == _label && km.CloseProximity(cv::Point(x + w / 2, y + h / 2))) {
		this->_isFound = true;
		km.Correct(x,y,w,h);
		return true;
	}
	else return false;
}
cv::Point Track::Predict() {
	return km.Predict();
}
int Track::TrackNumber() {
	return this->TrackNum;
}
bool Track::isFound() {
	return this->_isFound;
}
bool Track::isDead() {
	if (_isFound) {
		_isFound = false;	// reset
		this->LostCounter = 0;
		return false;
	}
	else {
		this->LostCounter++;
		if (LostCounter == LostLimit) return true;
		else return false;
	}
}
void Track::SetLostLimit(int cnt) {
	this->LostLimit = cnt;
}

void Track::Draw(cv::Mat& frame) {
	// Only start to draw after we are sure the object really exsists
	if (LostCounter < km.HistorySize()) {
		cv::putText(frame, LabelString[this->label] + std::to_string(TrackNum), km.pos(), cv::FONT_HERSHEY_PLAIN, 2, (255, 0, 0));
		this->km.DrawTrack(frame);
	}

}

void Track::SetColor(cv::Scalar _NewColor) {
	this->Color = _NewColor;
	km.SetColor(_NewColor);
}
void Track::SetDrawSize(int _NewSize) {
	this->DrawSize = _NewSize;
	km.SetDrawSize(_NewSize);
}
TrackManager::TrackManager() :mt((std::random_device())()) {
	// Something to do?
}
void TrackManager::SetCols(int value) {
	this->Cols = value;
}
void TrackManager::SetRows(int value) {
	this->Rows = value;
}
void TrackManager::SetMLCols(int value) {
	this->MLCols = value;
}
void TrackManager::SetMLRows(int value) {
	this->MLRows = value;
}


void TrackManager::ProcessInput(float* mlres, int size) {
	for (auto tr : Tracks) {
		tr->Predict();
	}

	for (int i = 0; i < size; i += 7) {		// the output of this model is (number_of_detected , 7), thus the increment by 7
		if (mlres[i + 6] > 0.8) {
			float x = mlres[i + 1] * (float)Cols / MLCols;
			float y = mlres[i + 2] * (float)Rows / MLRows;
			float w = (mlres[i + 3] - mlres[i + 1]) * (float)Cols / (float)MLCols;
			float h = (mlres[i + 4] - mlres[i + 2]) * (float)Rows / (float)MLRows;
			int label = mlres[i + 5];
			bool found = false;
			std::cout << i / 7 << ", " << LabelString[label] << " acc: " << mlres[i + 6] << " x," << x << " y," << y << " w," << w << " h," << h;
			for (auto tr : Tracks) {
				if (!tr->isFound()) {
					if (tr->isSame(label, x, y, w, h)) {
						found = true;
					}
				}
			}
			// Not Found, must be new
			if (!found) {
				Track* NewTrack = new Track(label, x, y, w, h, Cols, Rows, TrackNum);
				TrackNum += 1;
				std::cout << "New track: " << TrackNum << "\n";
				std::uniform_int_distribution<int> dist(0, Colors.size());
				int rnd= dist(mt);
				std::cout <<"rand"<< rnd << "\n";
				NewTrack->SetColor(Colors[rnd]);
				Tracks.push_back(NewTrack);
			}
		}
	}
	// Check whether is dead
	for (int i = 0; i < Tracks.size(); i++) {
		if (Tracks[i]->isDead()) {
			std::cout << "Track " << Tracks[i]->TrackNumber()  << "Lost";
			delete(Tracks[i]);
			Tracks.erase(Tracks.begin() + i);
			i--;
		}
	}
	std::cout << "Tracks: " << Tracks.size()<<"\n";
}

void TrackManager::Draw(cv::Mat& Frame) {
	for (auto tr : Tracks) {
		tr->Draw(Frame);
	}
}