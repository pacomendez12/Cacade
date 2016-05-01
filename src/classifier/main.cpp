


#include <train/cascade_data.h>
#include <classifier/classifier.h>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>


int main(int argc, const char *argv[])
{
	Cascade cascade;
	vector<Rect> faces;
	VideoCapture capture;
	Mat frame;
	string str = "data/training.xml";
	if (cascade.load_from_file(str)) {
		cout << "Training loaded correctly" << endl;
		capture.open(0);
		capture >> frame;
		if (frame.empty()) {
			cout << "could not read frame" << endl;
		}
		cascade.detectMultiScale(frame, faces, 1.1, 2, 0, Size(10, 10), Size(30, 30));


	} else
		cout << "failed while loading training" << endl;
	return 0;
}
