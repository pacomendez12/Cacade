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
	Mat frame, img;
	double t;
	namedWindow( "image", WINDOW_AUTOSIZE );
	string str = "data/training.xml";
	if (cascade.load_from_file(str)) {
		cout << "Training loaded correctly" << endl;
		capture.open(0);
		if (!capture.isOpened()) {
			cout << "Camera not ready" << endl;
			return -1;
		}

		for (;;) {
			capture >> frame;
			//frame = imread("data/image.jpg", 1);
			if (frame.empty()) {
				cout << "could not read frame" << endl;
				return -1;
			}

			cvtColor( frame, img, COLOR_BGR2GRAY );
			double scale = 1;
			double fx = 1 / scale;
			resize( img, img, Size(), fx, fx, INTER_LINEAR );
			equalizeHist( img, img );

			t = (double)cvGetTickCount();
			cascade.detectMultiScale(img, faces, 1.1, 2, 0, Size(10, 10), Size(500, 500));
			t = (double)cvGetTickCount() - t;
			cout << "faces detected = " << faces.size() << " in " <<
				t/((double)cvGetTickFrequency() * 1000.0) << "ms" << endl;

			for(int i = 0; i < faces.size(); i++) {
				Rect r = faces[i];
				double aspectRatio = (double) r.width / r.height;
				Point c;
				int radius;

				if (0.75 < aspectRatio && aspectRatio < 1.3) {
					c.x = cvRound((r.x + r.width * 0.5) * scale);
					c.y = cvRound((r.y + r.height * 0.5) * scale);
					radius = cvRound((r.width + r.height) * 0.25 * scale);
					circle(frame, c, radius, Scalar(255, 0, 0), 3, 8, 0);
				} else {
					rectangle( frame, cvPoint(cvRound(r.x * scale), cvRound(r.y * scale)),
							cvPoint(cvRound((r.x + r.width - 1) * scale), cvRound((r.y + r.height - 1) * scale)),
							Scalar(255, 0, 0), 3, 8, 0);
				}

			}

			imshow("image", frame);
			int c = waitKey(1);
			if( c == 27 || c == 'q' || c == 'Q' )
				break;
		}


	} else
		cout << "failed while loading training" << endl;
	return 0;
}
