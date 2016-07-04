#ifndef CLASSIFIER_H

#define CLASSIFIER_H

#include <train/cascade_data.h>
#include <classifier/feature_evaluator.h>
#include "opencv2/objdetect.hpp"

using namespace cv;

class Cascade;


class Classifier {
	public:
		Classifier() {}

		void runClassifier(int start, int end);

		void setDataClassifier(Cascade * cptr, int _nscales, int _nstripes,
			const feature_evaluator::ScaleData * _scaleData, const int * _stripeSize,
			std::vector<Rect>& _vec, std::vector<int>& _levels,
			std::vector<double>& _weights, bool outputLevels, Mutex& _mtx);


	private:
		Cascade * CascadePtr;
		int nscales;
		int nstripes;
		const feature_evaluator::ScaleData * scaleData;
		const int * stripeSizes;
		std::vector<Rect>* rectangles;
		std::vector<int>* rejectedLevels;
		std::vector<double>* levelWeights;
		std::vector<float> scales;
		Mutex * mtx;
};


class Cascade {
	friend class Classifier;
	public:
		Cascade();
		~Cascade();
		bool load_from_file(const std::string file);
		inline bool isLoaded() { return isLoaded_; }
		bool detectMultiScale(Mat image,
				std::vector<Rect>& objects,
				double scaleFactor = 1.1,
				int minNeighbors = 3, int flags = 0,
				Size minSize = Size(),
				Size maxSize = Size() );

		enum {ENOSETWINDOW = -1,
			  EPREDICTION = 0};

	private:
		bool read(const FileNode& root);
		const Size getOriginalWindowSize() {
			return data.getOriginalWindowSize();
		}
		bool detectObjectsMultiScaleNoGrouping(Mat image,
				std::vector<Rect>& candidates, std::vector<int>& rejectLevels,
				std::vector<double>& levelWeights, double scaleFactor,
				Size minObjectSize, Size maxObjectSize,
				bool outputRejectLevels = false );

		int predictOrderedStump(Cascade & cascade, Ptr<feature_evaluator> &
				evaluator, double & sum);

		int runAt(Ptr<feature_evaluator>& feval, const Point & p, int scaleIdx,
				double& weight);

		Cascade_data data;

		Ptr<feature_evaluator> evaluator;
		//Ptr<MaskGenerator> maskGenerator;
		Mat facepos, stages, nodes, leaves, subsets;

		bool isLoaded_;

		Ptr<Classifier> classifier;

		Mutex mtx;
};




#endif /* end of include guard: CLASSIFIER_H */
