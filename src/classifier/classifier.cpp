#include "opencv2/imgproc.hpp"
#include <classifier/classifier.h>
#include <vector>
#include <math.h>
#include <iostream>

using namespace cv;

Cascade::Cascade() {

	evaluator = makePtr<feature_evaluator>();
	isLoaded_ = false;
	classifier = makePtr<Classifier>();
}

Cascade::~Cascade() {}

bool Cascade::load_from_file(const std::string filename)
{
	data = Cascade_data();

	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened()) {
		return false;
	}

	isLoaded_ = read(fs.getFirstTopLevelNode());
	return isLoaded_;
}


bool Cascade::read(const FileNode& root)
{
	stages.release();
	nodes.release();
	leaves.release();
	if (data.read(root)) {
		FileNode fn_features = root[FEATURES];
		return fn_features.empty()? false :
			evaluator->read(fn_features, data.getOriginalWindowSize());
	}
	return false;
}

bool
Cascade::detectMultiScale(Mat image, std::vector<Rect>& objects,
		double scaleFactor, int minNeighbors, int flags,
		Size minSize, Size maxSize) {

	if (!isLoaded())
		return false;


	/* convert image to grayscale */
	if (image.channels() > 1 || image.depth() != CV_8U) {
		cvtColor(image, image, COLOR_BGR2GRAY);
	}

	std::vector<int> rl;
	std::vector<double> lw;
	bool orl = false;
	if (detectObjectsMultiScaleNoGrouping(image, objects, rl, lw, scaleFactor,
			minSize, maxSize, orl)) {

		//TODO: group rectangles
		return true;
	}

	std::cout << "detectObjectsMultiScaleNoGrouping failed" << std::endl;
	return false;
}

bool
Cascade::detectObjectsMultiScaleNoGrouping(Mat image,
		std::vector<Rect>& candidates, std::vector<int>& rejectLevels,
		std::vector<double>& levelWeights, double scaleFactor,
		Size minObjectSize, Size maxObjectSize,
		bool outputRejectLevels) {

	Size img_size = image.size();
	candidates.clear();
	rejectLevels.clear();
	levelWeights.clear();

	/* TODO: check if this is really working */
	if (maxObjectSize.height == 0 || maxObjectSize.width == 0 ||
			maxObjectSize.height > img_size.height ||
			maxObjectSize.width > img_size.width)
		maxObjectSize = img_size;

	std::vector<float> scales;
	scales.reserve(1024);

	for (double factor = 1; ; factor *= scaleFactor) {
		Size originalWindowSize = getOriginalWindowSize();
		//cout << __func__ << ":  inside for, originalWindowSize = " <<
		//	originalWindowSize << " and factor = " << factor << endl;
		Size winSize(cvRound(originalWindowSize.width * factor),
				cvRound(originalWindowSize.height * factor));

		if (winSize.height > maxObjectSize.height ||
				winSize.width > maxObjectSize.width)
			/* stop increasing scaleFactor */
			break;
		if (winSize.height < minObjectSize.height ||
				winSize.width < minObjectSize.width)
			/* ignore this scale because is lower than minimun window size */
			continue;

		scales.push_back(factor);
	}


	cout << __func__ << ": scales.size() = " << scales.size() << endl;

	bool evaluatorResult = evaluator->setImage(image, scales);
	cout << "evaluator->setImage returned " << evaluatorResult << endl;
	if (scales.size() == 0 || !evaluatorResult)
		return false;

	evaluator->getMats();

	{ /* private scope */
		//Mat currentMask;
		/*if(maskGenerator) {
			currentMask = maskGenerator->generateMask(image);
		}*/

		size_t nscales = scales.size();
		cv::AutoBuffer<int> stripeSizeBuf(nscales);
		int * stripeSizes = stripeSizeBuf;

		const feature_evaluator::ScaleData * s = &evaluator->getScaleData(0);
		Size winSize = s->getWorkingSize(data.getOriginalWindowSize());
		int nstripes = cvCeil(winSize.width / 32.0);

		for	(size_t i = 0; i < nscales; i++) {
			winSize = s[i].getWorkingSize(data.getOriginalWindowSize());
			stripeSizes[i] = std::max((winSize.height / s[i].ystep +
						nstripes - 1) / nstripes, 1) * s[i].ystep;

		}

		classifier->setDataClassifier(this, (int)nscales, nstripes, s,
				stripeSizes, candidates, rejectLevels, levelWeights,
				outputRejectLevels, mtx);
		/*TODO: finish it */
	}

	return true;
}



int
Cascade::runAt(Ptr<feature_evaluator>& feval, Point p, int scaleIdx,
		double& weight) {
	/* TODO: this method */
}

/* ************************ Classifier methods **************************** */
void
Classifier::setDataClassifier(Cascade * cptr, int _nscales, int _nstripes,
		const feature_evaluator::ScaleData * _scaleData, const int * _stripeSize,
		std::vector<Rect>& _vec, std::vector<int>& _levels,
		std::vector<double>& _weights, bool outputLevels, Mutex& _mtx) {
	CascadePtr = cptr;
	nscales = _nscales;
	nstripes = _nstripes;
	scaleData = _scaleData;
	stripeSize = _stripeSize;
	Rectangles = &_vec;
	rejectedLevels = outputLevels ? &_levels : nullptr;
	levelWeights = outputLevels ? &_weights : nullptr;
	mtx = &_mtx;
}


void Classifier::runClassifier(int start, int end) {

/* TODO: implement it */

}
