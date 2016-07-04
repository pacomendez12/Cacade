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
	const double GROUP_EPS = 0.2;
	if (detectObjectsMultiScaleNoGrouping(image, objects, rl, lw, scaleFactor,
			minSize, maxSize, orl)) {

		groupRectangles(objects, minNeighbors, GROUP_EPS);
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


	bool evaluatorResult = evaluator->setImage(image, scales);
	if (scales.size() == 0 || !evaluatorResult)
		return false;

	evaluator->getMats();

	{
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
		classifier->runClassifier(0, nstripes);
	}

	return true;
}

int itera = 0;
int
Cascade::predictOrderedStump(Cascade & cascade,
						Ptr<feature_evaluator> &_eval, double & sum) {
	if (data.getStumps().empty())
		return EPREDICTION;
	const Cascade_data::Stump * cascadeStumps = data.getStumps().data();
	const std::vector<Cascade_data::Stump> & stumps = data.getStumps();
	const std::vector<Cascade_data::Stage>& stages = data.getStages();
	feature_evaluator & eval = *_eval;


	int nstages = (int) data.getStages().size();
	double tmpData = 0;

//#pragma omp parallel for private(nstages)
	for (int idxStages = 0; idxStages < nstages; idxStages++) {
		tmpData = 0;
		const Cascade_data::Stage & stage = stages[idxStages];
		int ntrees = stage.ntrees;
		if (ntrees > 5000) {
			cout << "stage " << idxStages << ", ntrees = " << ntrees << endl;
			cascade.data.printStages();
			cout << "ERROR: ntrees is bad value" << endl;

		}

		for (int i = 0; i < ntrees; i++) {
			const Cascade_data::Stump & stump = cascadeStumps[i];
			if (stump.featureIdx >= data.getStumps().size()) {
				cout << " ERROR: stump.featureIdx = " << stump.featureIdx << endl;
				if (itera++ < 10)
					continue;
				else
					exit(0);
			}

			double value = eval(stump.featureIdx);
			tmpData += (value  < stump.threshold ? stump.left : stump.right);
		}

		if (tmpData < stage.threshold) {
			sum = tmpData;
			return -idxStages;
		}

		cascadeStumps += ntrees;
	}

	sum = tmpData;
	return 1;
}

//#define DEBUG_NO_DATA

int
Cascade::runAt(Ptr<feature_evaluator>& feval, const Point & p, int scaleIdx,
		double& weight) {

	if (!feval->setWindow(p, scaleIdx)) return ENOSETWINDOW;

	int result;
	if (data.getMaxNodesPerTree() == 1) {
		return predictOrderedStump(*this, feval, weight);
	} else
		cout << "Debug why here??" << endl;
		//return predictOrdered(*this, evaluator, weight);

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
	stripeSizes = _stripeSize;
	rectangles = &_vec;
	rejectedLevels = outputLevels ? &_levels : nullptr;
	levelWeights = outputLevels ? &_weights : nullptr;
	mtx = &_mtx;
}


void Classifier::runClassifier(int start, int end) {
#pragma omp parallel
{
	Ptr<feature_evaluator> evaluator = CascadePtr->evaluator->clone();
	double gypWeight = 0.0;
	Size origWinSize = CascadePtr->data.getOriginalWindowSize();

#pragma omp for schedule(dynamic)
	for (int sidx = 0; sidx < nscales; sidx++) {
		const feature_evaluator::ScaleData& s = scaleData[sidx];
		float scalingFactor = s.scale;
		int ystep = s.ystep;
		int stripeSize = stripeSizes[sidx];
		int y0 = start * stripeSize;
		Size sWin = s.getWorkingSize(origWinSize);
		int y1 = std::min(end * stripeSize, sWin.height);

		Size winSize(cvRound(origWinSize.width * scalingFactor),
				cvRound(origWinSize.height * scalingFactor));

		for (int y = y0; y < y1; y += ystep) {
			for (int x = 0; x < sWin.width; x += ystep) {
				int result = CascadePtr->runAt(evaluator, Point(x, y),
						sidx, gypWeight);
				if (rejectedLevels) {
					if (result == 1) {
						result = -(int) CascadePtr->data.getStages().size();
					}

					if (CascadePtr->data.getStages().size() + result == 0) {
						mtx->lock();
						{
							rectangles->push_back(Rect(
										cvRound(x * scalingFactor),
										cvRound(y * scalingFactor),
										winSize.width, winSize.height));

							rejectedLevels->push_back(-result);
							levelWeights->push_back(gypWeight);
						}
						mtx->unlock();
					}
				} else if (result > 0) {
					mtx->lock();
					{
						rectangles->push_back(Rect(
									cvRound(x * scalingFactor),
									cvRound(y * scalingFactor),
									winSize.width, winSize.height));
					}
					mtx->unlock();
				}

				if (result == 0)
					x += ystep;
			}
		}

	}
}
}



