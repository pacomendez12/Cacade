#include <classifier/feature_evaluator.h>
#include <util/util.h>
#include <math.h>
#include "opencv2/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

#define DEBUG

using namespace cv;
using namespace std;

/* util */

template<typename _Tp>
void copyVectorToUMat(const std::vector<_Tp>& v, UMat& um)
{
    if(v.empty())
        um.release();
    Mat(1, (int)(v.size()*sizeof(v[0])), CV_8U, (void*)&v[0]).copyTo(um);
}


feature_evaluator::feature_evaluator() {
	features = makePtr<std::vector<Feature>>();
	optfeatures = makePtr<std::vector<OptFeature>>();
	optfeatures_lbuf = makePtr<std::vector<OptFeature>>();

	localSize = Size(4, 2);
	lbufSize = Size(0, 0);
	nchannels = 0;
	tofs = 0;

}

feature_evaluator::~feature_evaluator() {}

bool
feature_evaluator::read(FileNode node, Size s) {
	origWinSize = s;
	localSize = Size(0, 0);
	lbufSize = Size(0, 0);

	if (scale_data.empty())
		scale_data = makePtr<std::vector<ScaleData> >();
	else
		scale_data->clear();

	size_t n = node.size();
	features->resize(n);
	FileNodeIterator it = node.begin();
	hasTiltedFeatures = false;
	sbufSize = Size();
	ufbuf.release();

	/* reading every feature from xml file */
	for(int i = 0; i < n; i++, it++) {
		if (!(*features)[i].read(*it)) {
			std::cout << " Error while reading a feature from file" <<
																	std::endl;
			return false;
		}

		if ((*features)[i].tilted)
			hasTiltedFeatures = true;
	}
#ifdef DEBUG
	std::cout << " hastTiltedFeatures = " << hasTiltedFeatures << std::endl;
#endif

	nchannels = hasTiltedFeatures? 3 : 2;
	/* don't use 2 pixels */
	normrect = Rect(1, 1, origWinSize.width - 2, origWinSize.height - 2);

	return true;
}


void
feature_evaluator::getMats() {

	if (!(sbufFlag & SBUF_VALID)) {
		usbuf.copyTo(sbuf);
		sbufFlag = SBUF_VALID;
	}
}

bool feature_evaluator::
updateScaleData(const Size& size, const std::vector<float>& scales) {

	if (scale_data.empty())
		scale_data = makePtr<std::vector<ScaleData>> ();

	size_t no_scales = scales.size();
	cout << __func__ << ": no_scales = " << no_scales << endl;
	bool recalculateOptFeatures;
	if (recalculateOptFeatures = (no_scales != getScaleData().size()))
		resizeScaleData(no_scales);

	int layer_dy = 0;
	Point layer_ofs(0, 0);
	Size prevBuffSize = sbufSize;
	sbufSize.width = std::max(sbufSize.width, (int) alignSize(std::round(
					size.width / scales[0]) + 31, 32));

	recalculateOptFeatures = recalculateOptFeatures ||
		sbufSize.width != prevBuffSize.width;

	const std::vector<ScaleData>& sc_dt = getScaleData();

	for (int i = 0; i < no_scales; i++) {
		//Error, we need the original data
		//feature_evaluator::ScaleData& sd = sc_dt.at(i);

		//FIXED
		feature_evaluator::ScaleData& sd = getScaleData(i);
		if (!recalculateOptFeatures && fabs(sd.scale - scales[i]) >
				FLT_EPSILON * 100 * scales[i])
			recalculateOptFeatures = true;

		float sc = scales[i];
		Size sz(std::round(size.width / sc), std::round(size.height / sc));
		sd.ystep = sc > 2 ? 1 : 2;
		sd.scale = sc;
		sd.szi = Size(sz.width + 1, sz.height + 1);

		if (0 == i)
			layer_dy = sd.szi.height;

		if (layer_ofs.x + sd.szi.width > sbufSize.width) {
			layer_ofs = Point(0, layer_ofs.y + layer_dy);
			layer_dy = sd.szi.height;
		}

		sd.layer_ofs = layer_ofs.y * sbufSize.width + layer_ofs.x;
		layer_ofs.x += sd.szi.width;
	}

	layer_ofs.y += layer_dy;
	sbufSize.height = std::max(sbufSize.height, layer_ofs.y);
	recalculateOptFeatures = recalculateOptFeatures || (sbufSize.height !=
			prevBuffSize.height);
	return recalculateOptFeatures;
}


bool
feature_evaluator::setImage(Mat& image, const std::vector<float>& scales) {
	Size img_size = image.size();

	bool recalculateFeatures = updateScaleData(img_size, scales);

	size_t no_scales = scale_data->size();

	if (no_scales > 0) {

		Size size_first  = getScaleData().at(0).szi;
		size_first = Size(std::max(rbuf.cols, (int) alignSize(size_first.width,
						16)), std::max(rbuf.rows, size_first.height));

		if (recalculateFeatures) {
			computeOptFeatures();
			copyVectorToUMat(*scale_data, uscaleData);
		}

		//if (image.isUMat() && localSize.area() > 0) {
			/* TODO: remove this part, I don't want to use UMat*/
		//} else {
			sbuf.create(sbufSize.height * nchannels, sbufSize.width, CV_32S);
			rbuf.create(size_first, CV_8U);

			for (int i = 0; i < no_scales; i++) {
				const ScaleData& s = scale_data->at(i);
				Mat dst(s.szi.height - 1, s.szi.width - 1, CV_8U, rbuf.ptr());
				cv::resize(image, dst, dst.size(), 1.0 / s.scale, 1.0 / s.scale,
						INTER_LINEAR);
				computeChannels((int) i, dst);
			}

			sbufFlag = SBUF_VALID;
		//}
		return true;
	}
	return false;
}


void
feature_evaluator::computeChannels(int scaleIdx, Mat img) {
		const ScaleData & s = getScaleData(scaleIdx);

		sqofs = hasTiltedFeatures ? sbufSize.area() * 2 : sbufSize.area();

		Mat sum(s.szi, CV_32S, sbuf.ptr<int>() + s.layer_ofs, sbuf.step);
		Mat sqsum(s.szi, CV_32S, sum.ptr<int>() + sqofs, sbuf.step);

		if (hasTiltedFeatures) {
			Mat tilted(s.szi, CV_32S, sum.ptr<int>() + tofs, sbuf.step);
			integral(img, sum, sqsum, tilted, CV_32S, CV_32S);
		} else {
			integral(img, sum, sqsum, noArray(), CV_32S, CV_32S);
		}
}


void
feature_evaluator::computeOptFeatures() {
	if (hasTiltedFeatures) {
		tofs = sbufSize.area();
	}

	int sstep = sbufSize.width;
	CV_SUM_OFS(nofs[0], nofs[1], nofs[2], nofs[3], 0, normrect, sstep);


	size_t nfeatures = features->size();
	const std::vector<Feature>& ff = *features;
	optfeatures->resize(nfeatures);
	optfeaturesPtr = &(*optfeatures)[0];

	for(size_t fi = 0; fi < nfeatures; fi++) {
		optfeaturesPtr[fi].setOffsets(ff[fi], sstep, tofs);
	}

	optfeatures_lbuf->resize(nfeatures);

	for (size_t fi = 0; fi < nfeatures; fi++) {
		optfeatures_lbuf->at(fi).setOffsets(ff[fi], lbufSize.width > 0 ?
				lbufSize.width : sstep, tofs);
	}

	copyVectorToUMat(*optfeatures_lbuf, ufbuf);
}


bool
feature_evaluator::setWindow(Point p, int scaleIdx) {
	const ScaleData& scale = getScaleData(scaleIdx);

	if (p.x < 0 || p.y < 0 || p.x + origWinSize.width >= scale.szi.width ||
			p.y + origWinSize.height >= scale.szi.height)
		return false;

	pwin = &sbuf.at<int>(p) + scale.layer_ofs;

	const int * pq = (const int *) (pwin + sqofs);
	int valsum = CALC_SUM_OFS(nofs, pwin);
	unsigned valsqsum = (unsigned) (CALC_SUM_OFS(nofs, pq));
	double area = normrect.area();
	double nf = area * valsum - (double) pow(valsum, 2);
	if (nf > 0.0) {
		nf = std::sqrt(nf);
		varianceNormFactor = (float) (1 / nf);
	} else {
		varianceNormFactor = 1.0;
		return false;
	}

	return true;
}

/************************* Feature methods  **************************/

Feature::Feature() {
	tilted = false;
	for (int i = 0; i < REC_MAX; i++) {
		rect[i].r = Rect();
		rect[i].weight = 0;
	}
}

bool
Feature::read(const FileNode& node) {
	FileNode rects_node = node[RECTS];
	FileNodeIterator it = rects_node.begin();
	FileNodeIterator it_end = rects_node.end();

	/* TODO: check if this is necesary */
	for (int i = 0; i < REC_MAX; i++) {
		rect[i].r = Rect();
		rect[i].weight = 0.0;
	}

	for (int i = 0; i < REC_MAX && it != it_end; i++, it++) {
		FileNodeIterator it_rect_val = (*it).begin();
		rect[i].r.x = *it_rect_val++;
		rect[i].r.y = *it_rect_val++;
		rect[i].r.width = *it_rect_val++;
		rect[i].r.height = *it_rect_val++;
		rect[i].weight = *it_rect_val;

#ifdef DEBUG
		std::cout << rect[i].r.x << ", " << rect[i].r.y << ", " <<
			rect[i].r.width << ", " << rect[i].r.height << ", " <<
			rect[i].weight << std::endl;
#endif
	}


	tilted = (int) node[TILTED] != 0;
	return true;
}


/* ************************* optFeatures methods *********************** */


inline OptFeature::OptFeature() {
    weight[0] = weight[1] = weight[2] = 0.f;

    ofs[0][0] = ofs[0][1] = ofs[0][2] = ofs[0][3] =
    ofs[1][0] = ofs[1][1] = ofs[1][2] = ofs[1][3] =
    ofs[2][0] = ofs[2][1] = ofs[2][2] = ofs[2][3] = 0;
}

inline float OptFeature::calc( const int* ptr ) const {
    float ret = weight[0] * CALC_SUM_OFS(ofs[0], ptr) +
                weight[1] * CALC_SUM_OFS(ofs[1], ptr);

    if( weight[2] != 0.0f )
        ret += weight[2] * CALC_SUM_OFS(ofs[2], ptr);

    return ret;
}

void OptFeature::setOffsets( const Feature& _f, int step, int _tofs )
{
    weight[0] = _f.rect[0].weight;
    weight[1] = _f.rect[1].weight;
    weight[2] = _f.rect[2].weight;

    if( _f.tilted )
    {
        CV_TILTED_OFS( ofs[0][0], ofs[0][1], ofs[0][2], ofs[0][3], _tofs,
				_f.rect[0].r, step );
        CV_TILTED_OFS( ofs[1][0], ofs[1][1], ofs[1][2], ofs[1][3], _tofs,
				_f.rect[1].r, step );
        CV_TILTED_OFS( ofs[2][0], ofs[2][1], ofs[2][2], ofs[2][3], _tofs,
				_f.rect[2].r, step );
    }
    else
    {
        CV_SUM_OFS( ofs[0][0], ofs[0][1], ofs[0][2], ofs[0][3], 0,
				_f.rect[0].r, step );
        CV_SUM_OFS( ofs[1][0], ofs[1][1], ofs[1][2], ofs[1][3], 0,
				_f.rect[1].r, step );
        CV_SUM_OFS( ofs[2][0], ofs[2][1], ofs[2][2], ofs[2][3], 0,
				_f.rect[2].r, step );
    }
}
