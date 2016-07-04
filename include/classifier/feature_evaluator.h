#ifndef FEATURE_EVALUATOR_H

#define FEATURE_EVALUATOR_H

#include "opencv2/core.hpp"
#include <iostream>
#include <sstream>
#include <string>

using namespace cv;
using namespace std;

class Feature;
class OptFeature;

class feature_evaluator
{
public:
	feature_evaluator();
	~feature_evaluator();

    struct ScaleData
    {
        ScaleData() { scale = 0.f; layer_ofs = ystep = 0; }
        Size getWorkingSize(Size winSize) const
        {
            return Size(std::max(szi.width - winSize.width, 0),
                        std::max(szi.height - winSize.height, 0));
        }

        float scale;
        Size szi;
        int layer_ofs, ystep;
    };

	/* read features from xml file */
	bool read(FileNode node, Size s);
	ScaleData& getScaleData(int scaleIdx) const
    {
        CV_Assert( 0 <= scaleIdx && scaleIdx < (int)scale_data->size());
        return scale_data->at(scaleIdx);
    }

	float operator()(int featureIdx) const;


	bool setImage(Mat& image, const std::vector<float>& scales);
	bool updateScaleData(const Size& size, const std::vector<float>& scales);
	void computeChannels(int scaleIdx, Mat img);
	bool setWindow(Point p, int scaleIdx);

	inline Ptr<feature_evaluator> clone() const {
		Ptr<feature_evaluator> ev = makePtr<feature_evaluator>();
		*ev = *this;
		return ev;
	}

	void getMats();

	const std::vector<ScaleData>& getScaleData() { return *scale_data; }
	inline void resizeScaleData(size_t size) { scale_data->resize(size); }


private:

	enum sbuf_flag_t { SBUF_VALID = 1, USBUF_VALID = 2};

	sbuf_flag_t sbufFlag;
	/* Methods */
	void computeOptFeatures();


	/* data */
	Ptr<std::vector<ScaleData> > scale_data;

	Size origWinSize, sbufSize, localSize, lbufSize;
    int nchannels;
    Mat sbuf, rbuf;
    UMat urbuf, usbuf, ufbuf, uscaleData;

	/* haar data */
    Ptr<std::vector<Feature> > features;
	Ptr<std::vector<OptFeature> > optfeatures;
	Ptr<std::vector<OptFeature> > optfeatures_lbuf;
	bool hasTiltedFeatures;
	int tofs, sqofs;

    Vec4i nofs;
    Rect normrect;
    const int* pwin;
    OptFeature* optfeaturesPtr; // optimization
    float varianceNormFactor;
};


#define REC_MAX 3

class Feature {
public:
	Feature();
	bool read(const FileNode& node);
	bool tilted;

	struct {
		Rect r;
		float weight;
	} rect[REC_MAX];

};


class OptFeature {
public:
	OptFeature();

	float calc( const int* pwin ) const;
	void setOffsets( const Feature& _f, int step, int tofs );

	inline std::string toString() {
		std::stringstream ss;
		ss << "{ ofs = [";
		for (int i = 0; i < REC_MAX; i++) {
			ss << "\n\t\t";
			for (int j = 0; j < 4; j++)
				ss << ofs[i][j] << ", ";
		}
		ss << "]\n\nweight = [";

		for (int i = 0; i < 4; i++)
			ss << weight[i] << ", ";
		ss << "]" << endl;
		return ss.str();
	}

	/* data */
	int ofs[REC_MAX][4];
	float weight[4];
	int test;
};


/* Tags */

#define RECTS				"rects"
#define TILTED				"tilted"


#endif /* end of include guard: FEATURE_EVALUATOR_H */
