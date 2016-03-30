/* This file has been based on cascadedetect.hpp from opencv cascade
 * algorithm, this is because I should use the same structure from the
 * xml file
 * */

#ifndef CASCADE_DATA_H

#define CASCADE_DATA_H

#include <string>
#include <vector>
#include <iostream>
#include "opencv2/core.hpp"

using namespace cv;
using namespace std;



#define STAGE_TYPE			"stageType"
#define FEATURE_TYPE		"featureType"
#define FEATURE_PARAMS		"featureParams"
#define STAGES				"stages"

#define WIDTH_WINDOW		"width"
#define HEIGHT_WINDOW		"height"


#define BOOST_TYPE			"BOOST"
#define STAGE_THRESHOLD		"stageThreshold"
#define WEAK_CLASSIFIERS	"weakClassifiers"
#define LEAF_VALUES			"leafValues"
#define INTERNAL_NODES		"internalNodes"

#define HAAR_FEATURE_TYPE	"HAAR"

#define MAX_CAT_COUNT		"maxCatCount"




class Cascade_data {

	public:

	Cascade_data();

	struct DTreeNode
	{
		int featureIdx;
		float threshold; // for ordered features only
		int left;
		int right;
	};

	struct DTree
	{
		int nodeCount;
	};

	struct Stage
	{
		int first;
		int ntrees;
		float threshold;
	};

	struct Stump
	{
		Stump() { }
		Stump(int _featureIdx, float _threshold, float _left, float _right)
			: featureIdx(_featureIdx), threshold(_threshold),
			left(_left), right(_right) {}

		int featureIdx;
		float threshold;
		float left;
		float right;
	};


	bool read(const FileNode &root);

	private:

	int stageType;
	int featureType;
	int ncategories;
	int minNodesPerTree, maxNodesPerTree;
	Size origWinSize;

	std::vector<Stage> stages;
	std::vector<DTree> classifiers;
	std::vector<DTreeNode> nodes;
	std::vector<float> leaves;
	std::vector<int> subsets;
	std::vector<Stump> stumps;


};

class Cascade {
	public:
		bool load_from_file(const std::string file);


	private:
		bool read(const FileNode& root);

		Cascade_data data;


};

#endif /* end of include guard: CASCADE_DATA_H */
