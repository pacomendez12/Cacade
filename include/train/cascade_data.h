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
#define MAX_WEAK_COUNT		"maxWeakCount"
#define LEAF_VALUES			"leafValues"
#define INTERNAL_NODES		"internalNodes"

#define HAAR_FEATURE_TYPE	"HAAR"

#define MAX_CAT_COUNT		"maxCatCount"

#define FEATURES			"features"


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

	inline Size& getOriginalWindowSize() { return origWinSize; }


	inline int getMaxNodesPerTree() { return maxNodesPerTree; }
	inline std::vector<Stage> getStages() { return stages; }
	inline std::vector<Stump>& getStumps() { return stumps; }

	inline void printStages() {

		for(int i = 0; i < stages.size(); i++) {
			std::cout << "stage[" << i << "] = { \n\t\t first = " <<
				stages[i].first << ", ntrees = " << stages[i].ntrees <<
				", threshold = " << stages[i].threshold << "\n}" << std::endl;
		}
	}

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


#endif /* end of include guard: CASCADE_DATA_H */
