#include <train/cascade_data.h>

#define DEBUG

Cascade_data::Cascade_data() {
	stageType = featureType = ncategories = minNodesPerTree =
		maxNodesPerTree = 0;

}


bool
Cascade_data::read(const FileNode &root) {

	static const float THRESHOLD_EPS = 1e-5f;

	String stage_type = (String) root[STAGE_TYPE];

	/* Checking if the type is boost */
	if (stage_type != BOOST_TYPE)
		return false;

	String feature_type = (String) root[FEATURE_TYPE];
	if (feature_type != HAAR_FEATURE_TYPE) {
		cout << "Error while reading training file, this is not a " <<
			" Haar like trainig" << endl;
		return false;
	}

	origWinSize.width = (int) root[WIDTH_WINDOW];
	origWinSize.height = (int) root[HEIGHT_WINDOW];

	if (origWinSize.width <= 0 || origWinSize.height <= 0) {
		cout << "Error: " << (origWinSize.width <= 0 ? WIDTH_WINDOW :
			HEIGHT_WINDOW) << " is not enough large" << endl;
		return false;
	}


	FileNode node = root[FEATURE_PARAMS];
	if (node.empty()) {
		cout << "Error: featureParams tag is not present" << endl;
		return false;
	}


	ncategories = node[MAX_CAT_COUNT];
	int subset_size = (ncategories + 31) / 32;
	int node_step = 3 + (ncategories > 0 ? subset_size : 1);


	node = root[STAGES];
	if (node.empty()) {
		cout << "Error: No stages defined" << endl;
		return false;
	}

	stages.reserve(node.size());
	classifiers.clear();
	nodes.clear();
	stumps.clear();


	FileNodeIterator it = node.begin(), end = node.end();
	minNodesPerTree = INT_MAX;
	maxNodesPerTree = 0;


	/* for each stage node */
	for (int i = 0; it != end; ++i, ++it) {
		FileNode node_stage = *it;
		Stage stage;
		stage.ntrees = (int) node_stage[MAX_WEAK_COUNT];
		stage.threshold = (float) node_stage[STAGE_THRESHOLD] - THRESHOLD_EPS;
		FileNode node_weak_classifier = node_stage[WEAK_CLASSIFIERS];

		if (node_weak_classifier.empty()) {
			cout << "There is no classifiers in the stage" << i << endl;
			return false;
		}

		//stage.ntrees = (int) node_weak_classifier.size();
		stage.first = (int) classifiers.size();
		stages.push_back(stage);
		classifiers.reserve(stages[i].first + stages[i].ntrees);

		FileNodeIterator it_weak = node_weak_classifier.begin();
		FileNodeIterator it_weak_end = node_weak_classifier.end();

		/* for each weak classifier */
		for ( ; it_weak != it_weak_end; ++it_weak) {
			FileNode node_weak = *it_weak;
			FileNode internal_nodes = node_weak[INTERNAL_NODES];
			FileNode leaf_values = node_weak[LEAF_VALUES];

			if (internal_nodes.empty() || leaf_values.empty()) {
				return false;
			}

			DTree tree;

			tree.nodeCount = (int) internal_nodes.size() / node_step;
			minNodesPerTree = std::min(minNodesPerTree, tree.nodeCount);
			maxNodesPerTree = std::max(maxNodesPerTree, tree.nodeCount);

			classifiers.push_back(tree);

			nodes.reserve(nodes.size() + tree.nodeCount);
			leaves.reserve(leaves.size() + leaf_values.size());
			if (subset_size > 0)
				subsets.reserve(subsets.size() + tree.nodeCount *
						subset_size);

			FileNodeIterator it_internal = internal_nodes.begin();
			FileNodeIterator it_internal_end = internal_nodes.end();

			/* for each internal node */
			for ( ; it_internal != it_internal_end; ) {
				DTreeNode tree_node;
				tree_node.left = (int) *it_internal++;
				tree_node.right = (int) *it_internal++;
				tree_node.featureIdx = (int) *it_internal++;
				if (subset_size > 0) {
					for (int j = 0; j < subset_size; j++)
						subsets.push_back(*it_internal);
					tree_node.threshold = 0.0;
				} else {
					tree_node.threshold = (float) *it_internal++;
				}

				nodes.push_back(tree_node);
			}


			it_internal = leaf_values.begin();
			it_internal_end = leaf_values.end();

			for ( ; it_internal != it_internal_end; ++it_internal)
				leaves.push_back((float) *it_internal);
		}
	}

	/*for (int i = 0; i < stages.size(); i++) {
		cout << " no weak = " << stages[i].ntrees << endl;
	}*/

	if (maxNodesPerTree == 1) {
		int nodeOfs = 0, leafOfs = 0;

		size_t nstages = stages.size();
		for (size_t stageIdx = 0; stageIdx < nstages; stageIdx++) {
			const Stage& stage = stages[stageIdx];
			int ntrees = stage.ntrees;

			for (int i = 0; i < ntrees; i++, nodeOfs++, leafOfs += 2) {
				const DTreeNode& node = nodes[nodeOfs];
				stumps.push_back(Stump(node.featureIdx, node.threshold,
							leaves[leafOfs], leaves[leafOfs + 1]));
			}
		}
	}

	return true;
}


