#ifndef GROUP_RECTANGLES_H

#define GROUP_RECTANGLES_H

#include <vector>
#include "opencv2/objdetect.hpp"

using namespace cv;

class group_rectangles {
	public:
		group_rectangles() {}
		void groupRectangles(std::vector<Rect> & rects, int neighbors, double eps);

};


#endif /* end of include guard: GROUP_RECTANGLES_H */
