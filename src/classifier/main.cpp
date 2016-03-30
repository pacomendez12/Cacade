#include <train/cascade_data.h>


int main(int argc, const char *argv[])
{
	Cascade cascade;
	string str = "data/training.xml";
	if (cascade.load_from_file(str))
		cout << "Training loaded correctly" << endl;
	else
		cout << "failed while loading training" << endl;
	return 0;
}
