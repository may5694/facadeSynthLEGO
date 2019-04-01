#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <set>
#include <experimental/filesystem>
using namespace std;
namespace fs = experimental::filesystem;

set<string> getClusters(int argc, char** argv, fs::path clusterDir);

int main(int argc, char** argv) {
	try {
		// Get directories
		fs::path rootDir("lego_root");
		fs::path clusterDir = rootDir / "01700_MODELING" / "BuildingClusters";

		// Get clusters from cmd args
		set<string> clusters = getClusters(argc, argv, clusterDir);

		// Print clusters
		for (auto cluster : clusters)
			cout << cluster << " ";
		cout << endl;

	} catch (const exception& e) {
		cerr << e.what() << endl;
		return -1;
	}

	return 0;
}


// Get clusters from cmd args
set<string> getClusters(int argc, char** argv, fs::path clusterDir) {
	set<string> clusters;
	if (argc > 1) {
		for (int c = 1; c < argc; c++) {
			// Add if argument string if cluster directory exists
			if (fs::exists(clusterDir / argv[c]))
				clusters.insert(argv[c]);
			else {
				try {
					// Convert argument to integer, pad with 0's
					int ci = stoi(argv[c]);
					stringstream ss;
					ss << setw(4) << setfill('0') << ci;
					string ciStr = ss.str();
					// Add if directory exists
					if (fs::exists(clusterDir / ciStr)) {
						clusters.insert(ciStr);
					} else {
						cerr << "Cluster \"" << ciStr << "\" does not exist" << endl;
					}
				} catch (const exception& e) {
					cerr << "Bad cluster \"" << argv[c] << "\"" << endl;
				}
			}
		}
	} else {
		// Scan directory for all clusters
		for (fs::directory_iterator di(clusterDir), dend; di != dend; ++di)
			if (fs::is_directory(di->path()))
				clusters.insert(di->path().filename().string());
	}
	// No clusters to process
	if (clusters.empty())
		throw runtime_error("No clusters found!");

	return clusters;
}
