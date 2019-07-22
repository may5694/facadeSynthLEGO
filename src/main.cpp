#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <set>
#include <experimental/filesystem>
#include "building.hpp"
using namespace std;
namespace fs = experimental::filesystem;

// Program options
struct Options {
	fs::path rootDir;		// LEGO root directory
	fs::path clusterDir;	// Directory containing building clusters
	set<string> clusters;	// Which clusters to process
	bool debugOut;			// Whether to output debug info
	fs::path debugDir;		// Directory for debug output

	Options(): debugOut(false) {}
};

// Functions
Options readArgs(int argc, char** argv);

int main(int argc, char** argv) {
	try {
		// Read command line arguments
		Options opts = readArgs(argc, argv);

		// Read models json
		ModelInfo mi;
		readModeljson("model_config.json", mi);

		// Process each cluster
		for (auto cluster : opts.clusters) {
			fs::path inputMetaPath = opts.clusterDir / cluster / "Output" /
				("building_cluster_" + cluster + "__TexturedModelMetadata.json");
			fs::path outputMetaPath = opts.clusterDir / cluster / "Output" /
				("building_cluster_" + cluster + "__SyntheticFacadeTexturedModelsMetadata.json");
			fs::path debugPath;
			if (opts.debugOut)
				debugPath = opts.debugDir / cluster;

			cout << "Cluster: " << cluster << endl;

			genFacadeModel(inputMetaPath, outputMetaPath, mi, debugPath);
		}

	} catch (const exception& e) {
		cerr << e.what() << endl;
		return -1;
	}

	return 0;
}

// Read arguments from command line
Options readArgs(int argc, char** argv) {
	auto usage = [&]() -> string {
		stringstream ss;
		ss << "Usage: " << argv[0] << " root-dir [clusters] [-D]" << endl;
		return ss.str();
	};

	Options opts;

	// Process all arguments
	for (int a = 1; a < argc; a++) {
		// Debug output
		if (string(argv[a]) == "-D") {
			opts.debugOut = true;
			// Debug directory
			if (!opts.rootDir.empty())
				opts.debugDir = fs::path("output") / opts.rootDir.filename();

		// First non-option argument is the root directory
		} else if (opts.rootDir.empty()) {
			// Remove trailing separator
			string rootDirStr = argv[a];
			while (rootDirStr.back() == '/' || rootDirStr.back() == '\\')
				rootDirStr.pop_back();
			// Check root directory
			opts.rootDir = rootDirStr;
			if (!fs::exists(opts.rootDir))
				throw runtime_error("Could not find root directory " + opts.rootDir.string());
			if (!fs::is_directory(opts.rootDir))
				throw runtime_error("Not a directory: " + opts.rootDir.string());
			// Check cluster directory
			opts.clusterDir = opts.rootDir / "01700_MODELING" / "BuildingClusters";
			if (!fs::exists(opts.clusterDir))
				throw runtime_error("Could not find cluster directory " + opts.clusterDir.string());
			if (!fs::is_directory(opts.clusterDir))
				throw runtime_error("Not a directory: " + opts.clusterDir.string());
			// Set debug output dir if debug output set
			if (opts.debugOut)
				opts.debugDir = fs::path("output") / opts.rootDir.filename();

		// All remaining arguments are cluster IDs
		} else {
			if (fs::exists(opts.clusterDir / argv[a]))
				opts.clusters.insert(argv[a]);
			else {
				try {
					// Convert argument to integer, pad with 0's
					int ci = stoi(argv[a]);
					stringstream ss;
					ss << setw(4) << setfill('0') << ci;
					string ciStr = ss.str();
					// Add if directory exists
					if (fs::exists(opts.clusterDir / ciStr)) {
						opts.clusters.insert(ciStr);
					} else {
						cerr << "Cluster \"" << ciStr << "\" does not exist" << endl;
					}
				} catch (const exception& e) {
					cerr << "Bad cluster \"" << argv[a] << "\"" << endl;
				}
			}
		}
	}

	if (opts.rootDir.empty())
		throw runtime_error(usage());

	// If no clusters specified, scan for all clusters
	if (opts.clusters.empty()) {
		for (fs::directory_iterator di(opts.clusterDir), dend; di != dend; ++di)
			if (fs::is_directory(di->path()))
				opts.clusters.insert(di->path().filename().string());
	}

	// No clusters found
	if (opts.clusters.empty())
		throw runtime_error("No clusters found!");

	return opts;
}
