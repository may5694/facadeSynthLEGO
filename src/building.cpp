#include <fstream>
#include <sstream>
#include <iomanip>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include "building.hpp"
using namespace std;
namespace rj = rapidjson;

void Building::load(fs::path metaPath) {
	// Clear any existing contents
	clear();

	// Set paths
	metaDir = metaPath.parent_path();
	facadeModelDir = metaDir / "FacadeModel";
	facadeTextureDir = facadeModelDir / "Textures";

	// Load the manifest metadata
	fs::path modelPath, texPath, surfPath;
	readManifest(metaPath, modelPath, texPath, surfPath);
}

void Building::clear() {
	// Reset to default-constructed Building
	*this = Building();
}

// Read textured model paths and cluster ID from metadata manifest
void Building::readManifest(fs::path metaPath, fs::path& modelPath,
	fs::path& texPath, fs::path& surfPath) {

	// Open and read the manifest file
	ifstream metaFile(metaPath);
	rj::IStreamWrapper isw(metaFile);
	rj::Document meta;
	meta.ParseStream(isw);

	// Read paths for textured model files
	const rj::Value& manifest = meta["_items"]["mesh_manifest"]["_items"];
	for (rj::SizeType i = 0; i < manifest.Size(); i++) {
		fs::path p = manifest[i].GetString();
		if (p.extension().string() == ".obj")
			modelPath = metaDir / p;
		else if (p.extension().string() == ".png")
			texPath = metaDir / p;
		else if (p.extension().string() == ".surfaces")
			surfPath = metaDir / p;
	}

	// Read cluster ID, convert to string
	int clusterID = meta["_items"]["cluster_id"].GetInt();
	stringstream ss;
	ss << setw(4) << setfill('0') << clusterID;
	cluster = ss.str();
}

// Default values for FacadeInfo members
FacadeInfo::FacadeInfo() :
	rectXform(1.0),
	iRectXform(1.0),
	ground(false),
	roof(false),
	score(0.0),
	valid(false),
	grammar(0),
	conf({}),
	rows(0),
	cols(0),
	grouping(0),
	relativeWidth(0.0),
	relativeHeight(0.0),
	doors(0),
	relativeDWidth(0.0),
	relativeDHeight(0.0) {}
