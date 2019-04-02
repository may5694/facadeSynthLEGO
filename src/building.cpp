#include <fstream>
#include <sstream>
#include <iomanip>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
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

	// Load the geometry
	readModel(modelPath);
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

// Read the .obj model and store the geometry
void Building::readModel(fs::path modelPath) {

	// Load the obj model
	tinyobj::attrib_t attrib;
	vector<tinyobj::shape_t> shapes;
	string objWarn, objErr;
	bool objLoaded = tinyobj::LoadObj(&attrib, &shapes, NULL, &objWarn, &objErr,
		modelPath.string().c_str(), NULL, true);
	// Check for errors
	if (!objLoaded) {
		stringstream ss;
		ss << "Failed to load " << modelPath.filename().string() << ":" << endl;
		ss << objErr;
		throw runtime_error(ss.str());
	}

	minBB_utm = glm::vec3(FLT_MAX);
	maxBB_utm = glm::vec3(-FLT_MAX);

	// Add faces to geometry buffers
	for (size_t s = 0; s < shapes.size(); s++) {
		size_t idx_offset = 0;
		for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
			int fv = shapes[s].mesh.num_face_vertices[f];

			// Add face to index buffer
			for (size_t v = 2; v < fv; v++) {
				indexBuf.push_back(posBuf.size());
				indexBuf.push_back(posBuf.size() + v - 1);
				indexBuf.push_back(posBuf.size() + v - 0);
			}
			// Add vertex attributes
			for (size_t v = 0; v < fv; v++) {
				tinyobj::index_t idx = shapes[s].mesh.indices[idx_offset + v];

				// Add position
				posBuf.push_back({
					attrib.vertices[3 * idx.vertex_index + 0],
					attrib.vertices[3 * idx.vertex_index + 1],
					attrib.vertices[3 * idx.vertex_index + 2]});

				// Add texture coordinate
				tcBuf.push_back({
					attrib.texcoords[2 * idx.texcoord_index + 0],
					attrib.texcoords[2 * idx.texcoord_index + 1]});

				// Update bounding box
				minBB_utm = glm::min(minBB_utm, posBuf.back());
				maxBB_utm = glm::max(maxBB_utm, posBuf.back());
			}
		}
	}
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
