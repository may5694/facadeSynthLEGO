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

	// Load the texture atlas
	atlasImg = cv::imread(texPath.string(), CV_LOAD_IMAGE_UNCHANGED);

	// Read surface groupings
	readSurfaces(surfPath);
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

			idx_offset += fv;
		}
	}
}

// Read surface groups and get facade info
void Building::readSurfaces(fs::path surfPath) {
	// Open surfaces file
	ifstream surfFile;
	surfFile.exceptions(ios::eofbit | ios::badbit | ios::failbit);
	surfFile.open(surfPath);

	// Loop over all faces
	for (size_t f = 0; f < indexBuf.size() / 3; f++) {
		// Read the surface group this face belongs to
		int fi;
		surfFile >> fi;

		// Add this face to the facade it belongs to
		facadeInfo[fi].faceIDs.push_back(f);
	}

	// Loop over facades
	for (auto& fi : facadeInfo) {
		FacadeInfo& fa = fi.second;
		// Get spatial extents
		glm::vec2 minBB_uv(FLT_MAX), maxBB_uv(-FLT_MAX);
		fa.zBB_utm = glm::vec2(FLT_MAX, -FLT_MAX);
		float wt = 0.0;
		glm::vec3 avgNorm(0.0);
		for (auto f : fa.faceIDs) {
			vector<glm::vec3> verts;
			for (int v = 0; v < 3; v++) {
				glm::vec3 p = posBuf[indexBuf[3 * f + v]];
				verts.push_back(p);
				fa.zBB_utm.x = glm::min(fa.zBB_utm.x, p.z);
				fa.zBB_utm.y = glm::max(fa.zBB_utm.y, p.z);

				glm::vec2 t = tcBuf[indexBuf[3 * f + v]];
				minBB_uv = glm::min(minBB_uv, t);
				maxBB_uv = glm::max(maxBB_uv, t);
			}

			// Get normal weighted by area
			glm::vec3 e1 = verts[1] - verts[0];
			glm::vec3 e2 = verts[2] - verts[0];
			glm::vec3 n = glm::normalize(glm::cross(e1, e2));
			float w = glm::length(glm::cross(e1, e2));
			avgNorm += n * w;
			wt += w;
		}
		// Get average normal
		fa.normal = glm::normalize(avgNorm / wt);
		fa.ground = (fa.zBB_utm.x - minBB_utm.z < 1e-3);
		fa.roof = (glm::dot(fa.normal, { 0.0, 0.0, 1.0 }) > 0.707f);

		// Get atlas UV bounding box
		fa.atlasBB_uv.x = minBB_uv.x;
		fa.atlasBB_uv.y = minBB_uv.y;
		fa.atlasBB_uv.width = maxBB_uv.x - minBB_uv.x;
		fa.atlasBB_uv.height = maxBB_uv.y - minBB_uv.y;
		// Get atlas PX bounding box
		fa.atlasBB_px.x = floor(fa.atlasBB_uv.x * atlasImg.cols);
		fa.atlasBB_px.y = floor((1.0 - maxBB_uv.y) * atlasImg.rows);
		fa.atlasBB_px.width = ceil(fa.atlasBB_uv.width * atlasImg.cols);
		fa.atlasBB_px.height = ceil(fa.atlasBB_uv.height * atlasImg.rows);
		// Get ROI of facade from atlas image
		fa.facadeImg = atlasImg(fa.atlasBB_px);

		// Get orientation matrix
		fa.rectXform = glm::mat4(1.0);
		if (glm::dot(fa.normal, { 0.0, 0.0, 1.0 }) < 1.0) {
			glm::vec3 up(0.0, 0.0, 1.0);
			glm::vec3 right = glm::normalize(glm::cross(up, fa.normal));
			up = glm::normalize(glm::cross(fa.normal, right));

			fa.rectXform[0] = glm::vec4(right, 0.0f);
			fa.rectXform[1] = glm::vec4(up, 0.0f);
			fa.rectXform[2] = glm::vec4(fa.normal, 0.0f);
			fa.rectXform = glm::transpose(fa.rectXform);
		}

		// Get the rotated facade offset and size
		glm::vec3 minBB_rutm(FLT_MAX);
		fa.size_utm = glm::vec2(-FLT_MAX);
		for (auto f : fa.faceIDs) {
			for (int vi = 0; vi < 3; vi++) {
				glm::vec3 v = posBuf[indexBuf[3 * f + vi]];
				minBB_rutm = glm::min(minBB_rutm, glm::vec3(fa.rectXform * glm::vec4(v, 1.0)));
				fa.size_utm = glm::max(fa.size_utm, glm::vec2(fa.rectXform * glm::vec4(v, 1.0)));
			}
		}
		fa.size_utm -= glm::vec2(minBB_rutm);
		fa.rectXform[3] = glm::vec4(-minBB_rutm, 1.0);
		fa.iRectXform = glm::inverse(fa.rectXform);
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
