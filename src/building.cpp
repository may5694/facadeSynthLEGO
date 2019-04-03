#include <fstream>
#include <sstream>
#include <iomanip>
#include <stack>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include "building.hpp"
#include "dn_predict.hpp"
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

// Calculate a score for each facade
void Building::scoreFacades() {
	// Loop over all facades
	for (auto& fi : facadeInfo) {
		FacadeInfo& finfo = fi.second;

		// Convert facade image to 32F
		cv::Mat bgraImg;
		finfo.facadeImg.convertTo(bgraImg, CV_32F, 1.0 / 255.0);

		// Separate into BGR and A
		cv::Mat bgrImg(bgraImg.size(), CV_32FC3), aImg(bgraImg.size(), CV_32FC1);
		cv::mixChannels(vector<cv::Mat>{ bgraImg }, vector<cv::Mat>{ bgrImg, aImg },
			{ 0, 0, 1, 1, 2, 2, 3, 3 });
		cv::Mat aMask = (aImg > 0.5);

		// Convert to HSV space
		cv::Mat hsvImg;
		cv::cvtColor(bgrImg, hsvImg, cv::COLOR_BGR2HSV);
		cv::Mat hImg(hsvImg.size(), CV_32FC1), vImg(hsvImg.size(), CV_32FC1);
		cv::mixChannels(vector<cv::Mat>{ hsvImg }, vector<cv::Mat>{ hImg, vImg },
			{ 0, 0, 2, 1 });

		cv::Size ks(7, 7);
		// Calculate shadows
		cv::Mat hShadow = hImg.clone();
		hShadow.forEach<float>([](float& p, const int* position) -> void {
			p = pow(cos((p / 360.0 - 0.6) * 2.0 * M_PI) * 0.5 + 0.5, 200.0);
		});
		cv::boxFilter(hShadow, hShadow, -1, ks);
		cv::Mat vShadow = vImg.clone();
		vShadow.forEach<float>([](float& p, const int* position) -> void {
			p = pow(max(0.25 - p, 0.0) / 0.25, 0.5);
		});
		cv::boxFilter(vShadow, vShadow, -1, ks);
		cv::Mat inShadow = hShadow.mul(vShadow).mul(aImg);

		// Calculate brightness
		cv::Mat vBright = vImg.clone();
		vBright.forEach<float>([](float& p, const int* position) -> void {
			p = min(p * 2.0, 1.0);
		});
		cv::boxFilter(vBright, vBright, -1, ks);

		// Calculate score
		float w1 = 0.35;		// Shadow
		float w2 = 0.35;		// Brightness
		float w3 = 0.30;		// Area
		cv::Mat score = aImg.mul(w1 * (1.0 - inShadow) + w2 * vBright + w3);
		finfo.score = cv::mean(score, aMask)[0];
	}
}

// Estimate facade parameters for each facade
void Building::estimParams(fs::path configPath) {
	// Loop over all facades
	for (auto& fi : facadeInfo) {
		// Skip roofs and very very small facades
		if (fi.second.roof || fi.second.inscRect_px.width < 2 || fi.second.inscRect_px.height < 2)
			continue;

		// Predict facade parameters
		dn_predict(fi.second, configPath.string());
	}
}

// Generate synthetic facade geometry and save it
void Building::synthFacades() {
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

	bool debugFacades = false;
	fs::path debugDir = fs::path("debug") / cluster;
	if (debugFacades) {
		if (!fs::exists(debugDir))
			fs::create_directories(debugDir);
	}

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

		fa.inscRect_px = findLargestRectangle(fa.facadeImg);
		fa.inscGround = (fa.ground &&
			(fa.inscRect_px.y + fa.inscRect_px.height == fa.atlasBB_px.height));

		if (debugFacades) {
			stringstream ss;
			ss << setw(4) << setfill('0') << fi.first;
			fs::path debugPath = debugDir / (ss.str() + ".png");
			cv::imwrite(debugPath.string(), fa.facadeImg);
		}

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
		fa.inscSize_utm.x = fa.inscRect_px.width * fa.size_utm.x / fa.atlasBB_px.width;
		fa.inscSize_utm.y = fa.inscRect_px.height * fa.size_utm.y / fa.atlasBB_px.height;
	}
}

// Returns the largest rectangle inscribed within regions of all non-zero alpha-ch pixels
cv::Rect Building::findLargestRectangle(cv::Mat img) {
	// Extract alpha channel
	cv::Mat aImg(img.size(), CV_8UC1);
	cv::mixChannels(vector<cv::Mat>{ img }, vector<cv::Mat>{ aImg }, { 3, 0 });
	cv::Mat mask = (aImg > 0) / 255;
	mask.convertTo(mask, CV_16S);

	// Get the largest area rectangle under a histogram
	auto maxHist = [](cv::Mat hist) -> cv::Rect {
		// Append -1 to both ends
		cv::copyMakeBorder(hist, hist, 0, 0, 1, 1, cv::BORDER_CONSTANT, cv::Scalar::all(-1));
		cv::Rect maxRect(-1, -1, 0, 0);

		// Initialize stack to first element
		stack<int> colStack;
		colStack.push(0);

		// Iterate over columns
		for (int c = 0; c < hist.cols; c++) {
			// Ensure stack is only increasing
			while (hist.at<int16_t>(c) < hist.at<int16_t>(colStack.top())) {
				// Pop larger element
				int h = hist.at<int16_t>(colStack.top()); colStack.pop();
				// Get largest rect at popped height using nearest smaller element on both sides
				cv::Rect rect(colStack.top(), 0, c - colStack.top() - 1, h);
				// Update best rect
				if (rect.area() > maxRect.area())
					maxRect = rect;
			}
			// Push this column
			colStack.push(c);
		}
		return maxRect;
	};

	cv::Rect maxRect(-1, -1, 0, 0);
	cv::Mat height = cv::Mat::zeros(1, mask.cols, CV_16SC1);
	for (int r = 0; r < mask.rows; r++) {
		// Extract a single row
		cv::Mat row = mask.row(r);
		// Get height of unbroken non-zero values per column
		height = (height + row);
		height.setTo(0, row == 0);

		// Get largest rectangle from this row up
		cv::Rect rect = maxHist(height);
		if (rect.area() > maxRect.area()) {
			maxRect = rect;
			maxRect.y = r - maxRect.height + 1;
		}
	}

	return maxRect;
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
