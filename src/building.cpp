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
	// Create output directories
	if (!fs::exists(facadeModelDir))
		fs::create_directory(facadeModelDir);
	if (!fs::exists(facadeTextureDir))
		fs::create_directory(facadeTextureDir);

	// Get paths to output files
	fs::path objPath = facadeModelDir /
		("building_cluster_" + cluster + "__" + model + "__output_mesh_facade.obj");
	fs::path mtlPath = objPath; mtlPath.replace_extension(".mtl");
	fs::path texPath = facadeTextureDir / ("cluster-00" + cluster + ".png");
	// Path to texture relative to model
	fs::path relTexPath; auto tp = texPath.begin();
	for (auto mp = facadeModelDir.begin(); mp != facadeModelDir.end(); ++mp, ++tp);
	for (; tp != texPath.end(); ++tp) relTexPath /= *tp;

	// Create OBJ and MTL files
	ofstream objFile(objPath);
	ofstream mtlFile(mtlPath);
	int vcount = 0;			// Number of vertices written
	int tcount = 0;			// Number of texcoords written
	float recess = 0.0;		// Amount to recess windows and doors into the building
	float g34_border = 2.0;	// Border above and below vertical windows

	// Create synthetic atlas texture
	cv::Mat synthAtlasImg = cv::Mat::zeros(atlasImg.size(), atlasImg.type());

	// Add material for texture-mapped roofs
	mtlFile << "newmtl atlas" << endl;
	mtlFile << "Ka 0.00000000 0.00000000 0.00000000" << endl;
	mtlFile << "Kd 1.00000000 1.00000000 1.00000000" << endl;
	mtlFile << "Ks 0.00000000 0.00000000 0.00000000" << endl;
	mtlFile << "illum 2" << endl;
	mtlFile << "Ns 0.00000000" << endl;
	mtlFile << "map_Kd " << relTexPath.string() << endl;

	objFile << "mtllib " << mtlPath.filename().string() << endl;

	// Vertical sections in which windows are placed
	struct WinSection {
		glm::vec2 minBB;	// Minimum rectified 2D coords of section
		glm::vec2 maxBB;	// Maximum rectified 2D coords of section
		int rows;			// Number of rows of windows
		int cols;			// Number of columns of windows
		float xscale;		// Horizontal scale factor for window spacing
		float yoffset;		// Vertical offset to align rows
		WinSection(): minBB(0.0, FLT_MAX), maxBB(0.0, -FLT_MAX) {}
	};
	map<size_t, vector<WinSection>> facadeSections;

	// "Fuzzy" height comparator
	auto heightCmp = [](const float& a, const float& b) -> bool {
		static const float eps = 1e-3;
		if (abs(a - b) > eps) return a < b;
		return false;
	};
	// Group facades with similar parameters together
	struct FacadeGroup {
		vector<size_t> facades;		// Which facades are in the group
		vector<float> grammars;		// Which grammars are represented and by how much
		set<float, decltype(heightCmp)> sheights;	// Heights of winsections in this group
		float yoffset;				// Y offset for this group

		bool valid;					// Whether group has valid parameters
		int grammar;				// The selected grammar

		float avgRowsPerMeter;		// Average parameter values
		float avgColsPerMeter;
		float avgRelWidth;
		float avgRelHeight;
		bool hasDoors;
		float avgDoorsPerMeter;
		float avgRelDWidth;
		float avgDHeight;			// Not relative D height; chip size differs

		FacadeGroup(decltype(heightCmp) cmp) : grammars(7, 0.0), sheights(cmp), yoffset(0.0),
			valid(false), grammar(0), avgRowsPerMeter(0.0), avgColsPerMeter(0.0),
			avgRelWidth(0.0), avgRelHeight(0.0), hasDoors(false), avgDoorsPerMeter(0.0),
			avgRelDWidth(0.0), avgDHeight(0.0) {}
	};
	vector<FacadeGroup> facadeGroups;
	map<int, int> whichGroup;		// Maps facade ID to group ID

	for (auto& fi : facadeInfo) {
		auto& fp = fi.second;
		// Skip roofs
		if (fp.roof) continue;

		// Brighten any dark facades
		float brt = glm::dot(fp.bg_color, glm::vec3(1.0f)) / 3.0f;
		if (brt < 0.3f) {
			if (brt > 0.0f) fp.bg_color *= 0.5f / brt;
			// Reduce saturation
			float avg = glm::dot(fp.bg_color, glm::vec3(1.0f)) / 3.0f;
			fp.bg_color += (glm::vec3(avg) - fp.bg_color) * 0.8f;

			fp.bg_color = glm::clamp(fp.bg_color, 0.0f, 1.0f);
		}
		fp.win_color = fp.bg_color * 0.6f;

		// Get section boundaries along X
		auto sepCmp = [](const float& a, const float& b) -> bool {
			static const float eps = 1e-2;
			if (abs(a - b) > eps) return a < b;
			return false;
		};
		set<float, decltype(sepCmp)> xsep(sepCmp);
		for (auto f : fp.faceIDs) {
			for (int vi = 0; vi < 3; vi++) {
				glm::vec3 v = posBuf[indexBuf[3 * f + vi]];
				xsep.insert((fp.rectXform * glm::vec4(v, 1.0)).x);
			}
		}

		// Separate facade into window sections
		auto& winSections = facadeSections[fi.first];
		for (auto xi = xsep.begin(); xi != xsep.end(); ++xi) {
			if (!winSections.empty())
				winSections.back().maxBB.x = *xi;
			auto xiNext = xi; ++xiNext;
			if (xiNext != xsep.end()) {
				winSections.push_back({});
				winSections.back().minBB.x = *xi;
			}
		}

		// Get vertical bounds of each section
		for (auto f : fp.faceIDs) {
			// Get triangle bbox
			glm::vec2 minTBB(FLT_MAX);
			glm::vec2 maxTBB(-FLT_MAX);
			for (int vi = 0; vi < 3; vi++) {
				glm::vec3 v = posBuf[indexBuf[3 * f + vi]];
				minTBB = glm::min(minTBB, glm::vec2(fp.rectXform * glm::vec4(v, 1.0)));
				maxTBB = glm::max(maxTBB, glm::vec2(fp.rectXform * glm::vec4(v, 1.0)));
			}

			// Intersect with all sections
			for (auto& s : winSections) {
				if (minTBB.x + 1e-2 < s.maxBB.x && maxTBB.x - 1e-2 > s.minBB.x) {
					s.minBB.y = min(s.minBB.y, minTBB.y);
					s.maxBB.y = max(s.maxBB.y, maxTBB.y);
				}
			}
		}

		// Combine adjacent sections of equal vertical bounds
		for (auto si = winSections.begin(); si != winSections.end();) {
			auto siNext = si; siNext++;
			if (siNext == winSections.end()) break;
			if (abs(si->minBB.y - siNext->minBB.y) < 1e-4 &&
				abs(si->maxBB.y - siNext->maxBB.y) < 1e-4) {
				si->maxBB.x = siNext->maxBB.x;
				winSections.erase(siNext);
			} else
				++si;
		}

		// Get tallest sections
		vector<float> tallestSections;
		for (auto& s : winSections)
			tallestSections.push_back(s.maxBB.y - s.minBB.y);
		sort(tallestSections.begin(), tallestSections.end(), [](float a, float b) -> bool {
			return b < a; });

		// Limit to two tallest sections
		if (tallestSections.size() >= 2) {
			tallestSections.resize(2);
			// Remove smaller height if much smaller than tallest
			if (tallestSections[1] / tallestSections[0] < 0.8)
				tallestSections.pop_back();
		}

		// Find a group for this facade
		bool inGroup = false;
		for (int g = 0; g < facadeGroups.size(); g++) {
			FacadeGroup& fg = facadeGroups[g];

			// Find a matching sections height
			bool foundSection = false;
			for (auto ts : tallestSections) {
				if (fg.sheights.count(ts))
					foundSection = true;
			}

			// Add this facade and its grammar and heights to the group
			if (foundSection) {
				fg.facades.push_back(fi.first);

				if (fp.valid)
					fg.grammars[fp.grammar] = max(fg.grammars[fp.grammar], fp.score);
				else
					fg.grammars[0] = max(fg.grammars[0], fp.score);

				for (auto ts : tallestSections)
					fg.sheights.insert(ts);

				whichGroup[fi.first] = g;
				inGroup = true;
				break;
			}
		}
		// If no group matched, add a new group
		if (!inGroup) {
			FacadeGroup fg(heightCmp);

			fg.facades.push_back(fi.first);

			if (fp.valid)
				fg.grammars[fp.grammar] += fp.score;
			else
				fg.grammars[0] += fp.score;

			for (auto ts : tallestSections)
				fg.sheights.insert(ts);

			whichGroup[fi.first] = facadeGroups.size();
			facadeGroups.push_back(fg);
		}
	}		// for (auto& fi : facadeInfo) {

	// Determine group grammar and parameters
	for (auto& fg : facadeGroups) {
		// Get the grammar with the highest total score
		float maxG = 0.1;
		fg.grammar = 0;
		for (int g = 1; g < fg.grammars.size(); g++)
			if (fg.grammars[g] > maxG) {
				maxG = fg.grammars[g];
				fg.grammar = g;
			}
		// If any in this group have doors, add doors
		if (fg.grammar == 1 || fg.grammar == 3 || fg.grammar == 5)
			if (fg.grammars[2] > 0 || fg.grammars[4] > 0 || fg.grammars[6] > 0)
				fg.grammar++;
		fg.hasDoors = (fg.grammar == 2 || fg.grammar == 4 || fg.grammar == 6);

		// Group is valid if frammar is non-zero
		fg.valid = (fg.grammar > 0);
		// Set facades to valid or invalid
		for (auto fi : fg.facades)
			facadeInfo[fi].valid = fg.valid;

		if (!fg.valid) continue;

		// Average out the parameter values for selected grammar
		int sz = 0;
		int szd = 0;
		for (auto fi : fg.facades) {
			auto& fp = facadeInfo[fi];
			// Average window params with compatible estimations within this group
			if (fp.grammar && (fp.grammar - 1) / 2 == (fg.grammar - 1) / 2) {
				sz++;
				// Adjust number of rows if there's doors
				if (fp.grammar == 2 || fp.grammar == 4 || fp.grammar == 6)
					fg.avgRowsPerMeter += fp.rows / (fp.chip_size.y * (1.0 - fp.relativeDHeight));
				else
					fg.avgRowsPerMeter += fp.rows / fp.chip_size.y;
				fg.avgColsPerMeter += fp.cols / fp.chip_size.x;
				fg.avgRelWidth += fp.relativeWidth;
				fg.avgRelHeight += fp.relativeHeight;
			}
			// Average door params over any facades with doors
			if (fp.grammar == 2 || fp.grammar == 4 || fp.grammar == 6) {
				szd++;
				fg.avgDoorsPerMeter += fp.doors / fp.chip_size.x;
				fg.avgRelDWidth += fp.relativeDWidth;
				fg.avgDHeight += fp.relativeDHeight * fp.chip_size.y;
			}
		}
		fg.avgRowsPerMeter /= sz;
		fg.avgColsPerMeter /= sz;
		fg.avgRelWidth /= sz;
		fg.avgRelHeight /= sz;
		if (szd) {
			fg.avgDoorsPerMeter /= szd;
			fg.avgRelDWidth /= szd;
			fg.avgDHeight /= szd;
		}

		// Calculate y offset
		if (fg.grammar != 3 && fg.grammar != 4) {
			// Get tallest facade in this group
			float minZts = 0.0;
			float maxHts = 0.0;
			for (auto fi : fg.facades) {
				auto& fp = facadeInfo[fi];
				for (auto& ws : facadeSections[fi]) {
					if (ws.maxBB.y - ws.minBB.y > maxHts) {
						maxHts = ws.maxBB.y - ws.minBB.y;
						minZts = fp.zBB_utm.x + ws.minBB.y;
					}
				}
			}

			int rows = floor(maxHts * fg.avgRowsPerMeter);
			fg.yoffset = (maxHts - rows / fg.avgRowsPerMeter) / 2 + minZts;
			fg.yoffset = fg.yoffset - floor(fg.yoffset * fg.avgRowsPerMeter) / fg.avgRowsPerMeter;
		}
	}		// for (auto& fg : facadeGroups) {


	// Iterate over all facades
	for (auto& fi : facadeInfo) {
		auto& fp = fi.second;
		string fiStr; {
			stringstream ss; ss << setw(6) << setfill('0') << fi.first;
			fiStr = ss.str();
		}

		// If valid parameters, add windows and doors
		if (fp.valid) {
			assert(whichGroup.count(fi.first));
			const auto& fg = facadeGroups[whichGroup[fi.first]];

			// Get sizes and spacing
			float winCellW = 1.0 / fg.avgColsPerMeter;
			float winCellH = 1.0 / fg.avgRowsPerMeter;
			float winW = winCellW * fg.avgRelWidth;
			float winH = winCellH * fg.avgRelHeight;
			float winXsep = winCellW * (1.0 - fg.avgRelWidth);
			float winYsep = winCellH * (1.0 - fg.avgRelHeight);
			float winXoff = winXsep / 2.0;
			float winYoff = winYsep / 2.0;
			float doorCellW = 1.0 / max(fg.avgDoorsPerMeter, 0.01f);
			float doorW = doorCellW * fg.avgRelDWidth;
			float doorH = fg.avgDHeight;
			float doorXsep = doorCellW * (1.0 - fg.avgRelDWidth);
			float doorXoff = doorXsep / 2.0;

			auto& xform = fp.rectXform;
			auto& invXform = fp.iRectXform;
			auto& winSections = facadeSections[fi.first];
			// Separate window sections into door sections if we have any doors
			struct DoorSection {
				glm::vec2 minBB;
				glm::vec2 maxBB;
				int cols;
				float xoffset;
			};
			vector<DoorSection> doorSections;
			if (fg.hasDoors) {
				// Iterate over window sections
				for (auto wi = winSections.begin(); wi != winSections.end();) {
					// Win section is entirely below door line
					if (wi->maxBB.y < doorH) {
						doorSections.push_back({});
						doorSections.back().minBB = wi->minBB;
						doorSections.back().maxBB = wi->maxBB;
						wi = winSections.erase(wi);

					// Win section is partially below door line
					} else if (wi->minBB.y < doorH) {
						doorSections.push_back({});
						doorSections.back().minBB = wi->minBB;
						doorSections.back().maxBB.x = wi->maxBB.x;
						doorSections.back().maxBB.y = doorH;
						wi->minBB.y = doorH;
						++wi;

					// Win section is completely above door line
					} else
						++wi;
				}

				// Combine adjacent door sections of equal vertical bounds
				for (auto di = doorSections.begin(); di != doorSections.end();) {
					auto diNext = di; ++diNext;
					if (diNext == doorSections.end()) break;
					if (abs(di->minBB.y - diNext->minBB.y) < 1e-4 &&
						abs(di->maxBB.y - diNext->maxBB.y) < 1e-4) {
						di->maxBB.x = diNext->maxBB.x;
						doorSections.erase(diNext);
					} else
						++di;
				}
			}

			// Method to write a face to the model and texture
			auto writeFace = [&](glm::vec3 va, glm::vec3 vb, glm::vec3 vc, glm::vec3 vd, bool window) {
				// Set the color to use
				glm::vec3 color = window ? fp.win_color : fp.bg_color;

				// Draw face onto synthetic atlas
				cv::Scalar drawCol(color.b * 255, color.g * 255, color.r * 255, 255);
				vector<vector<cv::Point>> pts(1);
				pts[0].push_back({ int(va.x / fp.size_utm.x * fp.atlasBB_px.width + fp.atlasBB_px.x),
					int((1.0 - va.y / fp.size_utm.y) * fp.atlasBB_px.height + fp.atlasBB_px.y) });
				pts[0].push_back({ int(vb.x / fp.size_utm.x * fp.atlasBB_px.width + fp.atlasBB_px.x),
					int((1.0 - vb.y / fp.size_utm.y) * fp.atlasBB_px.height + fp.atlasBB_px.y) });
				pts[0].push_back({ int(vc.x / fp.size_utm.x * fp.atlasBB_px.width + fp.atlasBB_px.x),
					int((1.0 - vc.y / fp.size_utm.y) * fp.atlasBB_px.height + fp.atlasBB_px.y) });
				pts[0].push_back({ int(vd.x / fp.size_utm.x * fp.atlasBB_px.width + fp.atlasBB_px.x),
					int((1.0 - vd.y / fp.size_utm.y) * fp.atlasBB_px.height + fp.atlasBB_px.y) });
				cv::fillPoly(synthAtlasImg, pts, drawCol);


				// Transform positions
				va = glm::vec3(invXform * glm::vec4(va, 1.0));
				vb = glm::vec3(invXform * glm::vec4(vb, 1.0));
				vc = glm::vec3(invXform * glm::vec4(vc, 1.0));
				vd = glm::vec3(invXform * glm::vec4(vd, 1.0));

				// Write positions
				objFile << "v " << va.x << " " << va.y << " " << va.z << endl;
				objFile << "v " << vb.x << " " << vb.y << " " << vb.z << endl;
				objFile << "v " << vc.x << " " << vc.y << " " << vc.z << endl;
				objFile << "v " << vd.x << " " << vd.y << " " << vd.z << endl;

				// Write indices
				objFile << "f " << vcount+1 << " " << vcount+2 << " " << vcount+3 << endl;
				objFile << "f " << vcount+3 << " " << vcount+4 << " " << vcount+1 << endl;

				vcount += 4;
			};

			// Add materials for window color and background
			mtlFile << "newmtl " << fiStr << "_bg" << endl;
			mtlFile << "Ka 0.00000000 0.00000000 0.00000000" << endl;
			mtlFile << "Kd "
				<< fixed << setprecision(8) << fp.bg_color.r << " "
				<< fixed << setprecision(8) << fp.bg_color.g << " "
				<< fixed << setprecision(8) << fp.bg_color.b << endl;
			mtlFile << "Ks 0.00000000 0.00000000 0.00000000" << endl;
			mtlFile << "illum 2" << endl;
			mtlFile << "Ns 0.00000000" << endl;

			mtlFile << "newmtl " << fiStr << "_win" << endl;
			mtlFile << "Ka 0.00000000 0.00000000 0.00000000" << endl;
			mtlFile << "Kd "
				<< fixed << setprecision(8) << fp.win_color.r << " "
				<< fixed << setprecision(8) << fp.win_color.g << " "
				<< fixed << setprecision(8) << fp.win_color.b << endl;
			mtlFile << "Ks 0.00000000 0.00000000 0.00000000" << endl;
			mtlFile << "illum 2" << endl;
			mtlFile << "Ns 0.00000000" << endl;

			objFile << "usemtl " << fiStr << "_bg" << endl;

			// Center doors on each door section
			for (auto& d : doorSections) {
				if (d.maxBB.y - d.minBB.y < doorH) {
					d.cols = 0;
				} else {
					d.cols = floor((d.maxBB.x - d.minBB.x + doorXsep / 2) / doorCellW);
					d.xoffset = ((d.maxBB.x - d.minBB.x) - d.cols * doorCellW) / 2;
				}

				// If no doors, just output the segment
				if (d.cols == 0) {
					glm::vec3 va(d.minBB.x, d.minBB.y, 0.0);
					glm::vec3 vb(d.maxBB.x, d.minBB.y, 0.0);
					glm::vec3 vc(d.maxBB.x, d.maxBB.y, 0.0);
					glm::vec3 vd(d.minBB.x, d.maxBB.y, 0.0);
					writeFace(va, vb, vc, vd, false);
					continue;
				}

				for (int c = 0; c < d.cols; c++) {
					float dMinX = d.minBB.x + d.xoffset + doorXoff + c * doorCellW;
					float dMaxX = dMinX + doorW;

					// If first doors, write spacing to left side of section
					if (c == 0) {
						glm::vec3 va(d.minBB.x, d.minBB.y, 0.0);
						glm::vec3 vb(dMinX, d.minBB.y, 0.0);
						glm::vec3 vc(dMinX, d.maxBB.y, 0.0);
						glm::vec3 vd(d.minBB.x, d.maxBB.y, 0.0);
						writeFace(va, vb, vc, vd, false);
					// Otherwise write the spacing before this door
					} else {
						glm::vec3 va(dMinX - doorXsep, d.minBB.y, 0.0);
						glm::vec3 vb(dMinX, d.minBB.y, 0.0);
						glm::vec3 vc(dMinX, d.maxBB.y, 0.0);
						glm::vec3 vd(dMinX - doorXsep, d.maxBB.y, 0.0);
						writeFace(va, vb, vc, vd, false);
					}

					// Get door vertices
					glm::vec3 va(dMinX, d.minBB.y, 0.0);
					glm::vec3 vb(dMaxX, d.minBB.y, 0.0);
					glm::vec3 vc(dMaxX, d.maxBB.y, 0.0);
					glm::vec3 vd(dMinX, d.maxBB.y, 0.0);
					glm::vec3 va2(dMinX, d.minBB.y, -recess);
					glm::vec3 vb2(dMaxX, d.minBB.y, -recess);
					glm::vec3 vc2(dMaxX, d.maxBB.y, -recess);
					glm::vec3 vd2(dMinX, d.maxBB.y, -recess);
					// Write the door boundaries
					if (recess > 0.0) {
						writeFace(vd2, vc2, vc, vd, false);
						writeFace(va, va2, vd2, vd, false);
						writeFace(vb2, vb, vc, vc2, false);
					}
					// Write the door face
					objFile << "usemtl " << fiStr << "_win" << endl;
					writeFace(va2, vb2, vc2, vd2, true);
					objFile << "usemtl " << fiStr << "_bg" << endl;

					// If last door, also write spacing to right side of section
					if (c+1 == d.cols) {
						glm::vec3 va(dMaxX, d.minBB.y, 0.0);
						glm::vec3 vb(d.maxBB.x, d.minBB.y, 0.0);
						glm::vec3 vc(d.maxBB.x, d.maxBB.y, 0.0);
						glm::vec3 vd(dMaxX, d.maxBB.y, 0.0);
						writeFace(va, vb, vc, vd, false);
					}
				}
			}

			// Skip if all window sections became door sections
			if (winSections.empty()) continue;

			// Output all windows
			for (int si = 0; si < winSections.size(); si++) {
				WinSection& s = winSections[si];
				if (fg.grammar != 5 && fg.grammar != 6) {
					// Stretch spacing between columns horizontally on all sections
					s.cols = floor((s.maxBB.x - s.minBB.x) / winCellW);
					s.xscale = (s.cols == 0) ? 1.0 : (s.maxBB.x - s.minBB.x) / (s.cols * winCellW);
				} else {
					// If horizontal windows, only use one column
					s.cols = 1;
					s.xscale = 1.0;
				}

				// Align all window rows
				if (fg.grammar != 3 && fg.grammar != 4) {
					s.yoffset = ceil((fp.zBB_utm.x + s.minBB.y - fg.yoffset) / winCellH) * winCellH
						+ fg.yoffset - (fp.zBB_utm.x + s.minBB.y);
					s.rows = floor((s.maxBB.y - s.minBB.y - s.yoffset) / winCellH);

				} else {
					// If vertical windows, only use one row (if section big enough)
					if (s.maxBB.y - s.minBB.y > 2 * g34_border) {
						s.rows = 1;
						s.yoffset = 0.0;
					} else {
						s.rows = 0;
						s.yoffset = 0.0;
					}
				}

				// If no rows or columns, just output the segment
				if (s.rows <= 0 || s.cols <= 0) {
					glm::vec3 va(s.minBB.x, s.minBB.y, 0.0);
					glm::vec3 vb(s.maxBB.x, s.minBB.y, 0.0);
					glm::vec3 vc(s.maxBB.x, s.maxBB.y, 0.0);
					glm::vec3 vd(s.minBB.x, s.maxBB.y, 0.0);
					writeFace(va, vb, vc, vd, false);
					continue;
				}

				for (int r = 0; r < s.rows; r++) {
					// Get spacing for special cases (vertical windows)
					float winYoff_s = winYoff;
					float winCellH_s = winCellH;
					float winH_s = winH;
					if (fg.grammar == 3 || fg.grammar == 4) {
						winCellH_s = s.maxBB.y - s.minBB.y;
						winYoff_s = g34_border;
						winH_s = winCellH_s - 2 * winYoff_s;
					}

					float wMinY = s.minBB.y + s.yoffset + winYoff_s + r * winCellH_s;
					float wMaxY = wMinY + winH_s;

					// If first row, write spacing below all windows
					if (r == 0) {
						glm::vec3 va(s.minBB.x, s.minBB.y, 0.0);
						glm::vec3 vb(s.maxBB.x, s.minBB.y, 0.0);
						glm::vec3 vc(s.maxBB.x, wMinY, 0.0);
						glm::vec3 vd(s.minBB.x, wMinY, 0.0);
						writeFace(va, vb, vc, vd, false);
					// Otherwise, write spacing between rows
					} else {
						glm::vec3 va(s.minBB.x, wMinY - winYsep, 0.0);
						glm::vec3 vb(s.maxBB.x, wMinY - winYsep, 0.0);
						glm::vec3 vc(s.maxBB.x, wMinY, 0.0);
						glm::vec3 vd(s.minBB.x, wMinY, 0.0);
						writeFace(va, vb, vc, vd, false);
					}

					// Write all windows in this row
					for (int c = 0; c < s.cols; c++) {
						float wXsep = winCellW * s.xscale - winW;
						float wMinX = s.minBB.x + wXsep / 2 + c * winCellW * s.xscale;
						float wMaxX = wMinX + winW;

						// Get spacing for special cases (horizontal windows)
						if (fg.grammar == 5 || fg.grammar == 6) {
							wMinX = s.minBB.x;
							wMaxX = s.maxBB.x;
						}

						if (fg.grammar != 5 && fg.grammar != 6) {
							// If first window, write spacing to the left of the row
							if (c == 0) {
								glm::vec3 va(s.minBB.x, wMinY, 0.0);
								glm::vec3 vb(wMinX, wMinY, 0.0);
								glm::vec3 vc(wMinX, wMaxY, 0.0);
								glm::vec3 vd(s.minBB.x, wMaxY, 0.0);
								writeFace(va, vb, vc, vd, false);
							// Otherwise, write spacing between columns
							} else {
								glm::vec3 va(wMinX - wXsep, wMinY, 0.0);
								glm::vec3 vb(wMinX, wMinY, 0.0);
								glm::vec3 vc(wMinX, wMaxY, 0.0);
								glm::vec3 vd(wMinX - wXsep, wMaxY, 0.0);
								writeFace(va, vb, vc, vd, false);
							}
						}

						// Get the window vertices
						glm::vec3 va(wMinX, wMinY, 0.0);
						glm::vec3 vb(wMaxX, wMinY, 0.0);
						glm::vec3 vc(wMaxX, wMaxY, 0.0);
						glm::vec3 vd(wMinX, wMaxY, 0.0);
						glm::vec3 va2(wMinX, wMinY, -recess);
						glm::vec3 vb2(wMaxX, wMinY, -recess);
						glm::vec3 vc2(wMaxX, wMaxY, -recess);
						glm::vec3 vd2(wMinX, wMaxY, -recess);
						// Write the window boundaries
						if (recess > 0.0) {
							writeFace(va, vb, vb2, va2, false);
							writeFace(vd2, vc2, vc, vd, false);
							writeFace(va, va2, vd2, vd, false);
							writeFace(vb2, vb, vc, vc2, false);
						}
						// Write the window in thie row/column
						objFile << "usemtl " << fiStr << "_win" << endl;
						writeFace(va2, vb2, vc2, vd2, true);
						objFile << "usemtl " << fiStr << "_bg" << endl;

						if (fg.grammar != 5 && fg.grammar != 6) {
							// If the last window, write spacing to the right of the row
							if (c+1 == s.cols) {
								glm::vec3 va(wMaxX, wMinY, 0.0);
								glm::vec3 vb(s.maxBB.x, wMinY, 0.0);
								glm::vec3 vc(s.maxBB.x, wMaxY, 0.0);
								glm::vec3 vd(wMaxX, wMaxY, 0.0);
								writeFace(va, vb, vc, vd, false);
							}
						}
					}

					// If last row, write spaceing above all windows
					if (r+1 == s.rows) {
						glm::vec3 va(s.minBB.x, wMaxY, 0.0);
						glm::vec3 vb(s.maxBB.x, wMaxY, 0.0);
						glm::vec3 vc(s.maxBB.x, s.maxBB.y, 0.0);
						glm::vec3 vd(s.minBB.x, s.maxBB.y, 0.0);
						writeFace(va, vb, vc, vd, false);
					}
				}
			}

		// If roof, use texture-mapped facade
		} else if (fp.roof) {
			// Use texture-mapped material
			objFile << "usemtl atlas" << endl;

			// Copy roof facade image to synth atlas
			atlasImg(fp.atlasBB_px).copyTo(synthAtlasImg(fp.atlasBB_px));

			// Add each triangle
			for (auto f : fp.faceIDs) {
				// Write positions
				glm::vec3 va = posBuf[indexBuf[3 * f + 0]];
				glm::vec3 vb = posBuf[indexBuf[3 * f + 1]];
				glm::vec3 vc = posBuf[indexBuf[3 * f + 2]];
				objFile << "v " << va.x << " " << va.y << " " << va.z << endl;
				objFile << "v " << vb.x << " " << vb.y << " " << vb.z << endl;
				objFile << "v " << vc.x << " " << vc.y << " " << vc.z << endl;

				// Write texture coords
				glm::vec2 ta = tcBuf[indexBuf[3 * f + 0]];
				glm::vec2 tb = tcBuf[indexBuf[3 * f + 1]];
				glm::vec2 tc = tcBuf[indexBuf[3 * f + 2]];
				objFile << "vt " << ta.x << " " << ta.y << endl;
				objFile << "vt " << tb.x << " " << tb.y << endl;
				objFile << "vt " << tc.x << " " << tc.y << endl;

				// Write indices
				objFile << "f " << vcount+1 << "/" << tcount+1 << " "
					<< vcount+2 << "/" << tcount+2 << " "
					<< vcount+3 << "/" << tcount+3 << endl;
				vcount += 3;
				tcount += 3;
			}

		// If non-roof, non-valid facade, use original facade
		} else {

			// Add material for this facade color
			mtlFile << "newmtl " << fiStr << "_bg" << endl;
			mtlFile << "Ka 0.00000000 0.00000000 0.00000000" << endl;
			mtlFile << "Kd "
				<< fixed << setprecision(8) << fp.bg_color.r << " "
				<< fixed << setprecision(8) << fp.bg_color.g << " "
				<< fixed << setprecision(8) << fp.bg_color.b << endl;
			mtlFile << "Ks 0.00000000 0.00000000 0.00000000" << endl;
			mtlFile << "illum 2" << endl;
			mtlFile << "Ns 0.00000000" << endl;

			// Use this material
			objFile << "usemtl " << fiStr << "_bg" << endl;

			// Add each triangle
			for (auto f : fp.faceIDs) {
				// Write positions
				glm::vec3 va = posBuf[indexBuf[3 * f + 0]];
				glm::vec3 vb = posBuf[indexBuf[3 * f + 1]];
				glm::vec3 vc = posBuf[indexBuf[3 * f + 2]];
				objFile << "v " << va.x << " " << va.y << " " << va.z << endl;
				objFile << "v " << vb.x << " " << vb.y << " " << vb.z << endl;
				objFile << "v " << vc.x << " " << vc.y << " " << vc.z << endl;

				// Write indices
				objFile << "f " << vcount+1 << " " << vcount+2 << " " << vcount+3 << endl;
				vcount += 3;

				// Draw triangle onto synth atlas
				cv::Scalar drawCol(fp.bg_color.b * 255, fp.bg_color.g * 255,
					fp.bg_color.r * 255, 255);
				vector<vector<cv::Point>> pts(1);
				glm::vec2 ta = tcBuf[indexBuf[3 * f + 0]];
				glm::vec2 tb = tcBuf[indexBuf[3 * f + 1]];
				glm::vec2 tc = tcBuf[indexBuf[3 * f + 2]];
				pts[0].push_back({ int(ta.x * atlasImg.cols), int((1.0 - ta.y) * atlasImg.rows) });
				pts[0].push_back({ int(tb.x * atlasImg.cols), int((1.0 - tb.y) * atlasImg.rows) });
				pts[0].push_back({ int(tc.x * atlasImg.cols), int((1.0 - tc.y) * atlasImg.rows) });
				cv::fillPoly(synthAtlasImg, pts, drawCol);
			}

		}
	}		// for (size_t fi = 0; fi < facadeInfo.size(); fi++) {

	// Save synthetic atlas
	cv::imwrite(texPath.string(), synthAtlasImg);
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

	// Get model name
	string modelFilename = modelPath.filename().string();
	size_t ib = modelFilename.find("__") + 2;
	size_t ie = modelFilename.find("__", ib);
	model = modelFilename.substr(ib, ie - ib);
//	model = "cgv_r";
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
	bg_color(0.7f),
	win_color(0.0f),
	rows(0),
	cols(0),
	grouping(0),
	relativeWidth(0.0),
	relativeHeight(0.0),
	doors(0),
	relativeDWidth(0.0),
	relativeDHeight(0.0) {}
