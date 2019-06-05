#ifndef BUILDING_HPP
#define BUILDING_HPP

#include <array>
#include <vector>
#include <map>
#include <experimental/filesystem>
#include <glm/glm.hpp>
#include <opencv2/opencv.hpp>
namespace fs = std::experimental::filesystem;

struct FacadeInfo;

void genFacadeModel(const std::string &input_model_metadata_path, const std::string &,
                    const std::string &config_json_path, fs::path debugPath = {});

class Building {
public:
	Building(): debugOut(false) {}
	void load(fs::path metaPath, fs::path outputMetaPath, fs::path debugPath = {});
	void clear();

	void scoreFacades();
	void estimParams(fs::path configPath);
	void synthFacades();

private:

	// Geometry buffers
	std::vector<glm::vec3> posBuf;		// Positions
	std::vector<glm::vec2> tcBuf;		// Texture coordinates
	std::vector<uint32_t> indexBuf;		// Triangle indices

	// Texture atlas
	cv::Mat atlasImg;					// Atlas texture image

	// Metadata
	fs::path outputModelPath;						// Output model name
	fs::path outputTexPath;						// Output texture name
	fs::path outputMtlPath;						// Output mtl name
	glm::vec3 minBB_utm, maxBB_utm;			// Bounding box min/max coords (UTM)
	std::map<int, FacadeInfo> facadeInfo;	// Per-facade info
	bool debugOut;							// Whether to output debug info
	fs::path debugDir;						// Directory to save debug info to

	// Methods
	void readManifest(fs::path metaPath, fs::path& modelPath, fs::path& texPath, fs::path& mtlPath, fs::path& surfPath);
	void readModel(fs::path modelPath);
	void readSurfaces(fs::path surfPath);
	static cv::Rect findLargestRectangle(cv::Mat img);
};

// Holds information about a facade
struct FacadeInfo {
	std::vector<int> faceIDs;	// List of face IDs within this facade
	cv::Mat facadeImg;			// Facade texture (ROI of Building::atlasImg)
	cv::Rect atlasBB_px;		// Bounding rect of facade in atlas (px, ul origin)
	cv::Rect2f atlasBB_uv;		// Bounding rect of facade in atlas (uv, ll origin)
	cv::Rect inscRect_px;		// Largest inscribed rectangle (px, wrt atlasBB_px)
	glm::vec3 normal;			// Facing direction (UTM)
	glm::vec2 size_utm;			// Width, height of rectified facade (rUTM)
	glm::vec2 inscSize_utm;		// Width, height of inscribed rectangle (rUTM)
	glm::vec2 zBB_utm;			// Z bounds ((min, max), UTM)
	glm::mat4 rectXform;		// UTM -> rUTM transformation matrix
	glm::mat4 iRectXform;		// rUTM -> UTM transformation matrix
	bool ground;				// Whether facade touches ground
	bool inscGround;			// Whether inscribed rect touches ground
	bool roof;					// Whether facade is a roof

	// Estimated params
	float score;				// Score of the facade image
	bool valid;					// Whether parameters are valid
	int grammar;				// Procedural grammar to use
	std::array<float, 6> conf;	// Grammar confidence values
	glm::vec2 chip_size;		// Width, height of selected chip (rUTM)
	glm::vec3 bg_color;			// Color of facade background
	glm::vec3 win_color;		// Color of windows and doors
	int rows;					// Number of rows of windows per chip
	int cols;					// Number of columns of windows per chip
	int grouping;				// Number of windows per group
	float relativeWidth;		// Window width wrt cell (0, 1)
	float relativeHeight;		// Window height wrt cell (0, 1)
	int doors;					// Number of doors per chip
	float relativeDWidth;		// Width of doors wrt cell (0, 1)
	float relativeDHeight;		// Height of doors wrt facade chip (0, 1)

	FacadeInfo();
};

#endif
