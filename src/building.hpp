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

class Building {
public:
	void load(fs::path metaPath);
	void clear();

private:
	// Directories
	fs::path metaDir;					// Metadata directory
	fs::path facadeModelDir;			// Synthetic model directory
	fs::path facadeTextureDir;			// Synthetic texture directory

	// Geometry buffers
	std::vector<glm::vec3> posBuf;		// Positions
	std::vector<glm::vec2> tcBuf;		// Texture coordinates
	std::vector<uint32_t> indexBuf;		// Triangle indices

	// Texture atlas
	cv::Mat atlasImg;					// Atlas texture image

	// Metadata
	std::string cluster;					// Cluster ID
	glm::vec3 minBB_utm, maxBB_utm;			// Bounding box min/max coords (UTM)
	std::map<int, FacadeInfo> facadeInfo;	// Per-facade info


	// Methods
	void readManifest(fs::path metaPath, fs::path& modelPath, fs::path& texPath, fs::path& surfPath);
	void readModel(fs::path modelPath);
};

// Holds information about a facade
struct FacadeInfo {
	std::vector<int> faceIDs;	// List of face IDs within this facade
	cv::Mat facadeImg;			// Facade texture (ROI of Building::atlasImg)
	cv::Rect atlasBB_px;		// Bounding rect of facade in atlas (px, ul origin)
	cv::Rect2f atlasBB_uv;		// Bounding rect of facade in atlas (uv, ll origin)
	glm::vec2 size_utm;			// Width, height of rectified facade (rUTM)
	glm::vec2 zBB_utm;			// Z bounds (UTM)
	glm::mat4 rectXform;		// UTM -> rUTM transformation matrix
	glm::mat4 iRectXform;		// rUTM -> UTM transformation matrix
	bool ground;				// Whether facade touches ground
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
