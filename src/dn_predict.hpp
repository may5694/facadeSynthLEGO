#ifndef DN_PREDICT_HPP
#define DN_PREDICT_HPP

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/filewritestream.h"
#include "building.hpp"

// Hold chip info
struct ChipInfo {
	cv::Mat src_image;
	cv::Mat seg_image;
	cv::Mat dnnIn_image;
	cv::Mat dilation_dst;
	cv::Mat aligned_img;
	int x; // Rect x
	int y; // Rect y
	int width; // Rect width
	int height; // Rect height
};

void dn_predict(FacadeInfo& fi, ModelInfo& mi, std::string debugPath);

/**** helper functions *****/
int reject(cv::Mat src_img, std::vector<double> facadeSize, std::vector<double> targetSize, double score, bool bDebug);
int reject(cv::Mat src_img, FacadeInfo& fi, ModelInfo& mi, bool bDebug);
cv::Mat cleanAlignedImage(cv::Mat src, float threshold);
cv::Mat deSkewImg(cv::Mat src_img);
void apply_segmentation_model(cv::Mat &croppedImage, cv::Mat &chip_seg, ModelInfo& mi, bool bDebug);
std::vector<int> adjust_chip(cv::Mat chip);
int choose_best_chip(std::vector<ChipInfo> chips, ModelInfo& mi, bool bDebug);
std::vector<double> compute_chip_info(ChipInfo chip, ModelInfo& mi, bool bDebug);
void find_spacing(cv::Mat src_img, std::vector<int> &separation_x, std::vector<int> &separation_y, bool bDebug);
void pre_process(cv::Mat &chip_seg, cv::Mat& croppedImage, ModelInfo& mi, bool bDebug);
/**** steps *****/
bool chipping(FacadeInfo& fi, ModelInfo& mi, ChipInfo &chip, bool bMultipleChips, bool bDebug);
std::vector<ChipInfo> crop_chip_ground(cv::Mat src_facade, int type, std::vector<double> facadeSize, std::vector<double> targetSize, bool bMultipleChips);
std::vector<ChipInfo> crop_chip_no_ground(cv::Mat src_facade, int type, std::vector<double> facadeSize, std::vector<double> targetSize, bool bMultipleChips);
bool process_chip(ChipInfo &chip, ModelInfo& mi, bool bDebug);
void feedDnn(ChipInfo &chip, FacadeInfo& fi, ModelInfo& mi, bool bDebug);

/**** grammar predictions ****/
std::vector<double> grammar1(ModelInfo& mi, std::vector<double> paras, bool bDebug);
std::vector<double> grammar2(ModelInfo& mi, std::vector<double> paras, bool bDebug);
std::vector<double> grammar3(ModelInfo& mi, std::vector<double> paras, bool bDebug);
std::vector<double> grammar4(ModelInfo& mi, std::vector<double> paras, bool bDebug);
std::vector<double> grammar5(ModelInfo& mi, std::vector<double> paras, bool bDebug);
std::vector<double> grammar6(ModelInfo& mi, std::vector<double> paras, bool bDebug);

#endif
