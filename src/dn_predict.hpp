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

void dn_predict(FacadeInfo& fi, const ModelInfo& mi, std::string debugPath);
cv::Mat pix2pix_seg(cv::Mat& src_img, ModelInfo& mi);

/**** helper functions *****/
int reject(cv::Mat src_img, std::vector<double> facadeSize, std::vector<double> targetSize, double score, bool bDebug);
int reject(cv::Mat src_img, FacadeInfo& fi, const ModelInfo& mi, bool bDebug);
cv::Mat cleanAlignedImage(cv::Mat src, float threshold);
cv::Mat deSkewImg(cv::Mat src_img);
void apply_segmentation_model(cv::Mat &croppedImage, cv::Mat &chip_seg, const ModelInfo& mi, bool bDebug);
std::vector<int> adjust_chip(cv::Mat chip);
int choose_best_chip(std::vector<ChipInfo> chips, const ModelInfo& mi, bool bDebug);
std::vector<double> compute_chip_info(ChipInfo chip, const ModelInfo& mi, bool bDebug);
void find_spacing(cv::Mat src_img, std::vector<int> &separation_x, std::vector<int> &separation_y, bool bDebug);
void pre_process(cv::Mat &chip_seg, cv::Mat& croppedImage, const ModelInfo& mi, bool bDebug);
/**** steps *****/
bool chipping(FacadeInfo& fi, const ModelInfo& mi, ChipInfo &chip, bool bMultipleChips, bool bDebug);
std::vector<ChipInfo> crop_chip_ground(cv::Mat src_facade, int type, std::vector<double> facadeSize, std::vector<double> targetSize, bool bMultipleChips);
std::vector<ChipInfo> crop_chip_no_ground(cv::Mat src_facade, int type, std::vector<double> facadeSize, std::vector<double> targetSize, bool bMultipleChips);
bool process_chip(ChipInfo &chip, const ModelInfo& mi, bool bDebug);
void feedDnn(ChipInfo &chip, FacadeInfo& fi, const ModelInfo& mi, bool bDebug);

/**** grammar predictions ****/
std::vector<double> grammar1(const ModelInfo& mi, std::vector<double> paras, bool bDebug);
std::vector<double> grammar2(const ModelInfo& mi, std::vector<double> paras, bool bDebug);
std::vector<double> grammar3(const ModelInfo& mi, std::vector<double> paras, bool bDebug);
std::vector<double> grammar4(const ModelInfo& mi, std::vector<double> paras, bool bDebug);
std::vector<double> grammar5(const ModelInfo& mi, std::vector<double> paras, bool bDebug);
std::vector<double> grammar6(const ModelInfo& mi, std::vector<double> paras, bool bDebug);

/**** opt ****/
void opt_without_doors(cv::Mat& seg_rbg, std::vector<double>& predictions_opt, std::vector<double> predictions_init);
void opt_with_doors(cv::Mat& seg_rbg, std::vector<double>& predictions_opt, std::vector<double> predictions_init);
cv::Mat synthesis_opt(std::vector<double> predictions, cv::Size src_size, cv::Scalar win_color, cv::Scalar bg_color, bool bDebug);
std::vector<double> eval_accuracy(const cv::Mat& seg_img, const cv::Mat& gt_img);

#endif
