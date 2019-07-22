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

void dn_predict(FacadeInfo& fi, ModelInfo& mi);

/**** helper functions *****/
int reject(cv::Mat src_img, std::vector<double> facadeSize, std::vector<double> targetSize, double score, bool bDebug);
int reject(cv::Mat src_img, ModelInfo& mi, std::vector<double> facadeSize, std::vector<double> targetSize, std::vector<double> defaultImgSize, bool bDebug);
cv::Mat cleanAlignedImage(cv::Mat src, float threshold);
cv::Mat deSkewImg(cv::Mat src_img);
void apply_segmentation_model(cv::Mat &croppedImage, cv::Mat &chip_seg, ModelInfo& mi, bool bDebug);
std::vector<int> adjust_chip(cv::Mat chip);
int choose_best_chip(std::vector<cv::Mat> chips, ModelInfo& mi, bool bDebug);
std::vector<double> compute_chip_info(cv::Mat chip, ModelInfo& mi, bool bDebug);

/**** steps *****/
bool chipping(FacadeInfo& fi, ModelInfo& mi, cv::Mat& chip_seg, bool bMultipleChips, bool bDebug);
std::vector<cv::Mat> crop_chip_ground(cv::Mat src_facade, int type, std::vector<double> facadeSize, std::vector<double> targetSize, bool bMultipleChips);
std::vector<cv::Mat> crop_chip_no_ground(cv::Mat src_facade, int type, std::vector<double> facadeSize, std::vector<double> targetSize, bool bMultipleChips);
bool process_chip(cv::Mat chip_seg, cv::Mat& dnn_img, ModelInfo& mi, bool bDebug);
void feedDnn(cv::Mat dnn_img, FacadeInfo& fi, ModelInfo& mi, bool bDebug);

/**** grammar predictions ****/
std::vector<double> grammar1(ModelInfo& mi, std::vector<double> paras, bool bDebug);
std::vector<double> grammar2(ModelInfo& mi, std::vector<double> paras, bool bDebug);
std::vector<double> grammar3(ModelInfo& mi, std::vector<double> paras, bool bDebug);
std::vector<double> grammar4(ModelInfo& mi, std::vector<double> paras, bool bDebug);
std::vector<double> grammar5(ModelInfo& mi, std::vector<double> paras, bool bDebug);
std::vector<double> grammar6(ModelInfo& mi, std::vector<double> paras, bool bDebug);

#endif
