#include "dn_predict.hpp"
#include <iostream>
#include <memory>
#include <stack>
#include <dlib/clustering.h>
#include <dlib/rand.h>
using namespace dlib;

void dn_predict(FacadeInfo& fi, ModelInfo& mi, std::string dn_path) {
	ChipInfo chip;
	bool bvalid = chipping(fi, mi, chip, true, false);
	if (bvalid) {
		cv::Mat dnn_img;
		process_chip(chip, mi, false);
		feedDnn(chip, fi, mi, false);
		if(!dn_path.empty()) {
			cv::imwrite(dn_path + "_facade.png", fi.facadeImg(fi.inscRect_px));
			cv::imwrite(dn_path + "_chip.png", chip.src_image);
			cv::imwrite(dn_path + "_seg.png", chip.seg_image);
			cv::imwrite(dn_path + "_dnnIn.png", chip.dnnIn_image);
		}
	}
}

int reject(cv::Mat src_img, std::vector<double> facadeSize, std::vector<double> targetSize, double score, bool bDebug) {
	int type = 0;
	if (facadeSize[0] < targetSize[0] && facadeSize[0] > 0.5 * targetSize[0] && facadeSize[1] < targetSize[1] && facadeSize[1] > 0.5 * targetSize[1] && score > 0.94) {
		type = 1;
	}
	else if (facadeSize[0] > targetSize[0] && facadeSize[1] < targetSize[1] && facadeSize[1] > 0.5 * targetSize[1] && score > 0.65) {
		type = 2;
	}
	else if (facadeSize[0] < targetSize[0] && facadeSize[0] > 0.5 * targetSize[0] && facadeSize[1] > targetSize[1] && score > 0.65) {
		type = 3;
	}
	else if (facadeSize[0] > targetSize[0] && facadeSize[1] > targetSize[1] && score > 0.68) {
		type = 4;
	}
	else {
		// do nothing
	}
	return type;
}

int reject(cv::Mat src_img, FacadeInfo& fi, ModelInfo& mi, bool bDebug) {
	// size of chip
	std::vector<double> facadeSize = { fi.inscSize_utm.x, fi.inscSize_utm.y };
	// size of nn image size
	std::vector<double> defaultImgSize = mi.defaultSize;
	// size of ideal chip size
	std::vector<double> targetSize = mi.targetChipSize;
	// if facades are too small threshold is 3m
	if (facadeSize[0] < 6 || facadeSize[1] < 6)
		return 0;
	// if the images are too small threshold is 25 by 25
	if (src_img.size().height < 25 || src_img.size().width < 25)
		return 0;
	cv::Mat scale_img;
	cv::resize(src_img, scale_img, cv::Size(defaultImgSize[0], defaultImgSize[1]));
	cv::Mat dnn_img_rgb;
	cv::cvtColor(scale_img, dnn_img_rgb, CV_BGR2RGB);
	cv::Mat img_float;
	dnn_img_rgb.convertTo(img_float, CV_32F, 1.0 / 255);
	auto img_tensor = torch::from_blob(img_float.data, { 1, 224, 224, 3 }).to(torch::kCUDA);
	img_tensor = img_tensor.permute({ 0, 3, 1, 2 });
	img_tensor[0][0] = img_tensor[0][0].sub(0.485).div(0.229);
	img_tensor[0][1] = img_tensor[0][1].sub(0.456).div(0.224);
	img_tensor[0][2] = img_tensor[0][2].sub(0.406).div(0.225);

	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(img_tensor);
	// reject model classifier
	// Deserialize the ScriptModule from a file using torch::jit::load().
	//std::shared_ptr<torch::jit::script::Module> reject_classifier_module = torch::jit::load(model_path);
	//reject_classifier_module->to(at::kCUDA);
	//assert(reject_classifier_module != nullptr);
	torch::Tensor out_tensor = mi.reject_classifier_module->forward(inputs).toTensor();

	torch::Tensor confidences_tensor = torch::softmax(out_tensor, 1);

	double best_score = 0;
	int best_class = -1;
	for (int i = 0; i < 2; i++) {
		double tmp = confidences_tensor.slice(1, i, i + 1).item<float>();
		if (tmp > best_score) {
			best_score = tmp;
			best_class = i;
		}
	}
	if (bDebug) {
		//std::cout << out_tensor.slice(1, 0, 2) << std::endl;
		std::cout << confidences_tensor.slice(1, 0, 2) << std::endl;
		std::cout << "Reject class is " << best_class << std::endl;
	}
	if (best_class == 1|| best_score < 0.96) // bad facades
		return 0;
	else {
		fi.good_conf = best_score;
		int type = 0;
		if (facadeSize[0] < targetSize[0] && facadeSize[1] < targetSize[1]) {
			type = 1;
		}
		else if (facadeSize[0] > targetSize[0] && facadeSize[1] < targetSize[1]) {
			type = 2;
		}
		else if (facadeSize[0] < targetSize[0] && facadeSize[1] > targetSize[1]) {
			type = 3;
		}
		else if (facadeSize[0] > targetSize[0] && facadeSize[1] > targetSize[1]) {
			type = 4;
		}
		else {
			// do nothing
		}
		return type;
	}
}

bool chipping(FacadeInfo& fi, ModelInfo& mi, ChipInfo &chip, bool bMultipleChips, bool bDebug) {
	// size of chip
	std::vector<double> facadeSize = { fi.inscSize_utm.x, fi.inscSize_utm.y };
	// roof 
	bool broof = fi.roof;
	// ground
	bool bground = fi.inscGround;
	// image file
	cv::Mat src_facade = fi.facadeImg(fi.inscRect_px).clone();
	if(src_facade.channels() == 4) // ensure there're 3 channels
		cv::cvtColor(src_facade, src_facade, CV_BGRA2BGR);
	// score
	double score = fi.score;
	// first decide whether it's a valid chip
	std::vector<double> targetSize = mi.targetChipSize;
	if (targetSize.size() != 2) {
		std::cout << "Please check the targetChipSize member in the JSON file" << std::endl;
		return false;
	}
	// if it's not a roof
	int type = 0;
	if (!broof)
		type = reject(src_facade, fi, mi, mi.debug);
	if (type == 0) {
		fi.valid = false;
		// compute avg color
		cv::Scalar avg_color(0, 0, 0);
		for (int i = 0; i < src_facade.size().height; i++) {
			for (int j = 0; j < src_facade.size().width; j++) {
				for (int c = 0; c < 3; c++) {
					if (src_facade.channels() == 4)
						avg_color.val[c] += src_facade.at<cv::Vec4b>(i, j)[c];
					if (src_facade.channels() == 3)
						avg_color.val[c] += src_facade.at<cv::Vec3b>(i, j)[c];
				}
			}
		}
		fi.bg_color.b = avg_color.val[0] / (src_facade.size().height * src_facade.size().width) / 255;
		fi.bg_color.g = avg_color.val[1] / (src_facade.size().height * src_facade.size().width) / 255;
		fi.bg_color.r = avg_color.val[2] / (src_facade.size().height * src_facade.size().width) / 255;
		return false;
	}
	if (bDebug) {
		std::cout << "facadeSize is " << facadeSize << std::endl;
		std::cout << "broof is " << broof << std::endl;
		std::cout << "bground is " << bground << std::endl;
		std::cout << "score is " << score << std::endl;
		std::cout << "targetSize is " << targetSize << std::endl;
	}
	fi.valid = true;

	// choose the best chip
	std::vector<ChipInfo> cropped_chips = crop_chip_no_ground(src_facade.clone(), type, facadeSize, targetSize, bMultipleChips);
	int best_chip_id = choose_best_chip(cropped_chips, mi, bDebug);
	// adjust the best chip
	cv::Mat croppedImage = cropped_chips[best_chip_id].src_image.clone();
	cv::Mat chip_seg;
	pre_process(chip_seg, croppedImage, mi, bDebug);

	// add real chip size
	int chip_width = croppedImage.size().width;
	int chip_height = croppedImage.size().height;
	int src_width = src_facade.size().width;
	int src_height = src_facade.size().height;
	fi.chip_size.x = chip_width * 1.0 / src_width * facadeSize[0];
	fi.chip_size.y = chip_height * 1.0 / src_height * facadeSize[1];
	// write back to json file
	cv::Scalar bg_avg_color(0, 0, 0);
	cv::Scalar win_avg_color(0, 0, 0);
	{
		int bg_count = 0;
		int win_count = 0;
		for (int i = 0; i < chip_seg.size().height; i++) {
			for (int j = 0; j < chip_seg.size().width; j++) {
				if ((int)chip_seg.at<uchar>(i, j) == 0) {
					if (croppedImage.channels() == 4) {
						win_avg_color.val[0] += croppedImage.at<cv::Vec4b>(i, j)[0];
						win_avg_color.val[1] += croppedImage.at<cv::Vec4b>(i, j)[1];
						win_avg_color.val[2] += croppedImage.at<cv::Vec4b>(i, j)[2];
					}
					if (croppedImage.channels() == 3) {
						win_avg_color.val[0] += croppedImage.at<cv::Vec3b>(i, j)[0];
						win_avg_color.val[1] += croppedImage.at<cv::Vec3b>(i, j)[1];
						win_avg_color.val[2] += croppedImage.at<cv::Vec3b>(i, j)[2];
					}
					win_count++;
				}
				else {
					if (croppedImage.channels() == 4) {
						bg_avg_color.val[0] += croppedImage.at<cv::Vec4b>(i, j)[0];
						bg_avg_color.val[1] += croppedImage.at<cv::Vec4b>(i, j)[1];
						bg_avg_color.val[2] += croppedImage.at<cv::Vec4b>(i, j)[2];
					}
					if (croppedImage.channels() == 3) {
						bg_avg_color.val[0] += croppedImage.at<cv::Vec3b>(i, j)[0];
						bg_avg_color.val[1] += croppedImage.at<cv::Vec3b>(i, j)[1];
						bg_avg_color.val[2] += croppedImage.at<cv::Vec3b>(i, j)[2];
					}
					bg_count++;
				}
			}
		}
		if (win_count > 0) {
			win_avg_color.val[0] = win_avg_color.val[0] / win_count;
			win_avg_color.val[1] = win_avg_color.val[1] / win_count;
			win_avg_color.val[2] = win_avg_color.val[2] / win_count;
		}
		if (bg_count > 0) {
			bg_avg_color.val[0] = bg_avg_color.val[0] / bg_count;
			bg_avg_color.val[1] = bg_avg_color.val[1] / bg_count;
			bg_avg_color.val[2] = bg_avg_color.val[2] / bg_count;
		}
	}
	fi.bg_color.b = bg_avg_color.val[0] / 255;
	fi.bg_color.g = bg_avg_color.val[1] / 255;
	fi.bg_color.r = bg_avg_color.val[2] / 255;

	fi.win_color.b = win_avg_color.val[0] / 255;
	fi.win_color.g = win_avg_color.val[1] / 255;
	fi.win_color.r = win_avg_color.val[2] / 255;
	// copy info to the chip
	chip.src_image = croppedImage.clone();
	chip.seg_image = chip_seg.clone();
	if (bDebug) {
		std::cout << "Facade type is " << type << std::endl;
	}
	return true;
}

void pre_process(cv::Mat &chip_seg, cv::Mat& croppedImage, ModelInfo& mi, bool bDebug) {
	// --- step 1
	apply_segmentation_model(croppedImage, chip_seg, mi, bDebug);
	cv::Mat scale_img;
	cv::resize(chip_seg, scale_img, cv::Size(mi.defaultSize[0], mi.defaultSize[1]));
	// correct the color
	for (int i = 0; i < scale_img.size().height; i++) {
		for (int j = 0; j < scale_img.size().width; j++) {
			//noise
			if ((int)scale_img.at<uchar>(i, j) < 128) {
				scale_img.at<uchar>(i, j) = (uchar)0;
			}
			else
				scale_img.at<uchar>(i, j) = (uchar)255;
		}
	}
	int dilation_type = cv::MORPH_RECT;
	cv::Mat img_dilation;
	int kernel_size = 3;
	cv::Mat element = cv::getStructuringElement(dilation_type, cv::Size(kernel_size, kernel_size), cv::Point(kernel_size / 2, kernel_size / 2));
	/// Apply the dilation operation
	cv::dilate(scale_img, img_dilation, element);
	cv::resize(img_dilation, img_dilation, chip_seg.size());
	// correct the color
	for (int i = 0; i < img_dilation.size().height; i++) {
		for (int j = 0; j < img_dilation.size().width; j++) {
			//noise
			if ((int)img_dilation.at<uchar>(i, j) < 128) {
				img_dilation.at<uchar>(i, j) = (uchar)0;
			}
			else
				img_dilation.at<uchar>(i, j) = (uchar)255;
		}
	}
	int padding_size = mi.paddingSize[0];
	int borderType = cv::BORDER_CONSTANT;
	cv::Mat padding_img;
	cv::copyMakeBorder(img_dilation, padding_img, padding_size, padding_size, padding_size, padding_size, borderType, cv::Scalar(255, 255, 255));
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(padding_img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	std::vector<float> area_contours(contours.size());
	std::vector<cv::Rect> boundRect(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		boundRect[i] = cv::boundingRect(cv::Mat(contours[i]));
		area_contours[i] = cv::contourArea(contours[i]);
	}
	std::sort(area_contours.begin(), area_contours.end());
	float target_contour_area = area_contours[area_contours.size() / 2];
	for (int i = 0; i < contours.size(); i++)
	{
		float area_contour = cv::contourArea(contours[i]);
		if (area_contour < 0.5 * target_contour_area) {
			cv::rectangle(padding_img, cv::Point(boundRect[i].tl().x, boundRect[i].tl().y), cv::Point(boundRect[i].br().x, boundRect[i].br().y), cv::Scalar(255, 255, 255), -1);
		}
	}
	chip_seg = padding_img(cv::Rect(padding_size, padding_size, chip_seg.size().width, chip_seg.size().height));
	// --- step 2
	std::vector<int> boundaries = adjust_chip(chip_seg.clone());
	chip_seg = chip_seg(cv::Rect(boundaries[2], boundaries[0], boundaries[3] - boundaries[2] + 1, boundaries[1] - boundaries[0] + 1));
	croppedImage = croppedImage(cv::Rect(boundaries[2], boundaries[0], boundaries[3] - boundaries[2] + 1, boundaries[1] - boundaries[0] + 1));
	// ---- step 3
	if (chip_seg.size().width < 25)
		return;
	std::vector<int> separation_x;
	std::vector<int> separation_y;
	cv::Mat spacing_img = chip_seg.clone();
	float threshold_spacing = 0.238;
	find_spacing(spacing_img, separation_x, separation_y, bDebug);
	if (separation_x.size() % 2 != 0)
		return;
	if (separation_x.size() > 0) {
		// compute spacing
		std::vector<int> space_x;
		for (int i = 0; i < separation_x.size(); i += 2) {
			space_x.push_back(separation_x[i + 1] - separation_x[i]);
		}
		int max_spacing_x_id = std::max_element(space_x.begin(), space_x.end()) - space_x.begin();
		double ratio_x = space_x[max_spacing_x_id] * 1.0 / spacing_img.size().width;
		if (ratio_x > threshold_spacing)
		{
			if (space_x.size() >= 2) {
				float average_spacing = accumulate(space_x.begin(), space_x.end(), 0.0) / space_x.size();
				if (abs(space_x[max_spacing_x_id] - average_spacing) / spacing_img.size().width < 0.05)
					return;
			}
			// split into two images
			cv::Mat img_1_seg = spacing_img(cv::Rect(0, 0, separation_x[max_spacing_x_id * 2], spacing_img.size().height));
			cv::Mat img_2_seg = spacing_img(cv::Rect(separation_x[max_spacing_x_id * 2 + 1], 0, spacing_img.size().width - separation_x[max_spacing_x_id * 2 + 1], spacing_img.size().height));
			cv::Mat chip_spacing;
			if (img_1_seg.size().width > img_2_seg.size().width) {
				chip_spacing = croppedImage(cv::Rect(0, 0, separation_x[max_spacing_x_id * 2], spacing_img.size().height));
				// clean up
				std::vector<int> boundaries = adjust_chip(img_1_seg.clone());
				img_1_seg = img_1_seg(cv::Rect(boundaries[2], boundaries[0], boundaries[3] - boundaries[2] + 1, boundaries[1] - boundaries[0] + 1));
				chip_spacing = chip_spacing(cv::Rect(boundaries[2], boundaries[0], boundaries[3] - boundaries[2] + 1, boundaries[1] - boundaries[0] + 1));
				chip_seg = img_1_seg.clone();
				croppedImage = chip_spacing.clone();
			}
			else {
				chip_spacing = croppedImage(cv::Rect(separation_x[max_spacing_x_id * 2 + 1], 0, spacing_img.size().width - separation_x[max_spacing_x_id * 2 + 1], spacing_img.size().height));
				// clearn up
				std::vector<int> boundaries = adjust_chip(img_2_seg.clone());
				img_2_seg = img_2_seg(cv::Rect(boundaries[2], boundaries[0], boundaries[3] - boundaries[2] + 1, boundaries[1] - boundaries[0] + 1));
				chip_spacing = chip_spacing(cv::Rect(boundaries[2], boundaries[0], boundaries[3] - boundaries[2] + 1, boundaries[1] - boundaries[0] + 1));
				chip_seg = img_2_seg.clone();
				croppedImage = chip_spacing.clone();
			}
		}
	}
}

std::vector<ChipInfo> crop_chip_no_ground(cv::Mat src_facade, int type, std::vector<double> facadeSize, std::vector<double> targetSize, bool bMultipleChips) {
	std::vector<ChipInfo> cropped_chips;
	double ratio_upper = 0.95;
	double ratio_lower = 0.05;
	double ratio_step = 0.05;
	double target_width = targetSize[0];
	double target_height = targetSize[1];
	if (type == 1) {
		ChipInfo chip;
		chip.src_image = src_facade;
		chip.x = 0;
		chip.y = 0;
		chip.width = src_facade.size().width;
		chip.height = src_facade.size().height;
		cropped_chips.push_back(chip);
	}
	else if (type == 2) {
		double target_ratio_width = target_width / facadeSize[0];
		double target_ratio_height = target_height / facadeSize[1];
		if (target_ratio_height > 1.0)
			target_ratio_height = 1.0;
		if (facadeSize[0] < 1.6 * target_width || !bMultipleChips) {
			double start_width_ratio = (1 - target_ratio_width) * 0.5;
			// crop target size
			ChipInfo chip;
			chip.src_image = src_facade(cv::Rect(src_facade.size().width * start_width_ratio, 0, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
			chip.x = src_facade.size().width * start_width_ratio;
			chip.y = 0;
			chip.width = src_facade.size().width * target_ratio_width;
			chip.height = src_facade.size().height * target_ratio_height;
			cropped_chips.push_back(chip);
		}
		else {
			// push back multiple chips
			int index = 1;
			double start_width_ratio = index * ratio_lower; // not too left
			std::vector<double> confidences;
			while (start_width_ratio + target_ratio_width < ratio_upper) { // not too right
																		   // get the cropped img
				ChipInfo chip;
				chip.src_image = src_facade(cv::Rect(src_facade.size().width * start_width_ratio, 0, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));;
				chip.x = src_facade.size().width * start_width_ratio;
				chip.y = 0;
				chip.width = src_facade.size().width * target_ratio_width;
				chip.height = src_facade.size().height * target_ratio_height;
				cropped_chips.push_back(chip);
				index++;
				start_width_ratio = index * ratio_step;
			}
		}
	}
	else if (type == 3) {
		double target_ratio_height = target_height / facadeSize[1];
		double target_ratio_width = target_width / facadeSize[0];
		if (target_ratio_width >= 1.0)
			target_ratio_width = 1.0;
		if (facadeSize[1] < 1.6 * target_height || !bMultipleChips) {
			double start_height_ratio = (1 - target_ratio_height) * 0.5;
			ChipInfo chip;
			chip.src_image = src_facade(cv::Rect(0, src_facade.size().height * start_height_ratio, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
			chip.x = 0;
			chip.y = src_facade.size().height * start_height_ratio;
			chip.width = src_facade.size().width * target_ratio_width;
			chip.height = src_facade.size().height * target_ratio_height;
			cropped_chips.push_back(chip);
		}
		else {
			// push back multiple chips
			int index = 1;
			double start_height_ratio = index * ratio_lower;
			while (start_height_ratio + target_ratio_height < ratio_upper) {
				// get the cropped img
				ChipInfo chip;
				chip.src_image = src_facade(cv::Rect(0, src_facade.size().height * start_height_ratio, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
				chip.x = 0;
				chip.y = src_facade.size().height * start_height_ratio;
				chip.width = src_facade.size().width * target_ratio_width;
				chip.height = src_facade.size().height * target_ratio_height;
				cropped_chips.push_back(chip);
				index++;
				start_height_ratio = index * ratio_step;
			}
		}
	}
	else if (type == 4) {
		double longer_dim = facadeSize[0] > facadeSize[1] ? facadeSize[0] : facadeSize[1];
		double target_dim = facadeSize[0] > facadeSize[1] ? target_width : target_height;
		bool bLonger_width = facadeSize[0] > facadeSize[1] ? true : false;
		double target_ratio_width = target_width / facadeSize[0];
		double target_ratio_height = target_height / facadeSize[1];
		if (longer_dim < 1.6 * target_dim || !bMultipleChips) {
			// crop target size
			double start_width_ratio = (1 - target_ratio_width) * 0.5;
			double start_height_ratio = (1 - target_ratio_height) * 0.5;
			ChipInfo chip;
			chip.src_image = src_facade(cv::Rect(src_facade.size().width * start_width_ratio, src_facade.size().height * start_height_ratio, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
			chip.x = src_facade.size().width * start_width_ratio;
			chip.y = src_facade.size().height * start_height_ratio;
			chip.width = src_facade.size().width * target_ratio_width;
			chip.height = src_facade.size().height * target_ratio_height;
			cropped_chips.push_back(chip);
		}
		else if (bLonger_width) {
			// check multiple chips and choose the one that has the highest confidence value
			int index = 1;
			double start_width_ratio = index * ratio_lower;
			double start_height_ratio = (1 - target_ratio_height) * 0.5;
			while (start_width_ratio + target_ratio_width < ratio_upper) {
				// get the cropped img
				ChipInfo chip;
				chip.src_image = src_facade(cv::Rect(src_facade.size().width * start_width_ratio, src_facade.size().height * start_height_ratio, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
				chip.x = src_facade.size().width * start_width_ratio;
				chip.y = src_facade.size().height * start_height_ratio;
				chip.width = src_facade.size().width * target_ratio_width;
				chip.height = src_facade.size().height * target_ratio_height;
				cropped_chips.push_back(chip);
				index++;
				start_width_ratio = index * ratio_step;
			}
		}
		else {
			// check multiple chips and choose the one that has the highest confidence value
			int index = 1;
			double start_height_ratio = index * ratio_lower;
			double start_width_ratio = (1 - target_ratio_width) * 0.5;
			while (start_height_ratio + target_ratio_height < ratio_upper) {
				// get the cropped img
				ChipInfo chip;
				chip.src_image = src_facade(cv::Rect(src_facade.size().width * start_width_ratio, src_facade.size().height * start_height_ratio, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
				chip.x = src_facade.size().width * start_width_ratio;
				chip.y = src_facade.size().height * start_height_ratio;
				chip.width = src_facade.size().width * target_ratio_width;
				chip.height = src_facade.size().height * target_ratio_height;
				cropped_chips.push_back(chip);
				index++;
				start_height_ratio = index * ratio_step;
			}
		}
	}
	else {
		// do nothing
	}
	return cropped_chips;
}

std::vector<ChipInfo> crop_chip_ground(cv::Mat src_facade, int type, std::vector<double> facadeSize, std::vector<double> targetSize, bool bMultipleChips) {
	double ratio_upper = 0.95;
	double ratio_lower = 0.05;
	double ratio_step = 0.1;
	std::vector<ChipInfo> cropped_chips;
	double target_width = targetSize[0];
	double target_height = targetSize[1];
	if (type == 1) {
		ChipInfo chip;
		chip.src_image = src_facade;
		chip.x = 0;
		chip.y = 0;
		chip.width = src_facade.size().width;
		chip.height = src_facade.size().height;
		cropped_chips.push_back(chip);
	}
	else if (type == 2) {
		double target_ratio_width = target_width / facadeSize[0];
		double target_ratio_height = target_height / facadeSize[1];
		if (target_ratio_height > 1.0)
			target_ratio_height = 1.0;
		if (facadeSize[0] < 1.6 * target_width || !bMultipleChips) {
			double start_width_ratio = (1 - target_ratio_width) * 0.5;
			// crop target size
			ChipInfo chip;
			chip.src_image = src_facade(cv::Rect(src_facade.size().width * start_width_ratio, 0, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
			chip.x = src_facade.size().width * start_width_ratio;
			chip.y = 0;
			chip.width = src_facade.size().width * target_ratio_width;
			chip.height = src_facade.size().height * target_ratio_height;
			cropped_chips.push_back(chip);
		}
		else {
			// push back multiple chips
			int index = 1;
			double start_width_ratio = index * ratio_lower; // not too left
			std::vector<double> confidences;
			while (start_width_ratio + target_ratio_width < ratio_upper) { // not too right
																		   // get the cropped img
				ChipInfo chip;
				chip.src_image = src_facade(cv::Rect(src_facade.size().width * start_width_ratio, 0, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
				chip.x = src_facade.size().width * start_width_ratio;
				chip.y = 0;
				chip.width = src_facade.size().width * target_ratio_width;
				chip.height = src_facade.size().height * target_ratio_height;
				cropped_chips.push_back(chip);
				index++;
				start_width_ratio = index * ratio_step;
			}
		}
	}
	else if (type == 3) {
		double target_ratio_height = target_height / facadeSize[1];
		double target_ratio_width = target_width / facadeSize[0];
		if (target_ratio_width >= 1.0)
			target_ratio_width = 1.0;
		if (facadeSize[1] < 1.6 * target_height || !bMultipleChips) {
			double start_height_ratio = (1 - target_ratio_height);
			ChipInfo chip;
			chip.src_image = src_facade(cv::Rect(0, src_facade.size().height * start_height_ratio, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
			chip.x = 0;
			chip.y = src_facade.size().height * start_height_ratio;
			chip.width = src_facade.size().width * target_ratio_width;
			chip.height = src_facade.size().height * target_ratio_height;
			cropped_chips.push_back(chip);
		}
	}
	else if (type == 4) {
		double longer_dim = facadeSize[0] > facadeSize[1] ? facadeSize[0] : facadeSize[1];
		double target_dim = facadeSize[0] > facadeSize[1] ? target_width : target_height;
		bool bLonger_width = facadeSize[0] > facadeSize[1] ? true : false;
		double target_ratio_width = target_width / facadeSize[0];
		double target_ratio_height = target_height / facadeSize[1];
		if (longer_dim < 1.6 * target_dim || !bMultipleChips) {
			// crop target size
			double start_width_ratio = (1 - target_ratio_width) * 0.5;
			double start_height_ratio = (1 - target_ratio_height);
			ChipInfo chip;
			chip.src_image = src_facade(cv::Rect(src_facade.size().width * start_width_ratio, src_facade.size().height * start_height_ratio, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
			chip.x = src_facade.size().width * start_width_ratio;
			chip.y = src_facade.size().height * start_height_ratio;
			chip.width = src_facade.size().width * target_ratio_width;
			chip.height = src_facade.size().height * target_ratio_height;
			cropped_chips.push_back(chip);
		}
		else if (bLonger_width) {
			// check multiple chips and choose the one that has the highest confidence value
			int index = 1;
			double start_width_ratio = index * ratio_lower;
			double start_height_ratio = (1 - target_ratio_height);
			while (start_width_ratio + target_ratio_width < ratio_upper) {
				// get the cropped img
				ChipInfo chip;
				chip.src_image = src_facade(cv::Rect(src_facade.size().width * start_width_ratio, src_facade.size().height * start_height_ratio, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
				chip.x = src_facade.size().width * start_width_ratio;
				chip.y = src_facade.size().height * start_height_ratio;
				chip.width = src_facade.size().width * target_ratio_width;
				chip.height = src_facade.size().height * target_ratio_height;
				cropped_chips.push_back(chip);
				index++;
				start_width_ratio = index * ratio_step;
			}
		}
		else {
			// check multiple chips and choose the one that has the highest confidence value
			int index = 1;
			double start_height_ratio = (1 - target_ratio_height);
			double start_width_ratio = (1 - target_ratio_width) * 0.5;
			ChipInfo chip;
			chip.src_image = src_facade(cv::Rect(src_facade.size().width * start_width_ratio, src_facade.size().height * start_height_ratio, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
			chip.x = src_facade.size().width * start_width_ratio;
			chip.y = src_facade.size().height * start_height_ratio;
			chip.width = src_facade.size().width * target_ratio_width;
			chip.height = src_facade.size().height * target_ratio_height;
			cropped_chips.push_back(chip);
		}
	}
	else {
		// do nothing
	}
	return cropped_chips;
}

std::vector<double> compute_chip_info(ChipInfo chip, ModelInfo& mi, bool bDebug) {
	std::vector<double> chip_info;
	apply_segmentation_model(chip.src_image, chip.seg_image, mi, false);
	//adjust boundaries
	std::vector<int> boundaries = adjust_chip(chip.seg_image.clone());
	chip.seg_image = chip.seg_image(cv::Rect(boundaries[2], boundaries[0], boundaries[3] - boundaries[2] + 1, boundaries[1] - boundaries[0] + 1));
	chip.src_image = chip.src_image(cv::Rect(boundaries[2], boundaries[0], boundaries[3] - boundaries[2] + 1, boundaries[1] - boundaries[0] + 1));
	process_chip(chip, mi, false);
	// go to grammar classifier
	int num_classes = mi.number_grammars;
	cv::Mat dnn_img_rgb;
	cv::cvtColor(chip.dnnIn_image.clone(), dnn_img_rgb, CV_BGR2RGB);
	cv::Mat img_float;
	dnn_img_rgb.convertTo(img_float, CV_32F, 1.0 / 255);
	auto img_tensor = torch::from_blob(img_float.data, { 1, 224, 224, 3 }).to(torch::kCUDA);
	img_tensor = img_tensor.permute({ 0, 3, 1, 2 });
	img_tensor[0][0] = img_tensor[0][0].sub(0.485).div(0.229);
	img_tensor[0][1] = img_tensor[0][1].sub(0.456).div(0.224);
	img_tensor[0][2] = img_tensor[0][2].sub(0.406).div(0.225);

	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(img_tensor);

	int best_class = -1;
	// Deserialize the ScriptModule from a file using torch::jit::load().
	torch::Tensor out_tensor = mi.classifier_module->forward(inputs).toTensor();
	torch::Tensor confidences_tensor = torch::softmax(out_tensor, 1);
	//std::cout << confidences_tensor.slice(1, 0, num_classes) << std::endl;
	double best_score = 0;
	for (int i = 0; i < num_classes; i++) {
		double tmp = confidences_tensor.slice(1, i, i + 1).item<float>();
		if (tmp > best_score) {
			best_score = tmp;
			best_class = i;
		}
	}
	// currently only add confidence value to the info vector
	chip_info.push_back(best_score);
	return chip_info;
}

int choose_best_chip(std::vector<ChipInfo> chips, ModelInfo& mi, bool bDebug) {
	int best_chip_id = 0;
	if (chips.size() == 1)
		best_chip_id = 0;
	else {
		std::vector<double> confidence_values;
		confidence_values.resize(chips.size());
		for (int i = 0; i < chips.size(); i++) {
			confidence_values[i] = compute_chip_info(chips[i], mi, bDebug)[0];
			if (bDebug) {
				std::cout << "chip " << i << " score is " << confidence_values[i] << std::endl;
			}
		}
		// find the best chip id
		best_chip_id = std::max_element(confidence_values.begin(), confidence_values.end()) - confidence_values.begin();
		if (bDebug) {
			std::cout << "best_chip_id is " << best_chip_id << std::endl;
		}
	}
	return best_chip_id;
}

std::vector<int> adjust_chip(cv::Mat chip) {
	std::vector<int> boundaries;
	boundaries.resize(4); // top, bottom, left and right
	if (chip.channels() != 1) {
		boundaries[0] = 0;
		boundaries[1] = chip.size().height - 1;
		boundaries[2] = 0;
		boundaries[3] = chip.size().width - 1;
		return boundaries;
	}
	// find the boundary
	double thre_upper = 1.1; // don't apply here
	double thre_lower = 0.1;
	// top 
	int pos_top = 0;
	int black_pixels = 0;
	for (int i = 0; i < chip.size().height; i++) {
		black_pixels = 0;
		for (int j = 0; j < chip.size().width; j++) {
			if ((int)chip.at<uchar>(i, j) == 0) {
				black_pixels++;
			}
		}
		double ratio = black_pixels * 1.0 / chip.size().width;
		if (ratio < thre_upper && ratio > thre_lower) {
			pos_top = i;
			break;
		}
	}

	// bottom 
	black_pixels = 0;
	int pos_bot = 0;
	for (int i = chip.size().height - 1; i >= 0; i--) {
		black_pixels = 0;
		for (int j = 0; j < chip.size().width; j++) {
			//noise
			if ((int)chip.at<uchar>(i, j) == 0) {
				black_pixels++;
			}
		}
		double ratio = black_pixels * 1.0 / chip.size().width;
		if (ratio < thre_upper && ratio > thre_lower) {
			pos_bot = i;
			break;
		}
	}

	// left
	black_pixels = 0;
	int pos_left = 0;
	for (int i = 0; i < chip.size().width; i++) {
		black_pixels = 0;
		for (int j = 0; j < chip.size().height; j++) {
			//noise
			if ((int)chip.at<uchar>(j, i) == 0) {
				black_pixels++;
			}
		}
		double ratio = black_pixels * 1.0 / chip.size().height;
		if (ratio < thre_upper && ratio > thre_lower) {
			pos_left = i;
			break;
		}
	}
	// right
	black_pixels = 0;
	int pos_right = 0;
	for (int i = chip.size().width - 1; i >= 0; i--) {
		black_pixels = 0;
		for (int j = 0; j < chip.size().height; j++) {
			//noise
			if ((int)chip.at<uchar>(j, i) == 0) {
				black_pixels++;
			}
		}
		double ratio = black_pixels * 1.0 / chip.size().height;
		if (ratio < thre_upper && ratio > thre_lower) {
			pos_right = i;
			break;
		}
	}
	boundaries[0] = pos_top;
	boundaries[1] = pos_bot;
	boundaries[2] = pos_left;
	boundaries[3] = pos_right;
	return boundaries;
}

void find_spacing(cv::Mat src_img, std::vector<int> &separation_x, std::vector<int> &separation_y, bool bDebug) {
	if (src_img.channels() == 4) {
		separation_x.clear();
		separation_y.clear();
		return;
	}
	if (src_img.channels() == 3) {
		cv::cvtColor(src_img, src_img, CV_BGR2GRAY);
	}
	// horizontal 
	bool bSpacing_pre = false;
	bool bSpacing_curr = false;
	for (int i = 0; i < src_img.size().width; i++) {
		bSpacing_curr = true;
		for (int j = 0; j < src_img.size().height; j++) {
			//noise
			if (src_img.channels() == 1) {
				if ((int)src_img.at<uchar>(j, i) == 0) {
					bSpacing_curr = false;
					break;
				}
			}
			else {
				if (src_img.at<cv::Vec3b>(j, i)[0] == 0 && src_img.at<cv::Vec3b>(j, i)[1] == 0 && src_img.at<cv::Vec3b>(j, i)[1] == 0) {
					bSpacing_curr = false;
					break;
				}
			}
		}
		if (bSpacing_pre != bSpacing_curr) {
			separation_x.push_back(i);
		}
		bSpacing_pre = bSpacing_curr;
	}

	bSpacing_pre = false;
	bSpacing_curr = false;
	int spacing_y = -1;
	for (int i = 0; i < src_img.size().height; i++) {
		bSpacing_curr = true;
		for (int j = 0; j < src_img.size().width; j++) {
			if (src_img.channels() == 1) {
				if ((int)src_img.at<uchar>(i, j) == 0) {
					bSpacing_curr = false;
					break;
				}
			}
			else {
				if (src_img.at<cv::Vec3b>(i, j)[0] == 0 && src_img.at<cv::Vec3b>(i, j)[1] == 0 && src_img.at<cv::Vec3b>(i, j)[1] == 0) {
					bSpacing_curr = false;
					break;
				}
			}
		}
		if (bSpacing_pre != bSpacing_curr) {
			separation_y.push_back(i);
		}
		bSpacing_pre = bSpacing_curr;
	}
	if (bDebug) {
		std::cout << "separation_x is " << separation_x << std::endl;
		std::cout << "separation_y is " << separation_y << std::endl;
	}
	return;
}

void apply_segmentation_model(cv::Mat &croppedImage, cv::Mat &chip_seg, ModelInfo& mi, bool bDebug) {
	int run_times = 3;
	cv::Mat src_img = croppedImage.clone();
	// scale to seg size
	cv::resize(src_img, src_img, cv::Size(mi.segImageSize[0], mi.segImageSize[1]));
	cv::Mat dnn_img_rgb;
	cv::cvtColor(src_img, dnn_img_rgb, CV_BGR2RGB);
	cv::Mat img_float;
	dnn_img_rgb.convertTo(img_float, CV_32F, 1.0 / 255);
	auto img_tensor = torch::from_blob(img_float.data, { 1, (int)mi.segImageSize[0], (int)mi.segImageSize[1], 3 }).to(torch::kCUDA);
	img_tensor = img_tensor.permute({ 0, 3, 1, 2 });
	img_tensor[0][0] = img_tensor[0][0].sub(0.5).div(0.5);
	img_tensor[0][1] = img_tensor[0][1].sub(0.5).div(0.5);
	img_tensor[0][2] = img_tensor[0][2].sub(0.5).div(0.5);

	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(img_tensor);
	std::vector<std::vector<int>> color_mark;
	color_mark.resize((int)mi.segImageSize[1]);
	for (int i = 0; i < color_mark.size(); i++) {
		color_mark[i].resize((int)mi.segImageSize[0]);
		for (int j = 0; j < color_mark[i].size(); j++) {
			color_mark[i][j] = 0;
		}
	}
	// run three times
	for (int i = 0; i < run_times; i++) {
		torch::Tensor out_tensor = mi.seg_module->forward(inputs).toTensor();
		out_tensor = out_tensor.squeeze().detach().permute({ 1,2,0 });
		out_tensor = out_tensor.add(1).mul(0.5 * 255).clamp(0, 255).to(torch::kU8);
		//out_tensor = out_tensor.mul(255).clamp(0, 255).to(torch::kU8);
		out_tensor = out_tensor.to(torch::kCPU);
		cv::Mat resultImg((int)mi.segImageSize[0], (int)mi.segImageSize[1], CV_8UC3);
		std::memcpy((void*)resultImg.data, out_tensor.data_ptr(), sizeof(torch::kU8)*out_tensor.numel());
		// gray img
		// correct the color
		for (int i = 0; i < resultImg.size().height; i++) {
			for (int j = 0; j < resultImg.size().width; j++) {
				if (resultImg.at<cv::Vec3b>(i, j)[0] > 160)
					color_mark[i][j] += 0;
				else
					color_mark[i][j] += 1;
			}
		}
		/*if (bDebug) {
		cv::cvtColor(resultImg, resultImg, CV_RGB2BGR);
		cv::imwrite("../data/test/seg_" + to_string(i) + ".png", resultImg);
		}*/
	}
	cv::Mat gray_img((int)mi.segImageSize[0], (int)mi.segImageSize[1], CV_8UC1);
	int num_majority = ceil(0.5 * run_times);
	for (int i = 0; i < color_mark.size(); i++) {
		for (int j = 0; j < color_mark[i].size(); j++) {
			if (color_mark[i][j] < num_majority)
				gray_img.at<uchar>(i, j) = (uchar)0;
			else
				gray_img.at<uchar>(i, j) = (uchar)255;
		}
	}
	// scale to grammar size
	cv::resize(gray_img, chip_seg, croppedImage.size());
	// correct the color
	for (int i = 0; i < chip_seg.size().height; i++) {
		for (int j = 0; j < chip_seg.size().width; j++) {
			//noise
			if ((int)chip_seg.at<uchar>(i, j) < 128) {
				chip_seg.at<uchar>(i, j) = (uchar)0;
			}
			else
				chip_seg.at<uchar>(i, j) = (uchar)255;
		}
	}
	if (bDebug) {
		std::cout << "num_majority is " << num_majority << std::endl;
	}
}

bool process_chip(ChipInfo &chip, ModelInfo& mi, bool bDebug) {
	// default size for NN
	int width = mi.defaultSize[0] - 2 * mi.paddingSize[0];
	int height = mi.defaultSize[1] - 2 * mi.paddingSize[1];
	cv::Scalar bg_color(255, 255, 255); // white back ground
	cv::Scalar window_color(0, 0, 0); // black for windows
	cv::Mat scale_img;
	cv::resize(chip.seg_image, scale_img, cv::Size(width, height));
	// correct the color
	for (int i = 0; i < scale_img.size().height; i++) {
		for (int j = 0; j < scale_img.size().width; j++) {
			//noise
			if ((int)scale_img.at<uchar>(i, j) < 128) {
				scale_img.at<uchar>(i, j) = (uchar)0;
			}
			else
				scale_img.at<uchar>(i, j) = (uchar)255;
		}
	}
	// dilate to remove noises
	int dilation_type = cv::MORPH_RECT;
	cv::Mat dilation_dst;
	int kernel_size = 3;
	cv::Mat element = cv::getStructuringElement(dilation_type, cv::Size(kernel_size, kernel_size), cv::Point(kernel_size / 2, kernel_size / 2));
	/// Apply the dilation operation
	cv::dilate(scale_img, dilation_dst, element);
	// alignment
	cv::Mat aligned_img = deSkewImg(dilation_dst);
	// add padding
	int padding_size = mi.paddingSize[0];
	int borderType = cv::BORDER_CONSTANT;
	cv::Mat aligned_img_padding;
	cv::copyMakeBorder(aligned_img, aligned_img_padding, padding_size, padding_size, padding_size, padding_size, borderType, bg_color);

	// find contours
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(aligned_img_padding, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	std::vector<cv::Rect> boundRect(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		boundRect[i] = cv::boundingRect(cv::Mat(contours[i]));
	}
	//
	cv::Mat dnn_img = cv::Mat(aligned_img_padding.size(), CV_8UC3, bg_color);
	for (int i = 1; i< contours.size(); i++)
	{
		if (hierarchy[i][3] != 0) continue;
		cv::rectangle(dnn_img, cv::Point(boundRect[i].tl().x, boundRect[i].tl().y), cv::Point(boundRect[i].br().x, boundRect[i].br().y), window_color, -1);
	}
	chip.dilation_dst = dilation_dst;
	chip.aligned_img = aligned_img;
	chip.dnnIn_image = dnn_img;
	return true;
}

void feedDnn(ChipInfo &chip, FacadeInfo& fi, ModelInfo& mi, bool bDebug) {
	int num_classes = mi.number_grammars;
	// 
	std::vector<int> separation_x;
	std::vector<int> separation_y;
	cv::Mat spacing_img = chip.dnnIn_image.clone();
	find_spacing(spacing_img, separation_x, separation_y, bDebug);
	int spacing_r = separation_y.size() / 2;
	int spacing_c = separation_x.size() / 2;

	cv::Mat dnn_img_rgb;
	cv::cvtColor(chip.dnnIn_image.clone(), dnn_img_rgb, CV_BGR2RGB);
	cv::Mat img_float;
	dnn_img_rgb.convertTo(img_float, CV_32F, 1.0 / 255);
	auto img_tensor = torch::from_blob(img_float.data, { 1, 224, 224, 3 }).to(torch::kCUDA);
	img_tensor = img_tensor.permute({ 0, 3, 1, 2 });
	img_tensor[0][0] = img_tensor[0][0].sub(0.485).div(0.229);
	img_tensor[0][1] = img_tensor[0][1].sub(0.456).div(0.224);
	img_tensor[0][2] = img_tensor[0][2].sub(0.406).div(0.225);

	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(img_tensor);

	int best_class = -1;
	std::vector<double> confidence_values;
	confidence_values.resize(num_classes);
	if (true)
	{
		// Deserialize the ScriptModule from a file using torch::jit::load().
		torch::Tensor out_tensor = mi.classifier_module->forward(inputs).toTensor();
		//std::cout << out_tensor.slice(1, 0, num_classes) << std::endl;

		torch::Tensor confidences_tensor = torch::softmax(out_tensor, 1);
		//std::cout << confidences_tensor.slice(1, 0, num_classes) << std::endl;

		double best_score = 0;
		for (int i = 0; i < num_classes; i++) {
			double tmp = confidences_tensor.slice(1, i, i + 1).item<float>();
			confidence_values[i] = tmp;
			if (tmp > best_score) {
				best_score = tmp;
				best_class = i;
			}
		}
		best_class = best_class + 1;
		if (bDebug) {
			std::cout << confidences_tensor.slice(1, 0, num_classes) << std::endl;
			std::cout << "DNN class is " << best_class << std::endl;
		}
	}
	// adjust the best_class
	if (!fi.inscGround) {
		if (best_class % 2 == 0)
			best_class = best_class - 1;
	}
	// choose conresponding estimation DNN
	// number of paras
	int num_paras = mi.grammars[best_class - 1].number_paras;

	torch::Tensor out_tensor_grammar = mi.grammars[best_class - 1].grammar_model->forward(inputs).toTensor();
	//std::cout << out_tensor_grammar.slice(1, 0, num_paras) << std::endl;
	std::vector<double> paras;
	for (int i = 0; i < num_paras; i++) {
		paras.push_back(out_tensor_grammar.slice(1, i, i + 1).item<float>());
	}
	for (int i = 0; i < num_paras; i++) {
		if (paras[i] < 0)
			paras[i] = 0;
	}

	std::vector<double> predictions;
	if (best_class == 1) {
		predictions = grammar1(mi, paras, bDebug);
	}
	else if (best_class == 2) {
		predictions = grammar2(mi, paras, bDebug);
	}
	else if (best_class == 3) {
		predictions = grammar3(mi, paras, bDebug);
	}
	else if (best_class == 4) {
		predictions = grammar4(mi, paras, bDebug);
	}
	else if (best_class == 5) {
		predictions = grammar5(mi, paras, bDebug);
	}
	else if (best_class == 6) {
		predictions = grammar6(mi, paras, bDebug);
	}
	else {
		//do nothing
		predictions = grammar1(mi, paras, bDebug);
	}
	if (best_class % 2 == 0) {
		if (abs(predictions[0] + 1 - spacing_r) <= 1 && predictions[0] > 1)
			predictions[0] = spacing_r - 1;
	}
	else {
		if (abs(predictions[0] - spacing_r) <= 1 && predictions[0] > 1)
			predictions[0] = spacing_r;
	}
	if (abs(predictions[1] - spacing_c) <= 1 && predictions[1] > 1)
		predictions[1] = spacing_c;
	// write back to fi
	for (int i = 0; i < num_classes; i++)
		fi.conf[i] = confidence_values[i];

	fi.grammar = best_class;
	if (predictions.size() == 5) {
		int img_rows = predictions[0];
		int img_cols = predictions[1];
		int img_groups = predictions[2];
		double relative_width = predictions[3];
		double relative_height = predictions[4];

		fi.rows = img_rows;
		fi.cols = img_cols;
		fi.grouping = img_groups;
		fi.relativeWidth = relative_width;
		fi.relativeHeight = relative_height;
	}
	if (predictions.size() == 8) {
		int img_rows = predictions[0];
		int img_cols = predictions[1];
		int img_groups = predictions[2];
		int img_doors = predictions[3];
		double relative_width = predictions[4];
		double relative_height = predictions[5];
		double relative_door_width = predictions[6];
		double relative_door_height = predictions[7];

		fi.rows = img_rows;
		fi.cols = img_cols;
		fi.grouping = img_groups;
		fi.doors = img_doors;
		fi.relativeWidth = relative_width;
		fi.relativeHeight = relative_height;
		fi.relativeDWidth = relative_door_width;
		fi.relativeDHeight = relative_door_height;

	}
}

cv::Mat deSkewImg(cv::Mat src_img) {
	// generate input image for DNN
	cv::Scalar bg_color(255, 255, 255); // white back ground
	cv::Scalar window_color(0, 0, 0); // black for windows
									  // add padding
	int padding_size = 5;
	int borderType = cv::BORDER_CONSTANT;
	cv::Scalar value(255, 255, 255);
	cv::Mat img_padding;
	cv::copyMakeBorder(src_img, img_padding, padding_size, padding_size, padding_size, padding_size, borderType, value);
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(img_padding, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	if (contours.size() <= 3)
	{
		contours.clear();
		hierarchy.clear();
		int clear_border = 3;
		cv::Mat tmp = src_img(cv::Rect(clear_border, clear_border, src_img.size().width - 2 * clear_border, src_img.size().width - 2 * clear_border)).clone();
		cv::Mat tmp_img_padding;
		cv::copyMakeBorder(tmp, tmp_img_padding, padding_size + clear_border, padding_size + clear_border, padding_size + clear_border, padding_size + clear_border, borderType, bg_color);
		cv::findContours(tmp_img_padding, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	}
	std::vector<cv::RotatedRect> minRect(contours.size());
	std::vector<cv::Moments> mu(contours.size());
	std::vector<cv::Point2f> mc(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		minRect[i] = minAreaRect(cv::Mat(contours[i]));
		mu[i] = moments(contours[i], false);
		mc[i] = cv::Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	}

	cv::Mat drawing = img_padding.clone();
	//first step
	for (int i = 0; i< contours.size(); i++)
	{
		if (hierarchy[i][3] != 0) continue;
		float angle = minRect[i].angle;
		if (angle < 0) {
			// draw RotatedRect
			cv::Point2f vertices2f[4];
			minRect[i].points(vertices2f);
			// Convert them so we can use them in a fillConvexPoly
			std::vector<cv::Point> points;
			for (int i = 0; i < 4; ++i) {
				points.push_back(vertices2f[i]);
			}
			cv::fillConvexPoly(drawing, points, bg_color, 8, 0);
		}
	}
	// second step
	for (int i = 0; i< contours.size(); i++)
	{
		if (hierarchy[i][3] != 0) continue;
		float angle = minRect[i].angle;
		if (angle < 0) {
			if (angle < -45.)
				angle += 90.;
			cv::Mat tmp(img_padding.size(), CV_8UC3, bg_color);
			drawContours(tmp, contours, i, window_color, -1, 8, hierarchy, 0, cv::Point());
			// rotate the contour
			cv::Mat tmp_gray;
			cvtColor(tmp, tmp_gray, cv::COLOR_BGR2GRAY);
			cv::Mat aligned_tmp;
			cv::Mat rot_mat = cv::getRotationMatrix2D(mc[i], angle, 1.0);
			cv::warpAffine(tmp_gray, aligned_tmp, rot_mat, tmp_gray.size(), cv::INTER_CUBIC, cv::BORDER_REPLICATE);
			tmp_gray.empty();
			tmp_gray = aligned_tmp.clone();
			// copy to the output img
			for (int m = 0; m < drawing.size().height; m++) {
				for (int n = 0; n < drawing.size().width; n++) {
					if ((int)tmp_gray.at<uchar>(m, n) < 128 && (int)drawing.at<uchar>(m, n) == 255) {
						drawing.at<uchar>(m, n) = (uchar)0;
					}
					else if ((int)tmp_gray.at<uchar>(m, n) < 128 && (int)drawing.at<uchar>(m, n) == 0) {
						drawing.at<uchar>(m, n) = (uchar)255;
					}
					else {

					}
				}
			}
		}
	}
	drawing = drawing(cv::Rect(padding_size, padding_size, src_img.size().width, src_img.size().height));
	cv::Mat aligned_img = cleanAlignedImage(drawing, 0.10);
	return aligned_img;
}

cv::Mat cleanAlignedImage(cv::Mat src, float threshold) {
	// horz
	int count = 0;
	cv::Mat result = src.clone();
	for (int i = 0; i < src.size().height; i++) {
		count = 0;
		for (int j = 0; j < src.size().width; j++) {
			//noise
			if ((int)src.at<uchar>(i, j) == 0) {
				count++;
			}
		}
		if (count * 1.0 / src.size().width < threshold) {
			for (int j = 0; j < src.size().width; j++) {
				result.at<uchar>(i, j) = (uchar)255;
			}
		}

	}
	// vertical
	for (int i = 0; i < src.size().width; i++) {
		count = 0;
		for (int j = 0; j < src.size().height; j++) {
			//noise
			if ((int)src.at<uchar>(j, i) == 0) {
				count++;
			}
		}
		if (count * 1.0 / src.size().height < threshold) {
			for (int j = 0; j < src.size().height; j++) {
				result.at<uchar>(j, i) = (uchar)255;
			}
		}

	}
	return result;
}

std::vector<double> grammar1(ModelInfo& mi, std::vector<double> paras, bool bDebug) {
	// range of Rows
	std::pair<int, int> imageRows(mi.grammars[0].rangeOfRows[0], mi.grammars[0].rangeOfRows[1]);
	// range of Cols
	std::pair<int, int> imageCols(mi.grammars[0].rangeOfCols[0], mi.grammars[0].rangeOfCols[1]);
	// relativeWidth
	std::pair<double, double> imageRelativeWidth(mi.grammars[0].relativeWidth[0], mi.grammars[0].relativeWidth[1]);
	// relativeHeight
	std::pair<double, double> imageRelativeHeight(mi.grammars[0].relativeHeight[0], mi.grammars[0].relativeHeight[1]);
	int img_rows = paras[0] * (imageRows.second - imageRows.first) + imageRows.first;
	if (paras[0] * (imageRows.second - imageRows.first) + imageRows.first - img_rows > 0.8)
		img_rows++;
	int img_cols = paras[1] * (imageCols.second - imageCols.first) + imageCols.first;
	if (paras[1] * (imageCols.second - imageCols.first) + imageCols.first - img_cols > 0.8)
		img_cols++;
	int img_groups = 1;
	double relative_width = paras[2] * (imageRelativeWidth.second - imageRelativeWidth.first) + imageRelativeWidth.first;
	double relative_height = paras[3] * (imageRelativeHeight.second - imageRelativeHeight.first) + imageRelativeHeight.first;
	if (bDebug) {
		std::cout << "paras[0] is " << paras[0] << std::endl;
		std::cout << "paras[1] is " << paras[1] << std::endl;
		std::cout << "img_rows is " << paras[0] * (imageRows.second - imageRows.first) + imageRows.first << std::endl;
		std::cout << "img_cols is " << paras[1] * (imageCols.second - imageCols.first) + imageCols.first << std::endl;
	}
	std::vector<double> results;
	results.push_back(img_rows);
	results.push_back(img_cols);
	results.push_back(img_groups);
	results.push_back(relative_width);
	results.push_back(relative_height);
	return results;
}

std::vector<double> grammar2(ModelInfo& mi, std::vector<double> paras, bool bDebug) {
	// range of Rows
	std::pair<int, int> imageRows(mi.grammars[1].rangeOfRows[0], mi.grammars[1].rangeOfRows[1]);
	// range of Cols
	std::pair<int, int> imageCols(mi.grammars[1].rangeOfCols[0], mi.grammars[1].rangeOfCols[1]);
	// range of Doors
	std::pair<int, int> imageDoors(mi.grammars[1].rangeOfDoors[0], mi.grammars[1].rangeOfDoors[1]);
	// relativeWidth
	std::pair<double, double> imageRelativeWidth(mi.grammars[1].relativeWidth[0], mi.grammars[1].relativeWidth[1]);
	// relativeHeight
	std::pair<double, double> imageRelativeHeight(mi.grammars[1].relativeHeight[0], mi.grammars[1].relativeHeight[1]);
	// relativeDWidth
	std::pair<double, double> imageDRelativeWidth(mi.grammars[1].relativeDWidth[0], mi.grammars[1].relativeDWidth[1]);
	// relativeDHeight
	std::pair<double, double> imageDRelativeHeight(mi.grammars[1].relativeDHeight[0], mi.grammars[1].relativeDHeight[1]);
	int img_rows = paras[0] * (imageRows.second - imageRows.first) + imageRows.first;
	if (paras[0] * (imageRows.second - imageRows.first) + imageRows.first - img_rows > 0.8)
		img_rows++;
	int img_cols = paras[1] * (imageCols.second - imageCols.first) + imageCols.first;
	if (paras[1] * (imageCols.second - imageCols.first) + imageCols.first - img_cols > 0.8)
		img_cols++;
	int img_groups = 1;
	int img_doors = paras[2] * (imageDoors.second - imageDoors.first) + imageDoors.first;
	if (paras[2] * (imageDoors.second - imageDoors.first) + imageDoors.first - img_doors > 0.8)
		img_doors++;
	double relative_width = paras[3] * (imageRelativeWidth.second - imageRelativeWidth.first) + imageRelativeWidth.first;
	double relative_height = paras[4] * (imageRelativeHeight.second - imageRelativeHeight.first) + imageRelativeHeight.first;
	double relative_door_width = paras[5] * (imageDRelativeWidth.second - imageDRelativeWidth.first) + imageDRelativeWidth.first;
	double relative_door_height = paras[6] * (imageDRelativeHeight.second - imageDRelativeHeight.first) + imageDRelativeHeight.first;
	std::vector<double> results;
	results.push_back(img_rows);
	results.push_back(img_cols);
	results.push_back(img_groups);
	results.push_back(img_doors);
	results.push_back(relative_width);
	results.push_back(relative_height);
	results.push_back(relative_door_width);
	results.push_back(relative_door_height);
	return results;
}

std::vector<double> grammar3(ModelInfo& mi, std::vector<double> paras, bool bDebug) {
	// range of Cols
	std::pair<int, int> imageCols(mi.grammars[2].rangeOfCols[0], mi.grammars[2].rangeOfCols[1]);
	// relativeWidth
	std::pair<double, double> imageRelativeWidth(mi.grammars[2].relativeWidth[0], mi.grammars[2].relativeWidth[1]);
	int img_rows = 1;
	int img_cols = paras[0] * (imageCols.second - imageCols.first) + imageCols.first;
	if (paras[0] * (imageCols.second - imageCols.first) + imageCols.first - img_cols > 0.8)
		img_cols++;
	int img_groups = 1;
	double relative_width = paras[1] * (imageRelativeWidth.second - imageRelativeWidth.first) + imageRelativeWidth.first;
	double relative_height = 1.0;
	std::vector<double> results;
	results.push_back(img_rows);
	results.push_back(img_cols);
	results.push_back(img_groups);
	results.push_back(relative_width);
	results.push_back(relative_height);
	return results;
}

std::vector<double> grammar4(ModelInfo& mi, std::vector<double> paras, bool bDebug) {
	// range of Rows
	std::pair<int, int> imageCols(mi.grammars[3].rangeOfCols[0], mi.grammars[3].rangeOfCols[1]);
	// range of Doors
	std::pair<int, int> imageDoors(mi.grammars[3].rangeOfDoors[0], mi.grammars[3].rangeOfDoors[1]);
	// relativeWidth
	std::pair<double, double> imageRelativeWidth(mi.grammars[3].relativeWidth[0], mi.grammars[3].relativeWidth[1]);
	// relativeDWidth
	std::pair<double, double> imageDRelativeWidth(mi.grammars[3].relativeDWidth[0], mi.grammars[3].relativeDWidth[1]);
	// relativeDHeight
	std::pair<double, double> imageDRelativeHeight(mi.grammars[3].relativeDHeight[0], mi.grammars[3].relativeDHeight[1]);
	int img_rows = 1;;
	int img_cols = paras[0] * (imageCols.second - imageCols.first) + imageCols.first;
	if (paras[0] * (imageCols.second - imageCols.first) + imageCols.first - img_cols > 0.8)
		img_cols++;
	int img_groups = 1;
	int img_doors = paras[1] * (imageDoors.second - imageDoors.first) + imageDoors.first;
	if (paras[1] * (imageDoors.second - imageDoors.first) + imageDoors.first - img_doors > 0.8)
		img_doors++;
	double relative_width = paras[2] * (imageRelativeWidth.second - imageRelativeWidth.first) + imageRelativeWidth.first;
	double relative_height = 1.0;
	double relative_door_width = paras[3] * (imageDRelativeWidth.second - imageDRelativeWidth.first) + imageDRelativeWidth.first;
	double relative_door_height = paras[4] * (imageDRelativeHeight.second - imageDRelativeHeight.first) + imageDRelativeHeight.first;
	std::vector<double> results;
	results.push_back(img_rows);
	results.push_back(img_cols);
	results.push_back(img_groups);
	results.push_back(img_doors);
	results.push_back(relative_width);
	results.push_back(relative_height);
	results.push_back(relative_door_width);
	results.push_back(relative_door_height);
	return results;
}

std::vector<double> grammar5(ModelInfo& mi, std::vector<double> paras, bool bDebug) {
	// range of Rows
	std::pair<int, int> imageRows(mi.grammars[4].rangeOfRows[0], mi.grammars[4].rangeOfRows[1]);
	// relativeHeight
	std::pair<double, double> imageRelativeHeight(mi.grammars[4].relativeHeight[0], mi.grammars[4].relativeHeight[1]);
	int img_rows = paras[0] * (imageRows.second - imageRows.first) + imageRows.first;
	if (paras[0] * (imageRows.second - imageRows.first) + imageRows.first - img_rows > 0.8)
		img_rows++;
	int img_cols = 1;
	int img_groups = 1;
	double relative_width = 1.0;
	double relative_height = paras[1] * (imageRelativeHeight.second - imageRelativeHeight.first) + imageRelativeHeight.first;
	std::vector<double> results;
	results.push_back(img_rows);
	results.push_back(img_cols);
	results.push_back(img_groups);
	results.push_back(relative_width);
	results.push_back(relative_height);
	return results;
}

std::vector<double> grammar6(ModelInfo& mi, std::vector<double> paras, bool bDebug) {
	// range of Rows
	std::pair<int, int> imageRows(mi.grammars[5].rangeOfRows[0], mi.grammars[5].rangeOfRows[1]);
	// range of Doors
	std::pair<int, int> imageDoors(mi.grammars[5].rangeOfDoors[0], mi.grammars[5].rangeOfDoors[1]);
	// relativeHeight
	std::pair<double, double> imageRelativeHeight(mi.grammars[5].relativeHeight[0], mi.grammars[5].relativeHeight[1]);
	// relativeDWidth
	std::pair<double, double> imageDRelativeWidth(mi.grammars[5].relativeDWidth[0], mi.grammars[5].relativeDWidth[1]);
	// relativeDHeight
	std::pair<double, double> imageDRelativeHeight(mi.grammars[5].relativeDHeight[0], mi.grammars[5].relativeDHeight[1]);
	int img_rows = paras[0] * (imageRows.second - imageRows.first) + imageRows.first;
	if (paras[0] * (imageRows.second - imageRows.first) + imageRows.first - img_rows > 0.8)
		img_rows++;
	int img_cols = 1;
	int img_groups = 1;
	int img_doors = paras[1] * (imageDoors.second - imageDoors.first) + imageDoors.first;
	if (paras[1] * (imageDoors.second - imageDoors.first) + imageDoors.first - img_doors > 0.8)
		img_doors++;
	double relative_width = 1.0;
	double relative_height = paras[2] * (imageRelativeHeight.second - imageRelativeHeight.first) + imageRelativeHeight.first;
	double relative_door_width = paras[3] * (imageDRelativeWidth.second - imageDRelativeWidth.first) + imageDRelativeWidth.first;
	double relative_door_height = paras[4] * (imageDRelativeHeight.second - imageDRelativeHeight.first) + imageDRelativeHeight.first;
	std::vector<double> results;
	results.push_back(img_rows);
	results.push_back(img_cols);
	results.push_back(img_groups);
	results.push_back(img_doors);
	results.push_back(relative_width);
	results.push_back(relative_height);
	results.push_back(relative_door_width);
	results.push_back(relative_door_height);
	return results;
}
