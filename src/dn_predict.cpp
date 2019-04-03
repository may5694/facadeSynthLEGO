#include "dn_predict.hpp"

#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include <stack>
#include <dlib/clustering.h>
#include <dlib/rand.h>
using namespace dlib;

void dn_predict(FacadeInfo& fi, std::string modeljson) {
	cv::Mat croppedImage;
	bool bvalid = chipping(fi, modeljson, croppedImage, true, false);
	if (bvalid) {
		cv::Mat dnn_img;
		segment_chip(croppedImage, dnn_img, fi, modeljson, false);
		feedDnn(dnn_img, fi, modeljson, false);
	}
}

double readNumber(const rapidjson::Value& node, const char* key, double default_value) {
	if (node.HasMember(key) && node[key].IsDouble()) {
		return node[key].GetDouble();
	}
	else if (node.HasMember(key) && node[key].IsInt()) {
		return node[key].GetInt();
	}
	else {
		return default_value;
	}
}

std::vector<double> read1DArray(const rapidjson::Value& node, const char* key) {
	std::vector<double> array_values;
	if (node.HasMember(key)) {
		const rapidjson::Value& data = node[key];
		array_values.resize(data.Size());
		for (int i = 0; i < data.Size(); i++)
			array_values[i] = data[i].GetDouble();
		return array_values;
	}
	else {
		return array_values;
	}
}

bool readBoolValue(const rapidjson::Value& node, const char* key, bool default_value) {
	if (node.HasMember(key) && node[key].IsBool()) {
		return node[key].GetBool();
	}
	else {
		return default_value;
	}
}

std::string readStringValue(const rapidjson::Value& node, const char* key) {
	if (node.HasMember(key) && node[key].IsString()) {
		return node[key].GetString();
	}
	else {
		throw "Could not read string from node";
	}
}

cv::Mat facade_clustering_kkmeans(cv::Mat src_img, int clusters) {
	// Here we declare that our samples will be 2 dimensional column vectors.  
	// (Note that if you don't know the dimensionality of your vectors at compile time
	// you can change the 2 to a 0 and then set the size at runtime)
	typedef matrix<double, 0, 1> sample_type;
	// Now we are making a typedef for the kind of kernel we want to use.  I picked the
	// radial basis kernel because it only has one parameter and generally gives good
	// results without much fiddling.
	typedef radial_basis_kernel<sample_type> kernel_type;


	// Here we declare an instance of the kcentroid object.  It is the object used to 
	// represent each of the centers used for clustering.  The kcentroid has 3 parameters 
	// you need to set.  The first argument to the constructor is the kernel we wish to 
	// use.  The second is a parameter that determines the numerical accuracy with which 
	// the object will perform part of the learning algorithm.  Generally, smaller values 
	// give better results but cause the algorithm to attempt to use more dictionary vectors 
	// (and thus run slower and use more memory).  The third argument, however, is the 
	// maximum number of dictionary vectors a kcentroid is allowed to use.  So you can use
	// it to control the runtime complexity.  
	kcentroid<kernel_type> kc(kernel_type(0.1), 0.01, 16);

	// Now we make an instance of the kkmeans object and tell it to use kcentroid objects
	// that are configured with the parameters from the kc object we defined above.
	kkmeans<kernel_type> test(kc);

	std::vector<sample_type> samples;
	std::vector<sample_type> initial_centers;

	sample_type m(src_img.channels());

	for (int i = 0; i < src_img.size().height; i++) {
		for (int j = 0; j < src_img.size().width; j++) {
			if (src_img.channels() == 4) {
				m(0) = src_img.at<cv::Vec4b>(i, j)[0] * 1.0 / 255;
				m(1) = src_img.at<cv::Vec4b>(i, j)[1] * 1.0 / 255;
				m(2) = src_img.at<cv::Vec4b>(i, j)[2] * 1.0 / 255;
			}
			else {
				m(0) = (int)src_img.at<uchar>(i, j) * 1.0 / 255;
			}
			// add this sample to our set of samples we will run k-means 
			samples.push_back(m);
		}
	}

	// tell the kkmeans object we made that we want to run k-means with k set to 3. 
	// (i.e. we want 3 clusters)
	test.set_number_of_centers(clusters);

	// You need to pick some initial centers for the k-means algorithm.  So here
	// we will use the dlib::pick_initial_centers() function which tries to find
	// n points that are far apart (basically).  
	pick_initial_centers(clusters, initial_centers, samples, test.get_kernel());

	// now run the k-means algorithm on our set of samples.  
	test.train(samples, initial_centers);

	std::vector<cv::Scalar> clusters_colors;
	std::vector<int> clusters_points;
	clusters_colors.resize(clusters);
	clusters_points.resize(clusters);
	for (int i = 0; i < clusters; i++) {
		clusters_colors[i] = cv::Scalar(0, 0, 0);
		clusters_points[i] = 0;
	}
	int count = 0;
	// 
	if (src_img.channels() == 4) {
		count = 0;
		for (int i = 0; i < src_img.size().height; i++) {
			for (int j = 0; j < src_img.size().width; j++) {
				clusters_colors[test(samples[count])][0] += src_img.at<cv::Vec4b>(i, j)[0];
				clusters_colors[test(samples[count])][1] += src_img.at<cv::Vec4b>(i, j)[1];
				clusters_colors[test(samples[count])][2] += src_img.at<cv::Vec4b>(i, j)[2];
				clusters_points[test(samples[count])] ++;
				count++;
			}
		}
		for (int i = 0; i < clusters; i++) {
			clusters_colors[i][0] = clusters_colors[i][0] / clusters_points[i];
			clusters_colors[i][1] = clusters_colors[i][1] / clusters_points[i];
			clusters_colors[i][2] = clusters_colors[i][2] / clusters_points[i];
		}
	}
	else if (src_img.channels() == 1) { //gray image
		int count = 0;
		for (int i = 0; i < src_img.size().height; i++) {
			for (int j = 0; j < src_img.size().width; j++) {
				clusters_colors[test(samples[count])][0] += (int)src_img.at<uchar>(i, j);
				clusters_points[test(samples[count])] ++;
				count++;
			}
		}
		for (int i = 0; i < clusters; i++) {
			clusters_colors[i][0] = clusters_colors[i][0] / clusters_points[i];
		}
	}
	else {
		//do nothing
	}
	// compute cluster colors
	int darkest_cluster = -1;
	cv::Scalar darkest_color(255, 255, 255);
	for (int i = 0; i < clusters; i++) {
		//std::cout << "clusters_colors " << i << " is " << clusters_colors[i] << std::endl;
		if (src_img.channels() == 3) {
			if (clusters_colors[i][0] < darkest_color[0] && clusters_colors[i][1] < darkest_color[1] && clusters_colors[i][2] < darkest_color[2]) {
				darkest_color[0] = clusters_colors[i][0];
				darkest_color[1] = clusters_colors[i][1];
				darkest_color[2] = clusters_colors[i][2];
				darkest_cluster = i;
			}
		}
		else {
			if (clusters_colors[i][0] < darkest_color[0]) {
				darkest_color[0] = clusters_colors[i][0];
				darkest_cluster = i;
			}
		}
	}
	cv::Mat out_img;
	cv::resize(src_img, out_img, cv::Size(src_img.size().width, src_img.size().height));
	count = 0;
	if (src_img.channels() == 1) {
		for (int i = 0; i < out_img.size().height; i++) {
			for (int j = 0; j < out_img.size().width; j++) {
				if (test(samples[count]) == darkest_cluster) {
					out_img.at<uchar>(i, j) = (uchar)0;
				}
				else {
					out_img.at<uchar>(i, j) = (uchar)255;

				}
				count++;
			}
		}
	}
	else {
		for (int i = 0; i < out_img.size().height; i++) {
			for (int j = 0; j < out_img.size().width; j++) {
				if (test(samples[count]) == darkest_cluster) {
					out_img.at<cv::Vec4b>(i, j)[0] = 0;
					out_img.at<cv::Vec4b>(i, j)[1] = 0;
					out_img.at<cv::Vec4b>(i, j)[2] = 0;
				}
				else {
					out_img.at<cv::Vec4b>(i, j)[0] = 255;
					out_img.at<cv::Vec4b>(i, j)[1] = 255;
					out_img.at<cv::Vec4b>(i, j)[2] = 255;

				}
				count++;
			}
		}
	}
	return out_img;
}

bool chipping(FacadeInfo& fi, std::string modeljson, cv::Mat& croppedImage, bool bMultipleChips, bool bDebug) {
	// size of chip
	std::vector<double> facChip_size = { fi.inscSize_utm.x, fi.inscSize_utm.y };
	// roof 
	bool broof = fi.roof;
	// ground
	bool bground = fi.inscGround;
	// score
	double score = fi.score;

	// first decide whether it's a valid chip
	bool bvalid = false;
	int type = 0;
	if (!broof && facChip_size[0] < 30.0 && facChip_size[0] > 15.0 && facChip_size[1] < 30.0 && facChip_size[1] > 15.0 && score > 0.94) {
		type = 1;
		bvalid = true;
	}
	else if (!broof && facChip_size[0] > 30.0 && facChip_size[1] < 30.0 && facChip_size[1] > 12.0 && score > 0.65) {
		type = 2;
		bvalid = true;
	}
	else if (!broof && facChip_size[0] < 30.0 && facChip_size[0] > 12.0 && facChip_size[1] > 30.0 && score > 0.65) {
		type = 3;
		bvalid = true;
	}
	else if (!broof && facChip_size[0] > 30.0 && facChip_size[1] > 30.0 && score > 0.67) {
		type = 4;
		bvalid = true;
	}
	else {
		// do nothing
	}
	// one more check
	if (bvalid) {
		bvalid = checkFacade(fi);
	}
	if (!bvalid) {
		saveInvalidFacade(fi, false);
		return false;
	}
	// read model config json file
	FILE* fp = fopen(modeljson.c_str(), "rb"); // non-Windows use "r"
	char readBuffer[10240];
	memset(readBuffer, 0, sizeof(readBuffer));
	rapidjson::FileReadStream isModel(fp, readBuffer, sizeof(readBuffer));
	rapidjson::Document docModel;
	docModel.ParseStream(isModel);

	std::vector<double> tmp_array = read1DArray(docModel, "targetChipSize");
	if (tmp_array.size() != 2) {
		std::cout << "Please check the targetChipSize member in the JSON file" << std::endl;
		return false;
	}
	double target_width = tmp_array[0];
	double target_height = tmp_array[1];

	cv::Mat src_chip = fi.facadeImg(fi.inscRect_px).clone();
	// crop a chip
	std::vector<cv::Mat> cropped_chips = crop_chip(src_chip, modeljson, type, bground, facChip_size, target_width, target_height, bMultipleChips);
	croppedImage = cropped_chips[0];// use the best chip to pass through those testings
	// get confidence value
	// get the number of contours
	// get the max contour area
	std::vector<double> info_facade = compute_confidence(croppedImage, modeljson, bDebug);
	if (bDebug) {
		std::cout << "info_facade confidence is " << info_facade[0] << std::endl;
		std::cout << "info_facade grammar is " << info_facade[1] << std::endl;
		std::cout << "info_facade number of contours is " << info_facade[2] << std::endl;
		std::cout << "info_facade max contour area is " << info_facade[3] << std::endl;
	}
	int grammar_type = (int)info_facade[1];
	if ((info_facade[0] < 0.92 && score < 0.94) || score > 0.994) {
		saveInvalidFacade(fi, bDebug);
		return false;
	}
	else if (info_facade[3] > 0.30) {
		saveInvalidFacade(fi, bDebug);
		return false;
	}
	else if (grammar_type == 1 && info_facade[2] < 9) {
		saveInvalidFacade(fi, bDebug);
		return false;
	}
	else if (grammar_type == 2 && info_facade[2] < 11) {
		saveInvalidFacade(fi, bDebug);
		return false;
	}
	else if (info_facade[2] <= 3) {
		saveInvalidFacade(fi, bDebug);
		return false;
	}
	else {

	}

	fi.valid = bvalid;
	fi.grammar = -1;

	// check wheter there are two chips in the vector
	if (cropped_chips.size() == 2 && grammar_type % 2 != 0) {
		std::vector<double> door_paras = compute_door_paras(cropped_chips[1], modeljson, bDebug);
		if (door_paras.size() == 8) {
			// compute the door height
			int src_width = src_chip.size().width;
			int src_height = src_chip.size().height;
			int door_chip_height = cropped_chips[1].size().height;
			double door_height = door_chip_height * 1.0 / src_height * facChip_size[1] * door_paras[7];
			// add real chip size
			int chip_width = croppedImage.size().width;
			int chip_height = croppedImage.size().height;
			fi.chip_size.x = chip_width * 1.0 / src_width * facChip_size[0];
			fi.chip_size.y = chip_height * 1.0 / src_height * facChip_size[1] + door_height;

			int img_rows = door_paras[0];
			int img_cols = door_paras[1];
			int img_groups = door_paras[2];
			int img_doors = door_paras[3];
			double relative_width = door_paras[4];
			double relative_height = door_paras[5];
			double relative_door_width = door_paras[6];
			double relative_door_height = door_paras[7];

			fi.grammar = 0;
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
	else {
		// add real chip size
		int src_width = src_chip.size().width;
		int src_height = src_chip.size().height;
		int chip_width = croppedImage.size().width;
		int chip_height = croppedImage.size().height;
		fi.chip_size.x = chip_width * 1.0 / src_width * facChip_size[0];
		fi.chip_size.y = chip_height * 1.0 / src_height * facChip_size[1];
	}

	return true;
}

std::vector<cv::Mat> crop_chip(cv::Mat src_chip, std::string modeljson, int type, bool bground, std::vector<double> facChip_size, double target_width, double target_height, bool bMultipleChips) {
	std::vector<cv::Mat> cropped_chips;
	if (type == 1) {
		cropped_chips.push_back(src_chip.clone());
	}
	else if (type == 2) {
		if (facChip_size[0] < 1.6 * target_width || !bMultipleChips) {
			double target_ratio_width = target_width / facChip_size[0];
			double target_ratio_height = target_height / facChip_size[1];
			if (target_ratio_height > 1.0)
				target_ratio_height = 1.0;
			double padding_width_ratio = (1 - target_ratio_width) * 0.5;
			// crop 30 * 30
			cv::Mat tmp = src_chip(cv::Rect(src_chip.size().width * padding_width_ratio, 0, src_chip.size().width * target_ratio_width, src_chip.size().height * target_ratio_height));
			cropped_chips.push_back(adjust_chip(tmp));
		}
		else {
			// check multiple chips and choose the one that has the highest confidence value
			int index = 0;
			double start_width_ratio = index * 0.1;
			double target_ratio_width = target_width / facChip_size[0];
			double target_ratio_height = target_height / facChip_size[1];
			if (target_ratio_height > 1.0)
				target_ratio_height = 1.0;
			std::vector<double> confidences;
			while (start_width_ratio + target_ratio_width < 1.0) {
				// get the cropped img
				cv::Mat tmp = src_chip(cv::Rect(src_chip.size().width * start_width_ratio, 0, src_chip.size().width * target_ratio_width, src_chip.size().height * target_ratio_height));
				cv::Mat croppedImage = adjust_chip(tmp);
				// get confidence value for the cropped img
				double conf_value = compute_confidence(croppedImage, modeljson, false)[0];
				confidences.push_back(conf_value);
				index++;
				start_width_ratio = index * 0.1;
			}
			// find the best img
			double best_conf = 0;
			int best_id = -1;
			for (int i = 0; i < confidences.size(); i++) {
				if (best_conf < confidences[i]) {
					best_conf = confidences[i];
					best_id = i;
				}
			}
			// output best img
			cv::Mat  best_cropped_tmp = src_chip(cv::Rect(src_chip.size().width * best_id * 0.1, 0, src_chip.size().width * target_ratio_width, src_chip.size().height * target_ratio_height));
			cv::Mat  best_cropped = adjust_chip(best_cropped_tmp);
			cropped_chips.push_back(best_cropped);
		}
	}
	else if (type == 3) {
		if (facChip_size[1] < 1.6 * target_width || !bMultipleChips) {
			double target_ratio_height = target_height / facChip_size[1];
			double padding_height_ratio = 0;
			double target_ratio_width = target_width / facChip_size[0];
			if (target_ratio_width >= 1.0)
				target_ratio_width = 1.0;
			if (!bground) {
				padding_height_ratio = (1 - target_ratio_height) * 0.5;
			}
			else {
				padding_height_ratio = (1 - target_ratio_height);
			}
			cv::Mat tmp = src_chip(cv::Rect(0, src_chip.size().height * padding_height_ratio, src_chip.size().width * target_ratio_width, src_chip.size().height * target_ratio_height));
			cropped_chips.push_back(adjust_chip(tmp));
		}
		else {
			// check multiple chips and choose the one that has the highest confidence value
			int index = 0;
			double start_height_ratio = index * 0.1;
			double target_ratio_height = target_height / facChip_size[1];
			double target_ratio_width = target_width / facChip_size[0];
			if (target_ratio_width >= 1.0)
				target_ratio_width = 1.0;
			std::vector<double> confidences;
			while (start_height_ratio + target_ratio_height < 1.0) {
				// get the cropped img
				cv::Mat tmp = src_chip(cv::Rect(0, src_chip.size().height * start_height_ratio, src_chip.size().width * target_ratio_width, src_chip.size().height * target_ratio_height));
				cv::Mat croppedImage = adjust_chip(tmp);
				// get confidence value for the cropped img
				double conf_value = compute_confidence(croppedImage, modeljson, false)[0];
				confidences.push_back(conf_value);
				index++;
				start_height_ratio = index * 0.1;
			}
			// find the best img
			double best_conf = 0;
			int best_id = -1;
			for (int i = 0; i < confidences.size(); i++) {
				if (best_conf < confidences[i]) {
					best_conf = confidences[i];
					best_id = i;
				}
			}
			// output best img
			cv::Mat best_cropped_tmp = src_chip(cv::Rect(0, src_chip.size().height * best_id * 0.1, src_chip.size().width * target_ratio_width, src_chip.size().height * target_ratio_height));
			cv::Mat best_cropped = adjust_chip(best_cropped_tmp);
			// always add best chip
			cropped_chips.push_back(best_cropped);
			if (bground && best_id != confidences.size() - 1) {//if best chip == door chip, ignore
															   // check the grammar of the last chip
				cv::Mat tmp = src_chip(cv::Rect(0, src_chip.size().height * (1 - target_ratio_height), src_chip.size().width  * target_ratio_width, src_chip.size().height * target_ratio_height));
				cv::Mat tmp_adjust = adjust_chip(tmp);
				// get confidence value for the cropped img
				int grammar_type = compute_confidence(tmp_adjust, modeljson, false)[1];
				if (grammar_type % 2 == 0) {// doors
					cropped_chips.push_back(tmp_adjust);
				}
			}
		}
	}
	else if (type == 4) {
		double longer_dim = 0;
		double target_dim = 0;
		bool bLonger_width = false;
		if (facChip_size[0] > facChip_size[1]) {
			longer_dim = facChip_size[0];
			target_dim = target_width;
			bLonger_width = true;
		}
		else {
			longer_dim = facChip_size[1];
			target_dim = target_height;
		}
		if (longer_dim < 1.6 * target_dim || !bMultipleChips) {
			// crop 30 * 30
			double target_ratio_width = target_width / facChip_size[0];
			double target_ratio_height = target_height / facChip_size[1];
			double padding_width_ratio = (1 - target_ratio_width) * 0.5;
			double padding_height_ratio = 0;
			if (!bground) {
				padding_height_ratio = (1 - target_ratio_height) * 0.5;
			}
			else {
				padding_height_ratio = (1 - target_ratio_height);
			}
			cv::Mat tmp = src_chip(cv::Rect(src_chip.size().width * padding_width_ratio, src_chip.size().height * padding_height_ratio, src_chip.size().width * target_ratio_width, src_chip.size().height * target_ratio_height));
			cropped_chips.push_back(adjust_chip(tmp));
		}
		else if (bLonger_width) {
			// check multiple chips and choose the one that has the highest confidence value
			int index = 0;
			double start_width_ratio = index * 0.1;
			double target_ratio_width = target_width / facChip_size[0];
			double target_ratio_height = target_height / facChip_size[1];
			double padding_height_ratio = (1 - target_ratio_height) * 0.5;
			std::vector<double> confidences;
			while (start_width_ratio + target_ratio_width < 1.0) {
				// get the cropped img
				cv::Mat tmp = src_chip(cv::Rect(src_chip.size().width * start_width_ratio, src_chip.size().height * padding_height_ratio, src_chip.size().width * target_ratio_width, src_chip.size().height * target_ratio_height));
				cv::Mat croppedImage = adjust_chip(tmp);
				// get confidence value for the cropped img
				double conf_value = compute_confidence(croppedImage, modeljson, false)[0];
				confidences.push_back(conf_value);
				index++;
				start_width_ratio = index * 0.1;
			}
			// find the best img
			double best_conf = 0;
			int best_id = -1;
			for (int i = 0; i < confidences.size(); i++) {
				if (best_conf < confidences[i]) {
					best_conf = confidences[i];
					best_id = i;
				}
			}
			// output best img
			cv::Mat  best_cropped_tmp = src_chip(cv::Rect(src_chip.size().width * best_id * 0.1, src_chip.size().height * padding_height_ratio, src_chip.size().width * target_ratio_width, src_chip.size().height * target_ratio_height));
			cv::Mat best_cropped = adjust_chip(best_cropped_tmp);
			// always add best chip
			cropped_chips.push_back(best_cropped);
			if (bground) {
				double padding_width_ratio = (1 - target_ratio_width) * 0.5;
				padding_height_ratio = (1 - target_ratio_height);
				cv::Mat tmp = src_chip(cv::Rect(src_chip.size().width * padding_width_ratio, src_chip.size().height * padding_height_ratio, src_chip.size().width * target_ratio_width, src_chip.size().height * target_ratio_height));
				cv::Mat tmp_adjust = adjust_chip(tmp);
				// get confidence value for the cropped img
				int grammar_type = compute_confidence(tmp_adjust, modeljson, false)[1];
				if (grammar_type % 2 == 0) {// doors
					cropped_chips.push_back(tmp_adjust);
				}
			}
		}
		else {
			// check multiple chips and choose the one that has the highest confidence value
			int index = 0;
			double start_height_ratio = index * 0.1;
			double target_ratio_width = target_width / facChip_size[0];
			double target_ratio_height = target_height / facChip_size[1];
			double padding_width_ratio = (1 - target_ratio_width) * 0.5;
			std::vector<double> confidences;
			while (start_height_ratio + target_ratio_height < 1.0) {
				// get the cropped img
				cv::Mat tmp = src_chip(cv::Rect(src_chip.size().width * padding_width_ratio, src_chip.size().height * start_height_ratio, src_chip.size().width * target_ratio_width, src_chip.size().height * target_ratio_height));
				cv::Mat croppedImage = adjust_chip(tmp);
				// get confidence value for the cropped img
				double conf_value = compute_confidence(croppedImage, modeljson, false)[0];
				confidences.push_back(conf_value);
				index++;
				start_height_ratio = index * 0.1;
			}
			// find the best img
			double best_conf = 0;
			int best_id = -1;
			for (int i = 0; i < confidences.size(); i++) {
				if (best_conf < confidences[i]) {
					best_conf = confidences[i];
					best_id = i;
				}
			}
			// output best img
			cv::Mat  best_cropped_tmp = src_chip(cv::Rect(src_chip.size().width * padding_width_ratio, src_chip.size().height * best_id * 0.1, src_chip.size().width * target_ratio_width, src_chip.size().height * target_ratio_height));
			cv::Mat best_cropped = adjust_chip(best_cropped_tmp);
			cropped_chips.push_back(best_cropped);
			if (bground && best_id != confidences.size() - 1) {
				double padding_height_ratio = (1 - target_ratio_height);
				cv::Mat tmp = src_chip(cv::Rect(src_chip.size().width * padding_width_ratio, src_chip.size().height * padding_height_ratio, src_chip.size().width * target_ratio_width, src_chip.size().height * target_ratio_height));
				cv::Mat tmp_adjust = adjust_chip(tmp);
				// get confidence value for the cropped img
				int grammar_type = compute_confidence(tmp_adjust, modeljson, false)[1];
				if (grammar_type % 2 == 0) {// doors
					cropped_chips.push_back(tmp_adjust);
				}
			}
		}
	}
	else {
		// do nothing
	}
	return cropped_chips;
}

cv::Mat adjust_chip(cv::Mat chip) {
	// load image
	cv::Mat dst_ehist, dst_classify;
	cv::Mat hsv;
	cvtColor(chip, hsv, cv::COLOR_BGR2HSV);
	std::vector<cv::Mat> bgr;   //destination array
	cv::split(hsv, bgr);//split source 
	for (int i = 0; i < 3; i++)
		cv::equalizeHist(bgr[i], bgr[i]);
	dst_ehist = bgr[2];
	int threshold = 0;
	// threshold classification
	dst_classify = facade_clustering_kkmeans(dst_ehist, cluster_number);
	// find the boundary
	int scan_line = 0;
	// bottom 
	int pos_top = 0;
	for (int i = 0; i < dst_classify.size().height; i++) {
		scan_line = 0;
		for (int j = 0; j < dst_classify.size().width; j++) {
			//noise
			if ((int)dst_classify.at<uchar>(i, j) == 0) {
				scan_line++;
			}
		}
		if (scan_line * 1.0 / dst_classify.size().width < 0.9) { // threshold is 0.9
			pos_top = i;
			break;
		}

	}
	// bottom 
	int pos_bot = 0;
	for (int i = dst_classify.size().height - 1; i >= 0; i--) {
		scan_line = 0;
		for (int j = 0; j < dst_classify.size().width; j++) {
			//noise
			if ((int)dst_classify.at<uchar>(i, j) == 0) {
				scan_line++;
			}
		}
		if (scan_line * 1.0 / dst_classify.size().width < 0.90) { // threshold is 0.9
			pos_bot = i;
			break;
		}

	}

	// left
	int pos_left = 0;
	for (int i = 0; i < dst_classify.size().width; i++) {
		scan_line = 0;
		for (int j = 0; j < dst_classify.size().height; j++) {
			//noise
			if ((int)dst_classify.at<uchar>(j, i) == 0) {
				scan_line++;
			}
		}
		if (scan_line * 1.0 / dst_classify.size().height > 0.1) { // threshold is 0.1
			pos_left = i;
			break;
		}

	}
	// right
	int pos_right = 0;
	for (int i = dst_classify.size().width - 1; i >= 0; i--) {
		scan_line = 0;
		for (int j = 0; j < dst_classify.size().height; j++) {
			//noise
			if ((int)dst_classify.at<uchar>(j, i) == 0) {
				scan_line++;
			}
		}
		if (scan_line * 1.0 / dst_classify.size().height > 0.1) { // threshold is 0.1
			pos_right = i;
			break;
		}

	}
	if (pos_left >= pos_right)
		return chip;
	if (pos_top >= pos_bot)
		return chip;
	// crop the img
	cv::Mat croppedImage = chip(cv::Rect(pos_left, pos_top, pos_right - pos_left, pos_bot - pos_top));
	return croppedImage;
}

bool checkFacade(FacadeInfo& fi) {
	// load image
	cv::Mat src, dst_ehist, dst_classify;
	src = fi.facadeImg(fi.inscRect_px).clone();
	cv::Mat hsv;
	cvtColor(src, hsv, cv::COLOR_BGR2HSV);
	std::vector<cv::Mat> bgr;   //destination array
	cv::split(hsv, bgr);//split source 
	for (int i = 0; i < 3; i++)
		cv::equalizeHist(bgr[i], bgr[i]);
	dst_ehist = bgr[2];
	int threshold = 0;
	// kkmeans classification
	dst_classify = facade_clustering_kkmeans(dst_ehist, cluster_number);
	// compute coverage of black pixels
	int count = 0;
	for (int i = 0; i < dst_classify.size().height; i++) {
		for (int j = 0; j < dst_classify.size().width; j++) {
			//noise
			if ((int)dst_classify.at<uchar>(i, j) == 0) {
				count++;
			}
		}
	}
	double coverage = count * 1.0 / (dst_classify.size().height * dst_classify.size().width);
	if (coverage > 0.7 || coverage < 0.3)
		return false;
	else
		return true;
}

void saveInvalidFacade(FacadeInfo& fi, bool bDebug) {

	fi.valid = false;

	// compute avg color
	cv::Scalar avg_color(0, 0, 0);
	cv::Mat src = fi.facadeImg(fi.inscRect_px).clone();
	for (int i = 0; i < src.size().height; i++) {
		for (int j = 0; j < src.size().width; j++) {
			for (int c = 0; c < 3; c++)
				avg_color.val[c] += src.at<cv::Vec4b>(i, j)[c];
		}
	}
	fi.bg_color.b = avg_color.val[0] / (src.size().height * src.size().width) / 255;
	fi.bg_color.g = avg_color.val[1] / (src.size().height * src.size().width) / 255;
	fi.bg_color.r = avg_color.val[2] / (src.size().height * src.size().width) / 255;
}

std::vector<double> compute_confidence(cv::Mat croppedImage, std::string modeljson, bool bDebug) {
	FILE* fp = fopen(modeljson.c_str(), "rb"); // non-Windows use "r"
	char readBuffer[10240];
	rapidjson::FileReadStream isModel(fp, readBuffer, sizeof(readBuffer));
	rapidjson::Document docModel;
	docModel.ParseStream(isModel);
	// default size for NN
	int height = 224; // DNN image height
	int width = 224; // DNN image width
	std::vector<double> tmp_array = read1DArray(docModel, "defaultSize");
	width = tmp_array[0];
	height = tmp_array[1];
	// load image
	cv::Mat src, dst_ehist, dst_classify;
	src = croppedImage.clone();
	cv::Mat hsv;
	cvtColor(src, hsv, cv::COLOR_BGR2HSV);
	std::vector<cv::Mat> bgr;   //destination array
	cv::split(hsv, bgr);//split source 
	for (int i = 0; i < 3; i++)
		cv::equalizeHist(bgr[i], bgr[i]);
	dst_ehist = bgr[2];
	int threshold = 0;
	// kkmeans classification
	dst_classify = facade_clustering_kkmeans(dst_ehist, cluster_number);
	// generate input image for DNN
	cv::Scalar bg_color(255, 255, 255); // white back ground
	cv::Scalar window_color(0, 0, 0); // black for windows
	cv::Mat scale_img;
	cv::resize(dst_classify, scale_img, cv::Size(width, height));
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
	int padding_size = 5;
	int borderType = cv::BORDER_CONSTANT;
	cv::Scalar value(255, 255, 255);
	cv::Mat aligned_img_padding;
	cv::copyMakeBorder(aligned_img, aligned_img_padding, padding_size, padding_size, padding_size, padding_size, borderType, value);

	// find contours
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(aligned_img_padding, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	std::vector<cv::Rect> boundRect(contours.size());
	std::vector<std::vector<cv::Rect>> largestRect(contours.size());
	std::vector<bool> bIntersectionbbox(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		boundRect[i] = cv::boundingRect(cv::Mat(contours[i]));
		bIntersectionbbox[i] = false;
	}
	// find the largest rectangles
	cv::Mat drawing(aligned_img_padding.size(), CV_8UC3, bg_color);
	for (int i = 0; i< contours.size(); i++)
	{
		if (hierarchy[i][3] != 0) continue;
		cv::Mat tmp(aligned_img_padding.size(), CV_8UC3, window_color);
		drawContours(tmp, contours, i, bg_color, -1, 8, hierarchy, 0, cv::Point());
		cv::Mat tmp_gray;
		cvtColor(tmp, tmp_gray, cv::COLOR_BGR2GRAY);
		cv::Rect tmp_rect = findLargestRectangle(tmp_gray);
		largestRect[i].push_back(tmp_rect);
		float area_contour = cv::contourArea(contours[i]);
		float area_rect = 0;
		area_rect += tmp_rect.width * tmp_rect.height;
		float ratio = area_rect / area_contour;
		while (ratio < 0.90) { // find more largest rectangles in the rest area
							   // clear up the previous rectangles
			tmp_gray.empty();
			cv::rectangle(tmp, cv::Point(tmp_rect.tl().x, tmp_rect.tl().y), cv::Point(tmp_rect.br().x, tmp_rect.br().y), window_color, -1);
			cvtColor(tmp, tmp_gray, cv::COLOR_BGR2GRAY);
			tmp_rect = findLargestRectangle(tmp_gray);
			area_rect += tmp_rect.width * tmp_rect.height;
			if (tmp_rect.width * tmp_rect.height > 100)
				largestRect[i].push_back(tmp_rect);
			ratio = area_rect / area_contour;
		}
	}
	// check intersection
	for (int i = 0; i < contours.size(); i++) {
		if (hierarchy[i][3] != 0 || bIntersectionbbox[i]) {
			bIntersectionbbox[i] = true;
			continue;
		}
		for (int j = i + 1; j < contours.size(); j++) {
			if (findIntersection(boundRect[i], boundRect[j])) {
				bIntersectionbbox[i] = true;
				bIntersectionbbox[j] = true;
				break;
			}
		}
	}
	//
	cv::Mat dnn_img(aligned_img_padding.size(), CV_8UC3, bg_color);
	int num_contours = 0;
	double largest_rec_area = 0;
	double largest_ratio = 0;
	for (int i = 1; i< contours.size(); i++)
	{
		if (hierarchy[i][3] != 0) continue;
		// check the validity of the rect
		float area_contour = cv::contourArea(contours[i]);
		float area_rect = boundRect[i].width * boundRect[i].height;
		if (area_rect < 50 || area_contour < 50) continue;
		num_contours++;
		float ratio = area_contour / area_rect;
		if (!bIntersectionbbox[i] /*&& (ratio > 0.60 || area_contour < 160)*/) {
			cv::rectangle(dnn_img, cv::Point(boundRect[i].tl().x, boundRect[i].tl().y), cv::Point(boundRect[i].br().x, boundRect[i].br().y), window_color, -1);
			if (largest_rec_area < area_rect)
				largest_rec_area = area_rect;
		}
		else {
			for (int j = 0; j < 1; j++)
				cv::rectangle(dnn_img, cv::Point(largestRect[i][j].tl().x, largestRect[i][j].tl().y), cv::Point(largestRect[i][j].br().x, largestRect[i][j].br().y), window_color, -1);
			if (largest_rec_area < area_contour)
				largest_rec_area = area_contour;
		}
	}
	largest_ratio = largest_rec_area / (aligned_img_padding.size().width * aligned_img_padding.size().height);
	// remove padding
	dnn_img = dnn_img(cv::Rect(padding_size, padding_size, width, height));
	// feed DNN
	rapidjson::Value& grammars = docModel["grammars"];
	// classifier
	rapidjson::Value& grammar_classifier = grammars["classifier"];
	// path of DN model
	std::string classifier_name = readStringValue(grammar_classifier, "model");
	int num_classes = readNumber(grammar_classifier, "number_paras", 6);
	if (bDebug) {
		std::cout << "classifier_name is " << classifier_name << std::endl;
	}
	cv::Mat dnn_img_rgb;
	cv::cvtColor(dnn_img, dnn_img_rgb, CV_BGR2RGB);
	cv::Mat img_float;
	dnn_img_rgb.convertTo(img_float, CV_32F, 1.0 / 255);
	auto img_tensor = torch::from_blob(img_float.data, { 1, 224, 224, 3 }).to(torch::kCPU);
	img_tensor = img_tensor.permute({ 0, 3, 1, 2 });
	img_tensor[0][0] = img_tensor[0][0].sub(0.485).div(0.229);
	img_tensor[0][1] = img_tensor[0][1].sub(0.456).div(0.224);
	img_tensor[0][2] = img_tensor[0][2].sub(0.406).div(0.225);

	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(img_tensor);

	// Deserialize the ScriptModule from a file using torch::jit::load().
	std::shared_ptr<torch::jit::script::Module> classifier_module = torch::jit::load(classifier_name);
	classifier_module->to(at::kCPU);
	assert(classifier_module != nullptr);
	torch::Tensor out_tensor = classifier_module->forward(inputs).toTensor();
	torch::Tensor confidences_tensor = torch::softmax(out_tensor, 1);
	if (bDebug)
		std::cout << confidences_tensor.slice(1, 0, num_classes) << std::endl;

	double best_score = 0;
	int best_id;
	for (int i = 0; i < num_classes; i++) {
		double tmp = confidences_tensor.slice(1, i, i + 1).item<float>();
		if (tmp > best_score) {
			best_score = tmp;
			best_id = i;
		}
	}
	fclose(fp);
	std::vector<double> results;
	results.push_back(best_score);
	results.push_back(best_id + 1);
	results.push_back(num_contours);
	results.push_back(largest_ratio);
	return results;
}

std::vector<double> compute_door_paras(cv::Mat croppedImage, std::string modeljson, bool bDebug) {
	FILE* fp = fopen(modeljson.c_str(), "rb"); // non-Windows use "r"
	char readBuffer[10240];
	rapidjson::FileReadStream isModel(fp, readBuffer, sizeof(readBuffer));
	rapidjson::Document docModel;
	docModel.ParseStream(isModel);
	// default size for NN
	int height = 224; // DNN image height
	int width = 224; // DNN image width
	std::vector<double> tmp_array = read1DArray(docModel, "defaultSize");
	width = tmp_array[0];
	height = tmp_array[1];
	// load image
	cv::Mat src, dst_ehist, dst_classify;
	src = croppedImage.clone();
	cv::Mat hsv;
	cvtColor(src, hsv, cv::COLOR_BGR2HSV);
	std::vector<cv::Mat> bgr;   //destination array
	cv::split(hsv, bgr);//split source 
	for (int i = 0; i < 3; i++)
		cv::equalizeHist(bgr[i], bgr[i]);
	dst_ehist = bgr[2];
	int threshold = 0;
	// kkmeans classification
	dst_classify = facade_clustering_kkmeans(dst_ehist, cluster_number);
	// generate input image for DNN
	cv::Scalar bg_color(255, 255, 255); // white back ground
	cv::Scalar window_color(0, 0, 0); // black for windows
	cv::Mat scale_img;
	cv::resize(dst_classify, scale_img, cv::Size(width, height));
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
	int padding_size = 5;
	int borderType = cv::BORDER_CONSTANT;
	cv::Scalar value(255, 255, 255);
	cv::Mat aligned_img_padding;
	cv::copyMakeBorder(aligned_img, aligned_img_padding, padding_size, padding_size, padding_size, padding_size, borderType, value);

	// find contours
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(aligned_img_padding, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	std::vector<cv::Rect> boundRect(contours.size());
	std::vector<std::vector<cv::Rect>> largestRect(contours.size());
	std::vector<bool> bIntersectionbbox(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		boundRect[i] = cv::boundingRect(cv::Mat(contours[i]));
		bIntersectionbbox[i] = false;
	}
	// find the largest rectangles
	cv::Mat drawing(aligned_img_padding.size(), CV_8UC3, bg_color);
	for (int i = 0; i< contours.size(); i++)
	{
		if (hierarchy[i][3] != 0) continue;
		cv::Mat tmp(aligned_img_padding.size(), CV_8UC3, window_color);
		drawContours(tmp, contours, i, bg_color, -1, 8, hierarchy, 0, cv::Point());
		cv::Mat tmp_gray;
		cvtColor(tmp, tmp_gray, cv::COLOR_BGR2GRAY);
		cv::Rect tmp_rect = findLargestRectangle(tmp_gray);
		largestRect[i].push_back(tmp_rect);
		float area_contour = cv::contourArea(contours[i]);
		float area_rect = 0;
		area_rect += tmp_rect.width * tmp_rect.height;
		float ratio = area_rect / area_contour;
		while (ratio < 0.90) { // find more largest rectangles in the rest area
							   // clear up the previous rectangles
			tmp_gray.empty();
			cv::rectangle(tmp, cv::Point(tmp_rect.tl().x, tmp_rect.tl().y), cv::Point(tmp_rect.br().x, tmp_rect.br().y), window_color, -1);
			cvtColor(tmp, tmp_gray, cv::COLOR_BGR2GRAY);
			tmp_rect = findLargestRectangle(tmp_gray);
			area_rect += tmp_rect.width * tmp_rect.height;
			if (tmp_rect.width * tmp_rect.height > 100)
				largestRect[i].push_back(tmp_rect);
			ratio = area_rect / area_contour;
		}
	}
	// check intersection
	for (int i = 0; i < contours.size(); i++) {
		if (hierarchy[i][3] != 0 || bIntersectionbbox[i]) {
			bIntersectionbbox[i] = true;
			continue;
		}
		for (int j = i + 1; j < contours.size(); j++) {
			if (findIntersection(boundRect[i], boundRect[j])) {
				bIntersectionbbox[i] = true;
				bIntersectionbbox[j] = true;
				break;
			}
		}
	}
	//
	cv::Mat dnn_img(aligned_img_padding.size(), CV_8UC3, bg_color);
	int num_contours = 0;
	double largest_rec_area = 0;
	double largest_ratio = 0;
	for (int i = 1; i< contours.size(); i++)
	{
		if (hierarchy[i][3] != 0) continue;
		// check the validity of the rect
		float area_contour = cv::contourArea(contours[i]);
		float area_rect = boundRect[i].width * boundRect[i].height;
		if (area_rect < 80 || area_contour < 80) continue;
		num_contours++;
		float ratio = area_contour / area_rect;
		if (!bIntersectionbbox[i] /*&& (ratio > 0.60 || area_contour < 160)*/) {
			cv::rectangle(dnn_img, cv::Point(boundRect[i].tl().x, boundRect[i].tl().y), cv::Point(boundRect[i].br().x, boundRect[i].br().y), window_color, -1);
			if (largest_rec_area < area_rect)
				largest_rec_area = area_rect;
		}
		else {
			for (int j = 0; j < 1; j++)
				cv::rectangle(dnn_img, cv::Point(largestRect[i][j].tl().x, largestRect[i][j].tl().y), cv::Point(largestRect[i][j].br().x, largestRect[i][j].br().y), window_color, -1);
			if (largest_rec_area < area_contour)
				largest_rec_area = area_contour;
		}
	}
	largest_ratio = largest_rec_area / (aligned_img_padding.size().width * aligned_img_padding.size().height);
	// remove padding
	dnn_img = dnn_img(cv::Rect(padding_size, padding_size, width, height));
	// feed DNN
	rapidjson::Value& grammars = docModel["grammars"];
	// classifier
	rapidjson::Value& grammar_classifier = grammars["classifier"];
	// path of DN model
	std::string classifier_name = readStringValue(grammar_classifier, "model");
	int num_classes = readNumber(grammar_classifier, "number_paras", 6);
	if (bDebug) {
		std::cout << "classifier_name is " << classifier_name << std::endl;
	}
	cv::Mat dnn_img_rgb;
	cv::cvtColor(dnn_img, dnn_img_rgb, CV_BGR2RGB);
	cv::Mat img_float;
	dnn_img_rgb.convertTo(img_float, CV_32F, 1.0 / 255);
	auto img_tensor = torch::from_blob(img_float.data, { 1, 224, 224, 3 }).to(torch::kCPU);
	img_tensor = img_tensor.permute({ 0, 3, 1, 2 });
	img_tensor[0][0] = img_tensor[0][0].sub(0.485).div(0.229);
	img_tensor[0][1] = img_tensor[0][1].sub(0.456).div(0.224);
	img_tensor[0][2] = img_tensor[0][2].sub(0.406).div(0.225);

	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(img_tensor);

	// Deserialize the ScriptModule from a file using torch::jit::load().
	std::shared_ptr<torch::jit::script::Module> classifier_module = torch::jit::load(classifier_name);
	classifier_module->to(at::kCPU);
	assert(classifier_module != nullptr);
	torch::Tensor out_tensor = classifier_module->forward(inputs).toTensor();
	torch::Tensor confidences_tensor = torch::softmax(out_tensor, 1);
	if (bDebug)
		std::cout << confidences_tensor.slice(1, 0, num_classes) << std::endl;

	double best_score = 0;
	int best_id;
	for (int i = 0; i < num_classes; i++) {
		double tmp = confidences_tensor.slice(1, i, i + 1).item<float>();
		if (tmp > best_score) {
			best_score = tmp;
			best_id = i;
		}
	}
	best_id = best_id + 1;
	fclose(fp);
	std::vector<double>results;
	if (best_id % 2 != 0) {// impossible
		results.clear();
		return results;
	}
	else { // get door paras
		   // choose conresponding estimation DNN
		std::string model_name;
		std::string grammar_name = "grammar" + std::to_string(best_id);
		rapidjson::Value& grammar = grammars[grammar_name.c_str()];
		// path of DN model
		model_name = readStringValue(grammar, "model");
		if (bDebug)
			std::cout << "model_name is " << model_name << std::endl;
		// number of paras
		int num_paras = readNumber(grammar, "number_paras", 5);

		std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(model_name);
		module->to(at::kCPU);
		assert(module != nullptr);
		torch::Tensor out_tensor_grammar = module->forward(inputs).toTensor();
		if (bDebug)
			std::cout << out_tensor_grammar.slice(1, 0, num_paras) << std::endl;
		std::vector<double> paras;
		for (int i = 0; i < num_paras; i++) {
			paras.push_back(out_tensor_grammar.slice(1, i, i + 1).item<float>());
		}
		for (int i = 0; i < num_paras; i++) {
			if (paras[i] < 0)
				paras[i] = 0;
		}
		std::vector<double> predictions;
		if (best_id == 2) {
			predictions = grammar2(modeljson, paras, bDebug);
		}
		else if (best_id == 4) {
			predictions = grammar4(modeljson, paras, bDebug);
		}
		else if (best_id == 6) {
			predictions = grammar6(modeljson, paras, bDebug);
		}
		else {
			//do nothing
			predictions = grammar2(modeljson, paras, bDebug);
		}
		return predictions;
	}
}

bool segment_chip(cv::Mat croppedImage, cv::Mat& dnn_img, FacadeInfo& fi, std::string modeljson, bool bDebug) {
	FILE* fp = fopen(modeljson.c_str(), "rb"); // non-Windows use "r"
	char readBuffer[10240];
	rapidjson::FileReadStream isModel(fp, readBuffer, sizeof(readBuffer));
	rapidjson::Document docModel;
	docModel.ParseStream(isModel);
	// default size for NN
	int height = 224; // DNN image height
	int width = 224; // DNN image width
	std::vector<double> tmp_array = read1DArray(docModel, "defaultSize");
	width = tmp_array[0];
	height = tmp_array[1];
	// load image
	cv::Mat src, dst_ehist, dst_classify, src_histeq;
	src = croppedImage.clone();
	cv::Mat hsv;
	cvtColor(src, hsv, cv::COLOR_BGR2HSV);
	std::vector<cv::Mat> bgr;   //destination array
	cv::split(hsv, bgr);//split source 
	for (int i = 0; i < 3; i++)
		cv::equalizeHist(bgr[i], bgr[i]);
	dst_ehist = bgr[2];
	cv::merge(bgr, src_histeq);
	cvtColor(src_histeq, src_histeq, cv::COLOR_HSV2BGR);
	//
	dst_ehist = bgr[2];
	int threshold = 0;
	// kkmeans classification
	dst_classify = facade_clustering_kkmeans(dst_ehist, cluster_number);
	// generate input image for DNN
	cv::Scalar bg_color(255, 255, 255); // white back ground
	cv::Scalar window_color(0, 0, 0); // black for windows
	cv::Mat scale_img;
	cv::resize(dst_classify, scale_img, cv::Size(width, height));
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
	int padding_size = 5;
	int borderType = cv::BORDER_CONSTANT;
	cv::Scalar value(255, 255, 255);
	cv::Mat aligned_img_padding;
	cv::copyMakeBorder(aligned_img, aligned_img_padding, padding_size, padding_size, padding_size, padding_size, borderType, value);

	// find contours
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(aligned_img_padding, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	std::vector<cv::Rect> boundRect(contours.size());
	std::vector<std::vector<cv::Rect>> largestRect(contours.size());
	std::vector<bool> bIntersectionbbox(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		boundRect[i] = cv::boundingRect(cv::Mat(contours[i]));
		bIntersectionbbox[i] = false;
	}
	// find the largest rectangles
	cv::Mat drawing(aligned_img_padding.size(), CV_8UC3, bg_color);
	for (int i = 0; i< contours.size(); i++)
	{
		if (hierarchy[i][3] != 0) continue;
		cv::Mat tmp(aligned_img_padding.size(), CV_8UC3, window_color);
		drawContours(tmp, contours, i, bg_color, -1, 8, hierarchy, 0, cv::Point());
		cv::Mat tmp_gray;
		cvtColor(tmp, tmp_gray, cv::COLOR_BGR2GRAY);
		cv::Rect tmp_rect = findLargestRectangle(tmp_gray);
		largestRect[i].push_back(tmp_rect);
		float area_contour = cv::contourArea(contours[i]);
		float area_rect = 0;
		area_rect += tmp_rect.width * tmp_rect.height;
		float ratio = area_rect / area_contour;
		while (ratio < 0.90) { // find more largest rectangles in the rest area
							   // clear up the previous rectangles
			tmp_gray.empty();
			cv::rectangle(tmp, cv::Point(tmp_rect.tl().x, tmp_rect.tl().y), cv::Point(tmp_rect.br().x, tmp_rect.br().y), window_color, -1);
			cvtColor(tmp, tmp_gray, cv::COLOR_BGR2GRAY);
			tmp_rect = findLargestRectangle(tmp_gray);
			area_rect += tmp_rect.width * tmp_rect.height;
			if (tmp_rect.width * tmp_rect.height > 100)
				largestRect[i].push_back(tmp_rect);
			ratio = area_rect / area_contour;
		}
	}
	// check intersection
	for (int i = 0; i < contours.size(); i++) {
		if (hierarchy[i][3] != 0 || bIntersectionbbox[i]) {
			bIntersectionbbox[i] = true;
			continue;
		}
		for (int j = i + 1; j < contours.size(); j++) {
			if (findIntersection(boundRect[i], boundRect[j])) {
				bIntersectionbbox[i] = true;
				bIntersectionbbox[j] = true;
				break;
			}
		}
	}
	//
	dnn_img = cv::Mat(aligned_img_padding.size(), CV_8UC3, bg_color);
	for (int i = 1; i< contours.size(); i++)
	{
		if (hierarchy[i][3] != 0) continue;
		// check the validity of the rect
		float area_contour = cv::contourArea(contours[i]);
		float area_rect = boundRect[i].width * boundRect[i].height;
		if (area_rect < 50 || area_contour < 50) continue;
		float ratio = area_contour / area_rect;
		if (!bIntersectionbbox[i] /*&& (ratio > 0.60 || area_contour < 160)*/) {
			cv::rectangle(dnn_img, cv::Point(boundRect[i].tl().x, boundRect[i].tl().y), cv::Point(boundRect[i].br().x, boundRect[i].br().y), window_color, -1);
		}
		else {
			for (int j = 0; j < 1; j++)
				cv::rectangle(dnn_img, cv::Point(largestRect[i][j].tl().x, largestRect[i][j].tl().y), cv::Point(largestRect[i][j].br().x, largestRect[i][j].br().y), window_color, -1);
		}
	}
	// remove padding
	dnn_img = dnn_img(cv::Rect(padding_size, padding_size, width, height));
	fclose(fp);
	// write back to json file
	cv::Scalar bg_avg_color(0, 0, 0);
	cv::Scalar win_avg_color(0, 0, 0);
	cv::Scalar bg_histeq_color(0, 0, 0);
	cv::Scalar win_histeq_color(0, 0, 0);
	{
		int bg_count = 0;
		int win_count = 0;
		for (int i = 0; i < dst_classify.size().height; i++) {
			for (int j = 0; j < dst_classify.size().width; j++) {
				if ((int)dst_classify.at<uchar>(i, j) == 0) {
					win_avg_color.val[0] += src.at<cv::Vec4b>(i, j)[0];
					win_avg_color.val[1] += src.at<cv::Vec4b>(i, j)[1];
					win_avg_color.val[2] += src.at<cv::Vec4b>(i, j)[2];
					win_histeq_color.val[0] += src_histeq.at<cv::Vec4b>(i, j)[0];
					win_histeq_color.val[1] += src_histeq.at<cv::Vec4b>(i, j)[1];
					win_histeq_color.val[2] += src_histeq.at<cv::Vec4b>(i, j)[2];
					win_count++;
				}
				else {
					bg_avg_color.val[0] += src.at<cv::Vec4b>(i, j)[0];
					bg_avg_color.val[1] += src.at<cv::Vec4b>(i, j)[1];
					bg_avg_color.val[2] += src.at<cv::Vec4b>(i, j)[2];
					bg_histeq_color.val[0] += src_histeq.at<cv::Vec4b>(i, j)[0];
					bg_histeq_color.val[1] += src_histeq.at<cv::Vec4b>(i, j)[1];
					bg_histeq_color.val[2] += src_histeq.at<cv::Vec4b>(i, j)[2];
					bg_count++;
				}
			}
		}
		if (win_count > 0) {
			win_avg_color.val[0] = win_avg_color.val[0] / win_count;
			win_avg_color.val[1] = win_avg_color.val[1] / win_count;
			win_avg_color.val[2] = win_avg_color.val[2] / win_count;
			win_histeq_color.val[0] = win_histeq_color.val[0] / win_count;
			win_histeq_color.val[1] = win_histeq_color.val[1] / win_count;
			win_histeq_color.val[2] = win_histeq_color.val[2] / win_count;
		}
		if (bg_count > 0) {
			bg_avg_color.val[0] = bg_avg_color.val[0] / bg_count;
			bg_avg_color.val[1] = bg_avg_color.val[1] / bg_count;
			bg_avg_color.val[2] = bg_avg_color.val[2] / bg_count;
			bg_histeq_color.val[0] = bg_histeq_color.val[0] / bg_count;
			bg_histeq_color.val[1] = bg_histeq_color.val[1] / bg_count;
			bg_histeq_color.val[2] = bg_histeq_color.val[2] / bg_count;
		}
	}

	fi.bg_color.b = bg_avg_color.val[0] / 255;
	fi.bg_color.g = bg_avg_color.val[1] / 255;
	fi.bg_color.r = bg_avg_color.val[2] / 255;

	fi.win_color.b = win_avg_color.val[0] / 255;
	fi.win_color.g = win_avg_color.val[1] / 255;
	fi.win_color.r = win_avg_color.val[2] / 255;

	return true;
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
	drawing = drawing(cv::Rect(padding_size, padding_size, 224, 224));
	cv::Mat aligned_img = cleanAlignedImage(drawing, 0.10);
	return aligned_img;
}

// Returns the largest rectangle inscribed within regions of all non-zero pixels
cv::Rect findLargestRectangle(cv::Mat image) {
	assert(image.channels() == 1);
	cv::Mat mask = (image > 0) / 255;
	mask.convertTo(mask, CV_16S);

	// Get the largest area rectangle under a histogram
	auto maxHist = [](cv::Mat hist) -> cv::Rect {
		// Append -1 to both ends
		cv::copyMakeBorder(hist, hist, 0, 0, 1, 1, cv::BORDER_CONSTANT, cv::Scalar::all(-1));
		cv::Rect maxRect(-1, -1, 0, 0);

		// Initialize stack to first element
		std::stack<int> colStack;
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

bool insideRect(cv::Rect a1, cv::Point p) {
	bool bresult = false;
	if (p.x >= a1.tl().x && p.x <= a1.br().x && p.y >= a1.tl().y && p.y <= a1.br().y)
		bresult = true;
	return bresult;
}

bool findIntersection(cv::Rect a1, cv::Rect a2) {
	// a2 insection with a1 
	if (insideRect(a1, a2.tl()))
		return true;
	if (insideRect(a1, cv::Point(a2.tl().x, a2.br().y)))
		return true;
	if (insideRect(a1, a2.br()))
		return true;
	if (insideRect(a1, cv::Point(a2.br().x, a2.tl().y)))
		return true;
	// a1 insection with a2
	if (insideRect(a2, a1.tl()))
		return true;
	if (insideRect(a2, cv::Point(a1.tl().x, a1.br().y)))
		return true;
	if (insideRect(a2, a1.br()))
		return true;
	if (insideRect(a2, cv::Point(a1.br().x, a1.tl().y)))
		return true;
	return false;
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

void feedDnn(cv::Mat dnn_img, FacadeInfo& fi, std::string modeljson, bool bDebug) {
	FILE* fp = fopen(modeljson.c_str(), "rb"); // non-Windows use "r"
	char readBuffer[10240];
	rapidjson::FileReadStream isModel(fp, readBuffer, sizeof(readBuffer));
	rapidjson::Document docModel;
	docModel.ParseStream(isModel);
	std::string classifier_name;

	rapidjson::Value& grammars = docModel["grammars"];
	// classifier
	rapidjson::Value& grammar_classifier = grammars["classifier"];
	// path of DN model
	classifier_name = readStringValue(grammar_classifier, "model");
	int num_classes = readNumber(grammar_classifier, "number_paras", 6);
	if (bDebug) {
		std::cout << "classifier_name is " << classifier_name << std::endl;
	}
	cv::Mat dnn_img_rgb;
	cv::cvtColor(dnn_img, dnn_img_rgb, CV_BGR2RGB);
	cv::Mat img_float;
	dnn_img_rgb.convertTo(img_float, CV_32F, 1.0 / 255);
	auto img_tensor = torch::from_blob(img_float.data, { 1, 224, 224, 3 }).to(torch::kCPU);
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
		std::shared_ptr<torch::jit::script::Module> classifier_module = torch::jit::load(classifier_name);
		classifier_module->to(at::kCPU);
		assert(classifier_module != nullptr);
		torch::Tensor out_tensor = classifier_module->forward(inputs).toTensor();
		//std::cout << out_tensor.slice(1, 0, num_classes) << std::endl;

		torch::Tensor confidences_tensor = torch::softmax(out_tensor, 1);
		if (bDebug)
			std::cout << confidences_tensor.slice(1, 0, num_classes) << std::endl;

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
		if (bDebug)
			std::cout << "DNN class is " << best_class << std::endl;
	}
	// adjust the best_class
	if (!readGround(fi)) {
		if (best_class % 2 == 0)
			best_class = best_class - 1;
	}
	// choose conresponding estimation DNN
	std::string model_name;
	std::string grammar_name = "grammar" + std::to_string(best_class);
	rapidjson::Value& grammar = grammars[grammar_name.c_str()];
	// path of DN model
	model_name = readStringValue(grammar, "model");
	if (bDebug)
		std::cout << "model_name is " << model_name << std::endl;
	// number of paras
	int num_paras = readNumber(grammar, "number_paras", 5);

	std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(model_name);
	module->to(at::kCPU);
	assert(module != nullptr);
	torch::Tensor out_tensor_grammar = module->forward(inputs).toTensor();
	if (bDebug)
		std::cout << out_tensor_grammar.slice(1, 0, num_paras) << std::endl;
	std::vector<double> paras;
	for (int i = 0; i < num_paras; i++) {
		paras.push_back(out_tensor_grammar.slice(1, i, i + 1).item<float>());
	}
	for (int i = 0; i < num_paras; i++) {
		if (paras[i] < 0)
			paras[i] = 0;
	}
	fclose(fp);
	std::vector<double> predictions;
	if (best_class == 1) {
		predictions = grammar1(modeljson, paras, bDebug);
	}
	else if (best_class == 2) {
		predictions = grammar2(modeljson, paras, bDebug);
	}
	else if (best_class == 3) {
		predictions = grammar3(modeljson, paras, bDebug);
	}
	else if (best_class == 4) {
		predictions = grammar4(modeljson, paras, bDebug);
	}
	else if (best_class == 5) {
		predictions = grammar5(modeljson, paras, bDebug);
	}
	else if (best_class == 6) {
		predictions = grammar6(modeljson, paras, bDebug);
	}
	else {
		//do nothing
		predictions = grammar1(modeljson, paras, bDebug);
	}

	fi.conf[0] = confidence_values[0];
	fi.conf[1] = confidence_values[1];
	fi.conf[2] = confidence_values[2];
	fi.conf[3] = confidence_values[3];
	fi.conf[4] = confidence_values[4];
	fi.conf[5] = confidence_values[5];
	if (fi.grammar == -1)
		fi.grammar = best_class;
	else
		fi.grammar = best_class + 1;

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

bool readGround(FacadeInfo& fi) {
	return fi.inscGround;
}

std::vector<double> grammar1(std::string modeljson, std::vector<double> paras, bool bDebug) {
	FILE* fp = fopen(modeljson.c_str(), "rb"); // non-Windows use "r"
	char readBuffer[10240];
	rapidjson::FileReadStream isModel(fp, readBuffer, sizeof(readBuffer));
	rapidjson::Document docModel;
	docModel.ParseStream(isModel);
	rapidjson::Value& grammars = docModel["grammars"];
	rapidjson::Value& grammar = grammars["grammar1"];
	// range of Rows
	std::vector<double> tmp_array = read1DArray(grammar, "rangeOfRows");
	std::pair<int, int> imageRows(tmp_array[0], tmp_array[1]);
	// range of Cols
	tmp_array.empty();
	tmp_array = read1DArray(grammar, "rangeOfCols");
	std::pair<int, int> imageCols(tmp_array[0], tmp_array[1]);
	// relativeWidth
	tmp_array.empty();
	tmp_array = read1DArray(grammar, "relativeWidth");
	std::pair<double, double> imageRelativeWidth(tmp_array[0], tmp_array[1]);
	// relativeHeight
	tmp_array.empty();
	tmp_array = read1DArray(grammar, "relativeHeight");
	std::pair<double, double> imageRelativeHeight(tmp_array[0], tmp_array[1]);
	if (bDebug) {
		std::cout << "imageRows is " << imageRows.first << ", " << imageRows.second << std::endl;
		std::cout << "imageCols is " << imageCols.first << ", " << imageCols.second << std::endl;
		std::cout << "imageRelativeWidth is " << imageRelativeWidth.first << ", " << imageRelativeWidth.second << std::endl;
		std::cout << "imageRelativeHeight is " << imageRelativeHeight.first << ", " << imageRelativeHeight.second << std::endl;
	}
	fclose(fp);
	int img_rows = paras[0] * (imageRows.second - imageRows.first) + imageRows.first;
	if (paras[0] * (imageRows.second - imageRows.first) + imageRows.first - img_rows > 0.7)
		img_rows++;
	int img_cols = paras[1] * (imageCols.second - imageCols.first) + imageCols.first;
	if (paras[1] * (imageCols.second - imageCols.first) + imageCols.first - img_cols > 0.7)
		img_cols++;
	int img_groups = 1;
	double relative_width = paras[2] * (imageRelativeWidth.second - imageRelativeWidth.first) + imageRelativeWidth.first;
	double relative_height = paras[3] * (imageRelativeHeight.second - imageRelativeHeight.first) + imageRelativeHeight.first;
	std::vector<double> results;
	results.push_back(img_rows);
	results.push_back(img_cols);
	results.push_back(img_groups);
	results.push_back(relative_width);
	results.push_back(relative_height);
	return results;
}

std::vector<double> grammar2(std::string modeljson, std::vector<double> paras, bool bDebug) {
	FILE* fp = fopen(modeljson.c_str(), "rb"); // non-Windows use "r"
	char readBuffer[10240];
	rapidjson::FileReadStream isModel(fp, readBuffer, sizeof(readBuffer));
	rapidjson::Document docModel;
	docModel.ParseStream(isModel);
	rapidjson::Value& grammars = docModel["grammars"];
	rapidjson::Value& grammar = grammars["grammar2"];
	// range of Rows
	std::vector<double> tmp_array = read1DArray(grammar, "rangeOfRows");
	std::pair<int, int> imageRows(tmp_array[0], tmp_array[1]);
	// range of Cols
	tmp_array.empty();
	tmp_array = read1DArray(grammar, "rangeOfCols");
	std::pair<int, int> imageCols(tmp_array[0], tmp_array[1]);
	// range of Doors
	tmp_array.empty();
	tmp_array = read1DArray(grammar, "rangeOfDoors");
	std::pair<int, int> imageDoors(tmp_array[0], tmp_array[1]);
	// relativeWidth
	tmp_array.empty();
	tmp_array = read1DArray(grammar, "relativeWidth");
	std::pair<double, double> imageRelativeWidth(tmp_array[0], tmp_array[1]);
	// relativeHeight
	tmp_array.empty();
	tmp_array = read1DArray(grammar, "relativeHeight");
	std::pair<double, double> imageRelativeHeight(tmp_array[0], tmp_array[1]);
	// relativeDWidth
	tmp_array.empty();
	tmp_array = read1DArray(grammar, "relativeDWidth");
	std::pair<double, double> imageDRelativeWidth(tmp_array[0], tmp_array[1]);
	// relativeDHeight
	tmp_array.empty();
	tmp_array = read1DArray(grammar, "relativeDHeight");
	std::pair<double, double> imageDRelativeHeight(tmp_array[0], tmp_array[1]);
	if (bDebug) {
		std::cout << "imageRows is " << imageRows.first << ", " << imageRows.second << std::endl;
		std::cout << "imageCols is " << imageCols.first << ", " << imageCols.second << std::endl;
		std::cout << "imageDoors is " << imageDoors.first << ", " << imageDoors.second << std::endl;
		std::cout << "imageRelativeWidth is " << imageRelativeWidth.first << ", " << imageRelativeWidth.second << std::endl;
		std::cout << "imageRelativeHeight is " << imageRelativeHeight.first << ", " << imageRelativeHeight.second << std::endl;
		std::cout << "imageDRelativeWidth is " << imageDRelativeWidth.first << ", " << imageDRelativeWidth.second << std::endl;
		std::cout << "imageDRelativeHeight is " << imageDRelativeHeight.first << ", " << imageDRelativeHeight.second << std::endl;
	}
	fclose(fp);
	int img_rows = paras[0] * (imageRows.second - imageRows.first) + imageRows.first;
	if (paras[0] * (imageRows.second - imageRows.first) + imageRows.first - img_rows > 0.7)
		img_rows++;
	int img_cols = paras[1] * (imageCols.second - imageCols.first) + imageCols.first;
	if (paras[1] * (imageCols.second - imageCols.first) + imageCols.first - img_cols > 0.7)
		img_cols++;
	int img_groups = 1;
	int img_doors = paras[2] * (imageDoors.second - imageDoors.first) + imageDoors.first;
	if (paras[2] * (imageDoors.second - imageDoors.first) + imageDoors.first - img_doors > 0.7)
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

std::vector<double> grammar3(std::string modeljson, std::vector<double> paras, bool bDebug) {
	FILE* fp = fopen(modeljson.c_str(), "rb"); // non-Windows use "r"
	char readBuffer[10240];
	rapidjson::FileReadStream isModel(fp, readBuffer, sizeof(readBuffer));
	rapidjson::Document docModel;
	docModel.ParseStream(isModel);
	rapidjson::Value& grammars = docModel["grammars"];
	rapidjson::Value& grammar = grammars["grammar3"];
	// range of Cols
	std::vector<double> tmp_array = read1DArray(grammar, "rangeOfCols");
	std::pair<int, int> imageCols(tmp_array[0], tmp_array[1]);
	// relativeWidth
	tmp_array.empty();
	tmp_array = read1DArray(grammar, "relativeWidth");
	std::pair<double, double> imageRelativeWidth(tmp_array[0], tmp_array[1]);
	if (bDebug) {
		std::cout << "imageCols is " << imageCols.first << ", " << imageCols.second << std::endl;
		std::cout << "imageRelativeWidth is " << imageRelativeWidth.first << ", " << imageRelativeWidth.second << std::endl;
	}
	fclose(fp);
	int img_rows = 1;
	int img_cols = paras[0] * (imageCols.second - imageCols.first) + imageCols.first;
	if (paras[0] * (imageCols.second - imageCols.first) + imageCols.first - img_cols > 0.7)
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

std::vector<double> grammar4(std::string modeljson, std::vector<double> paras, bool bDebug) {
	FILE* fp = fopen(modeljson.c_str(), "rb"); // non-Windows use "r"
	char readBuffer[10240];
	rapidjson::FileReadStream isModel(fp, readBuffer, sizeof(readBuffer));
	rapidjson::Document docModel;
	docModel.ParseStream(isModel);
	rapidjson::Value& grammars = docModel["grammars"];
	rapidjson::Value& grammar = grammars["grammar4"];
	// range of Rows
	std::vector<double> tmp_array = read1DArray(grammar, "rangeOfCols");
	std::pair<int, int> imageCols(tmp_array[0], tmp_array[1]);
	// range of Doors
	tmp_array.empty();
	tmp_array = read1DArray(grammar, "rangeOfDoors");
	std::pair<int, int> imageDoors(tmp_array[0], tmp_array[1]);
	// relativeWidth
	tmp_array.empty();
	tmp_array = read1DArray(grammar, "relativeWidth");
	std::pair<double, double> imageRelativeWidth(tmp_array[0], tmp_array[1]);
	// relativeDWidth
	tmp_array.empty();
	tmp_array = read1DArray(grammar, "relativeDWidth");
	std::pair<double, double> imageDRelativeWidth(tmp_array[0], tmp_array[1]);
	// relativeDHeight
	tmp_array.empty();
	tmp_array = read1DArray(grammar, "relativeDHeight");
	std::pair<double, double> imageDRelativeHeight(tmp_array[0], tmp_array[1]);
	if (bDebug) {
		std::cout << "imageCols is " << imageCols.first << ", " << imageCols.second << std::endl;
		std::cout << "imageDoors is " << imageDoors.first << ", " << imageDoors.second << std::endl;
		std::cout << "imageRelativeWidth is " << imageRelativeWidth.first << ", " << imageRelativeWidth.second << std::endl;
		std::cout << "imageDRelativeWidth is " << imageDRelativeWidth.first << ", " << imageDRelativeWidth.second << std::endl;
		std::cout << "imageDRelativeHeight is " << imageDRelativeHeight.first << ", " << imageDRelativeHeight.second << std::endl;
	}
	fclose(fp);
	int img_rows = 1;;
	int img_cols = paras[0] * (imageCols.second - imageCols.first) + imageCols.first;
	if (paras[0] * (imageCols.second - imageCols.first) + imageCols.first - img_cols > 0.7)
		img_cols++;
	int img_groups = 1;
	int img_doors = paras[1] * (imageDoors.second - imageDoors.first) + imageDoors.first;
	if (paras[1] * (imageDoors.second - imageDoors.first) + imageDoors.first - img_doors > 0.7)
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

std::vector<double> grammar5(std::string modeljson, std::vector<double> paras, bool bDebug) {
	FILE* fp = fopen(modeljson.c_str(), "rb"); // non-Windows use "r"
	char readBuffer[10240];
	rapidjson::FileReadStream isModel(fp, readBuffer, sizeof(readBuffer));
	rapidjson::Document docModel;
	docModel.ParseStream(isModel);
	rapidjson::Value& grammars = docModel["grammars"];
	rapidjson::Value& grammar = grammars["grammar5"];
	// range of Rows
	std::vector<double> tmp_array = read1DArray(grammar, "rangeOfRows");
	std::pair<int, int> imageRows(tmp_array[0], tmp_array[1]);
	// relativeHeight
	tmp_array.empty();
	tmp_array = read1DArray(grammar, "relativeHeight");
	std::pair<double, double> imageRelativeHeight(tmp_array[0], tmp_array[1]);
	if (bDebug) {
		std::cout << "imageRows is " << imageRows.first << ", " << imageRows.second << std::endl;
		std::cout << "imageRelativeHeight is " << imageRelativeHeight.first << ", " << imageRelativeHeight.second << std::endl;
	}
	fclose(fp);
	int img_rows = paras[0] * (imageRows.second - imageRows.first) + imageRows.first;
	if (paras[0] * (imageRows.second - imageRows.first) + imageRows.first - img_rows > 0.7)
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

std::vector<double> grammar6(std::string modeljson, std::vector<double> paras, bool bDebug) {
	FILE* fp = fopen(modeljson.c_str(), "rb"); // non-Windows use "r"
	char readBuffer[10240];
	rapidjson::FileReadStream isModel(fp, readBuffer, sizeof(readBuffer));
	rapidjson::Document docModel;
	docModel.ParseStream(isModel);
	rapidjson::Value& grammars = docModel["grammars"];
	rapidjson::Value& grammar = grammars["grammar6"];
	// range of Rows
	std::vector<double> tmp_array = read1DArray(grammar, "rangeOfRows");
	std::pair<int, int> imageRows(tmp_array[0], tmp_array[1]);
	// range of Doors
	tmp_array.empty();
	tmp_array = read1DArray(grammar, "rangeOfDoors");
	std::pair<int, int> imageDoors(tmp_array[0], tmp_array[1]);
	// relativeHeight
	tmp_array.empty();
	tmp_array = read1DArray(grammar, "relativeHeight");
	std::pair<double, double> imageRelativeHeight(tmp_array[0], tmp_array[1]);
	// relativeDWidth
	tmp_array.empty();
	tmp_array = read1DArray(grammar, "relativeDWidth");
	std::pair<double, double> imageDRelativeWidth(tmp_array[0], tmp_array[1]);
	// relativeDHeight
	tmp_array.empty();
	tmp_array = read1DArray(grammar, "relativeDHeight");
	std::pair<double, double> imageDRelativeHeight(tmp_array[0], tmp_array[1]);
	if (bDebug) {
		std::cout << "imageRows is " << imageRows.first << ", " << imageRows.second << std::endl;
		std::cout << "imageDoors is " << imageDoors.first << ", " << imageDoors.second << std::endl;
		std::cout << "imageRelativeHeight is " << imageRelativeHeight.first << ", " << imageRelativeHeight.second << std::endl;
		std::cout << "imageDRelativeWidth is " << imageDRelativeWidth.first << ", " << imageDRelativeWidth.second << std::endl;
		std::cout << "imageDRelativeHeight is " << imageDRelativeHeight.first << ", " << imageDRelativeHeight.second << std::endl;
	}
	fclose(fp);
	int img_rows = paras[0] * (imageRows.second - imageRows.first) + imageRows.first;
	if (paras[0] * (imageRows.second - imageRows.first) + imageRows.first - img_rows > 0.7)
		img_rows++;
	int img_cols = 1;
	int img_groups = 1;
	int img_doors = paras[1] * (imageDoors.second - imageDoors.first) + imageDoors.first;
	if (paras[1] * (imageDoors.second - imageDoors.first) + imageDoors.first - img_doors > 0.7)
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
