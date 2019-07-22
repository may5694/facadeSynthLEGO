#include <iostream>
#include "utils.hpp"

namespace util {

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
}
