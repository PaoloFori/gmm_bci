#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <string>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <fstream>

template<typename T>
void writeCSV(const std::string& filename, const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>&  matrix) {
	const static Eigen::IOFormat format(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");

	std::ofstream file(filename);
	if (file.is_open()) {
		file << matrix.format(format);
		file.close();
	}
}

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> readCSV(const std::string& filename) {

	std::vector<T> values;

	std::ifstream file(filename);
	std::string row;
	std::string entry;
	int nrows = 0;

	while (getline(file, row)) {
		std::stringstream rowstream(row);

		while (getline(rowstream, entry, ',')) {
			values.push_back(std::stod(entry));
		}
		nrows++; 
	}

	return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(values.data(), nrows, values.size() / nrows);

}


#endif