#pragma once

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

class Grid {
public:
	Grid(int dim, float x, float y, float z);

	bool WriteMesh(const std::string &filename);
	bool WriteMeshColor(const std::string& filename);

	void Carve(cv::Vec3d t, cv::Vec3d r, cv::Mat mask, cv::Mat cam);

	/// Carve according to a plane given by a normal and the projection of any point on the plane (= shortest distance
	/// to the origin * |n|)
	void CarveClipPlane(cv::Vec3d n, double orig);
	void CarveMask(cv::InputArray tvec, cv::InputArray rvec, cv::Mat mask, cv::InputArray cameraMatrix, cv::InputArray distCoeffs);
	void CarveMaskColor(cv::InputArray tvec, cv::InputArray rvec, cv::Mat mask, cv::InputArray cameraMatrix, cv::InputArray distCoeffs,cv::Mat image);

	float x_length, y_length, z_length;
	int dimension;
	std::vector<bool> voxels;
	std::vector<uint32_t> voxelsColor;
};