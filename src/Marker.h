#pragma once

#include <unordered_map>
#include <utility>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include "ImageSource.h"

class Marker {

public:
	std::vector<int> ids;
	std::vector<std::vector<cv::Point2f>> corners, rejected;

	std::vector<cv::Vec3d> translationVectors;
	std::vector<cv::Vec3d> rotationVectors;
private:

	//use standard parameters for detection
	cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();

	//define the used dictionary
	cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);

	ImageSource &image;
public:
	explicit Marker(ImageSource &image, float markerLength);

	cv::Mat visualize();
};

class MarkerTracker {
public:
	struct loc {
		cv::Vec3d translation, rotation;
	};

	std::optional<loc> getFirstMarkerLoc(Marker &mark);

private:
	// contains transformations that each marker needs to be multiplied with to get the position and rotation of the
	// "first" marker.
	std::unordered_map<int, loc> markers{};
	std::optional<int> first{};
};
