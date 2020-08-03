#include "ImageSource.h"

ImageSource::ImageSource(const std::string &config_filename) {
	cv::FileStorage fs(config_filename, cv::FileStorage::READ);
	if (!fs.isOpened()) {
		return;
	}
	fs["camera_matrix"] >> camera_matrix;
	fs["distortion_coefficients"] >> distortion_coefficients;
}

const cv::Mat &ImageSource::get_camera_matrix() const {
	return camera_matrix;
}

const cv::Mat &ImageSource::get_distortion_coefficients() const {
	return distortion_coefficients;
}

StillImageSource::StillImageSource(const std::string &image_filename, const std::string &config_filename)
		: ImageSource(config_filename) {
	frame = cv::imread(image_filename, 1);
}

VideoImageSource::VideoImageSource(const std::string &video_filename, const std::string &config_filename)
		: ImageSource(config_filename), capture(video_filename) {
	capture >> frame;
}

VideoImageSource::VideoImageSource(int deviceIdx, const std::string &config_filename)
		: ImageSource(config_filename), capture(deviceIdx) {
	capture >> frame;
}

VideoImageSource::~VideoImageSource() = default;

bool VideoImageSource::next() {
	capture >> frame;
	return !frame.empty();
}