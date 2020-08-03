#pragma once

#include <string>
#include <opencv2/opencv.hpp>

class ImageSource {
protected:
	cv::Mat camera_matrix;
	cv::Mat distortion_coefficients;

	cv::Mat frame;

public:
	explicit ImageSource(const std::string &config_filename);

	virtual ~ImageSource() = default;

	virtual const cv::Mat &get_camera_matrix() const;

	virtual const cv::Mat &get_distortion_coefficients() const;

	inline cv::Mat &get_frame() { return frame; }

	inline const cv::Mat &get_frame() const { return frame; }

	virtual bool is_open() const = 0;

	virtual bool next() = 0;
};

class StillImageSource : public ImageSource {
public:
	StillImageSource(const std::string &image_filename, const std::string &config_filename);

	inline bool is_open() const override { return !frame.empty(); }

	inline bool next() override { return false; };
};

class VideoImageSource : public ImageSource {

private:
	cv::VideoCapture capture;

public:
	VideoImageSource(const std::string &video_filename, const std::string &config_filename);

	VideoImageSource(int deviceIdx, const std::string &config_filename);

	~VideoImageSource() override;

	inline bool is_open() const override { return capture.isOpened(); }

	bool next() override;
};