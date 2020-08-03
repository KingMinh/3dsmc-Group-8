#pragma once

#include "ImageSource.h"

class Segmentation {

protected:
	cv::Mat mask;

public:
	Segmentation() = default;

	virtual ~Segmentation() = default;

	virtual void update(ImageSource &image) = 0;

	inline cv::Mat &get_mask() { return mask; }

	inline const cv::Mat &get_mask() const { return mask; }
};

class ChromaSegmentation : public Segmentation {

private:
	cv::Scalar lower;
	cv::Scalar upper;

public:
	ChromaSegmentation(cv::Scalar lower, cv::Scalar upper);

	void update(ImageSource &image) override;

public:
	static std::unique_ptr<ChromaSegmentation> Green();

	static std::unique_ptr<ChromaSegmentation> Blue();

	static std::unique_ptr<ChromaSegmentation> White();
};

class CleanplateSegmentation : public Segmentation {

protected:
	cv::Mat firstFrame;

public:
	explicit CleanplateSegmentation(const std::string &cleanPlatePath);

	void update(ImageSource &image) override;
};

class WatershedSegmentation : public Segmentation {
public:
	WatershedSegmentation();

	void update(ImageSource &image) override;
};

