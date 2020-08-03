#include "Marker.h"

Marker::Marker(ImageSource &image, float markerLength)
		: image(image) {
	parameters->cornerRefinementMethod = cv::aruco::CORNER_REFINE_CONTOUR;
	cv::aruco::detectMarkers(image.get_frame(), dictionary, corners, ids, parameters, rejected);
	cv::aruco::estimatePoseSingleMarkers(corners, markerLength,
	                                     image.get_camera_matrix(), image.get_distortion_coefficients(),
	                                     rotationVectors, translationVectors);
}

cv::Mat Marker::visualize() {
	cv::Mat img = image.get_frame().clone();

	cv::aruco::drawDetectedMarkers(img, corners, ids);

	for (size_t i = 0; i < ids.size(); i++)
		cv::aruco::drawAxis(img, image.get_camera_matrix(), image.get_distortion_coefficients(), rotationVectors[i],
		                    translationVectors[i], 0.1);

	return img;
}

inline cv::Vec3d combineRotations(const cv::Vec3d &a, const cv::Vec3d &b) {
	cv::Mat ma, mb;
	cv::Rodrigues(a, ma);
	cv::Rodrigues(b, mb);
	cv::Mat mr = ma * mb;
	cv::Vec3d r;
	cv::Rodrigues(mr, r);
	return r;
}

std::optional<MarkerTracker::loc> MarkerTracker::getFirstMarkerLoc(Marker &mark) {
	if (mark.ids.empty()) return {};
	if (!first) {
		first = mark.ids.front();
		std::cout << "First marker: " << *first << '\n';
		markers.emplace(*first, loc{0, 0});
	}

	loc firstPose{0, 0};
	bool poseReliable = false;
	if (auto it = std::find(mark.ids.begin(), mark.ids.end(), first.value()); it != mark.ids.end()) {
		poseReliable = true;
		auto idx = std::distance(mark.ids.begin(), it);
		firstPose.translation = mark.translationVectors[idx];
		firstPose.rotation = mark.rotationVectors[idx];
	} else {
		// estimate first marker pose from other markers.
		int count = 0;
		for (size_t i = 0; i < mark.ids.size(); i++) {
			auto id = mark.ids[i];
			if (auto search = markers.find(id); search != markers.end()) {
				count++;
				firstPose.translation += mark.translationVectors[i] + search->second.translation;
				// This is technically not accurate. But normally, these are all close to one another so it shouldn't matter.
				firstPose.rotation += combineRotations(mark.rotationVectors[i], search->second.rotation);
			}
		}

		if (count > 0) {
			firstPose.translation *= 1.0 / count;
			firstPose.rotation *= 1.0 / count;
		} else {
			return {};
		}
	}

	// Add new markers if any exist.
	for (size_t i = 0; i < mark.ids.size(); i++) {
		auto id = mark.ids[i];
		if (id == first) continue;
		// pose: marker i in camera space; firstPose: first marker in camera space.
		// wanted: first marker in marker i space = inv(pose) . firstPose

		// taking the negative here inverts the rotation axis, but keeps the amplitude (=angle)
		loc relative{firstPose.translation - mark.translationVectors[i],
		             combineRotations(-mark.rotationVectors[i], firstPose.rotation)};
		if (poseReliable) markers.insert_or_assign(id, relative);
		else markers.emplace(id, relative); // does not override
	}

	return firstPose;
}
