#pragma once

#include <variant>
#include <optional>
#include <string>

enum class SegmentMode {
	Watershed,
	ChromaBlue,
	ChromeGreen,
	ChromaWhite,
	FirstFrame,
};

struct Arguments {
	std::variant<std::string, int, std::nullptr_t> input;
	std::string output;
	std::string config;

    SegmentMode mode;
    std::optional<std::string> cleanPlate;

    float markerLength;

    std::string get_output_filepath(const std::string& filename);
};

int parse_args(Arguments &args, int argc, char **argv);
