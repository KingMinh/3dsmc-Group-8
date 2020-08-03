#include "Args.h"
#include <argp.h>
#include <string.h>
#include <sstream>

std::string Arguments::get_output_filepath(const std::string& filename) {
    std::stringstream ss;
    ss << output << "/" << filename;
    return ss.str();
}

static char args_doc[] = "INPUT";

static char doc[] = "creates 3D mesh from RBG input sequence using voxel carving";

enum fix_args {
    FIX_ARG_INPUT = 0,
    FIX_ARG_CNT
};

static struct argp_option options[] = {
    {
        "output",
        'o',
        "dir",
        OPTION_ARG_OPTIONAL,
        "output directory for mesh and segmentation data",
        0
    },
    {
        "config",
        'c',
        "file",
        0,
        "config file for camera parameters",
        0
    },
    {
        "cleanplate",
        'p',
        "path",
        0,
        "path to cleanplate image used for foreground segmentation",
        1
    },
    {
        "watershed",
        's',
        0,
        0,
        "use watershed algorithm for segmentation",
        1
    },
    {
        "green",
        'g',
        0,
        0,
        "use chroma green as background",
        1
    },
    {
        "blue",
        'b',
        0,
        0,
        "use chroma blue as background",
        1
    },
    {
        "white",
        'w',
        0,
        0,
        "use chroma white as background",
        1
    },
    {
        "markerlength",
        'l',
        "length",
        0,
        "length of ArUco markers in meters",
        0
    },
    { 0, 0, 0, 0, 0, 0 }
};

static error_t parse_opt(int key, char* arg, struct argp_state* state);

static struct argp argp = {
    options,
    parse_opt,
    args_doc,
    doc,
    0,
    0,
    0
};

static error_t parse_opt(int key, char *arg, struct argp_state *state) {
	Arguments &args = *reinterpret_cast<Arguments *>(state->input);
	char *ptr;

	switch (key) {
		case ARGP_KEY_ARG:
			switch (state->arg_num) {
				case FIX_ARG_INPUT:
					args.input = (int) std::strtol(arg, &ptr, 10);
					if (*ptr) {
						args.input = arg;
					}
					break;
				default:
					return ARGP_ERR_UNKNOWN;
			}
            break;
        case 'o':
            args.output = arg;
            break;
        case 'c':
            args.config = arg;
            break;
        case 'p':
            args.mode = SegmentMode::FirstFrame;
            args.cleanPlate = arg;
            break;
        case 's':
            args.mode = SegmentMode::Watershed;
            break;
        case 'g':
            args.mode = SegmentMode::ChromeGreen;
            break;
        case 'b':
            args.mode = SegmentMode::ChromaBlue;
            break;
        case 'w':
            args.mode = SegmentMode::ChromaWhite;
            break;
        case 'l':
            args.markerLength = strtof(arg, &ptr);
            if (*ptr) {
                return EINVAL;
            }
        break;
	default:
		return ARGP_ERR_UNKNOWN;
	}
	return 0;
}

int parse_args(Arguments& args, int argc, char** argv) {
    args.input = nullptr;
    args.output = ".";
    args.markerLength = 0.05;

	if (argp_parse(&argp, argc, argv, 0, 0, &args))
		return -1;
	// Index 2 is nullptr_t
	if (args.input.index() == 2)
		return -1;

	if (args.config.length() == 0)
		return -1;

	return 0;
}