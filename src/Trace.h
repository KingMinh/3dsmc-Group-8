#pragma once

#include <string>
#include <utility>
#include <chrono>
#include <optional>
#include <iostream>

class Trace {
private:
	std::string name;
	std::chrono::high_resolution_clock::time_point start;
	std::ostream &outfile;
	bool ended = false;
public:
	explicit Trace(std::string name, std::ostream *out = nullptr) : name(std::move(name)),
	                                                                outfile(out ? *out : std::cerr) {
		start = std::chrono::high_resolution_clock::now();
	}

	void end() {
		if (ended) return;
		auto end = std::chrono::high_resolution_clock::now();
		auto dur = end - start;
		double secs = std::chrono::duration_cast<std::chrono::duration<double>>(dur).count();
		outfile << name << ": " << secs << 's' << std::endl;
		ended = true;
	}

	~Trace() {
		end();
	}

	template<typename F, typename ...Args>
	static auto call(std::string name, F fn, Args &&...args) {
		Trace trace(std::move(name));
		return fn(std::forward<Args>(args)...);
	}
};
