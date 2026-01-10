#pragma once

#include <string>
#include "alpha_hist/image.hpp"

namespace alpha_hist {
    
    ImageRGBA8 load_rgba8(const std::string& path);

    ImageGray8 load_gray8(const std::string& path);

    void save_png(const std::string& path, const ImageRGBA8& img);

    void save_png(const std::string& path, const ImageGray8& img);

} // namespace alpha_hist