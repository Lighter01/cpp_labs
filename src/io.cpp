#include "alpha_hist/io.hpp"

#include <stdexcept>
#include <string>

#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

namespace alpha_hist {
    
    static void ensure_ok(bool ok, const std::string& msg) {
        if (!ok) throw std::runtime_error(msg);
    }

    ImageRGBA8 load_rgba8(const std::string& path) {
        int w = 0, h = 0, n = 0;
        // force 4 channels
        std::uint8_t* p = stbi_load(path.c_str(), &w, &h, &n, ImageRGBA8::channels);
        if (!p) {
            throw std::runtime_error("stbi_load failed for '" + path + "': " + stbi_failure_reason());
        }

        ImageRGBA8 img;
        img.width = w;
        img.height = h;
        img.data.assign(p, p + static_cast<size_t>(w) * static_cast<size_t>(h) * ImageRGBA8::channels);
        stbi_image_free(p);
        return img;
    }


    ImageGray8 load_gray8(const std::string& path) {
        int w = 0, h = 0, n = 0;
        // force 1 channel
        std::uint8_t* p = stbi_load(path.c_str(), &w, &h, &n, ImageGray8::channels);
        if (!p) {
            throw std::runtime_error("stbi_load failed for '" + path + "': " + stbi_failure_reason());
        }

        ImageGray8 img;
        img.width = w;
        img.height = h;
        img.data.assign(p, p + static_cast<size_t>(w) * static_cast<size_t>(h));
        stbi_image_free(p);
        return img;
    }


    void save_png(const std::string& path, const ImageRGBA8& img) {
            ensure_ok(img.width > 0 && img.height > 0, "save_png: invalid image size");
            ensure_ok(static_cast<std::uint32_t>(img.data.size()) == img.width * img.height * 4, "save_png: buffer size mismatch");

            int stride = img.width * 4;
            int ok = stbi_write_png(path.c_str(), img.width, img.height, 4, img.data.data(), stride);
            ensure_ok(ok != 0, "stbi_write_png failed for '" + path + "'");
    }


    void save_png(const std::string& path, const ImageGray8& img) {
            ensure_ok(img.width > 0 && img.height > 0, "save_png: invalid image size");
            ensure_ok(static_cast<std::uint32_t>(img.data.size()) == img.width * img.height, "save_png: buffer size mismatch");

            int stride = img.width;
            int ok = stbi_write_png(path.c_str(), img.width, img.height, 1, img.data.data(), stride);
            ensure_ok(ok != 0, "stbi_write_png failed for '" + path + "'");
    }

} // namespace alpha_hist