#ifndef OFFLINE_DATA_LOADER_H_

#include <array>
#include <fstream>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"

class OfflineDataLoader {
 public:
  static const size_t NUM_MOSAIC_IMAGES = 4;

  static const std::vector<std::string> MOSAIC_NAMES;

  OfflineDataLoader(const std::string& data_folder,
                    const std::string& filenames_file)
      : data_folder_(data_folder),
        fin_(filenames_file),
        session_(tensorflow::NewSession(tensorflow::SessionOptions())) {
    Initialize_();
  }

  ~OfflineDataLoader() {
    fin_.close();
  }

  std::vector<std::pair<std::string, tensorflow::Tensor>> Next() {
    size_t width = static_cast<size_t>(-1), height = static_cast<size_t>(-1);
    fin_ >> width >> height;

    if (width == static_cast<size_t>(-1) || height == static_cast<size_t>(-1)) {
      return {};
    }

    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
    std::vector<std::string> output_names;

    std::string filename;

    output_names.push_back("ref_image");
    inputs.emplace_back(
        output_names.back() + "_filename",
        tensorflow::Tensor(tensorflow::DT_STRING, tensorflow::TensorShape()));
    fin_ >> filename;
    inputs.back().second.scalar<std::string>()() =
        tensorflow::io::JoinPath(data_folder_, filename);

    output_names.push_back("global_color");
    inputs.emplace_back(
        output_names.back() + "_filename",
        tensorflow::Tensor(tensorflow::DT_STRING, tensorflow::TensorShape()));
    fin_ >> filename;
    inputs.back().second.scalar<std::string>()() =
        tensorflow::io::JoinPath(data_folder_, filename);

    output_names.push_back("global_disparity");
    inputs.emplace_back(
        output_names.back() + "_filename",
        tensorflow::Tensor(tensorflow::DT_STRING, tensorflow::TensorShape()));
    fin_ >> filename;
    inputs.back().second.scalar<std::string>()() =
        tensorflow::io::JoinPath(data_folder_, filename);

    fin_ >> filename;  // global normals
    fin_ >> filename;  // global output_dir

    for (size_t i = 0; i < NUM_MOSAIC_IMAGES; ++i) {
      const std::string suffix = "_" + std::to_string(i);

      for (const auto& name : MOSAIC_NAMES) {
        output_names.push_back(name + suffix);
        inputs.emplace_back(output_names.back() + "_filename",
                            tensorflow::Tensor(tensorflow::DT_STRING,
                                               tensorflow::TensorShape()));
        fin_ >> filename;
        inputs.back().second.scalar<std::string>()() =
            tensorflow::io::JoinPath(data_folder_, filename);
      }
    }

    std::vector<tensorflow::Tensor> out_tensors;
    session_->Run(inputs, output_names, {}, &out_tensors);

    // TODO (True): this is all very hard-coded right now.
    return {{"inputloader/convert_image_1", out_tensors[1]},
            {"inputloader/convert_image_2", out_tensors[2]},
            {"inputloader/convert_image_5", out_tensors[3]},
            {"inputloader/convert_image_6", out_tensors[4]},
            {"inputloader/convert_image_7", out_tensors[5]},
            {"inputloader/convert_image_8", out_tensors[6]},
            {"inputloader/convert_image_9", out_tensors[7]},
            {"inputloader/convert_image_10", out_tensors[8]},
            {"inputloader/convert_image_11", out_tensors[9]},
            {"inputloader/convert_image_12", out_tensors[10]},
            {"inputloader/convert_image_13", out_tensors[11]},
            {"inputloader/convert_image_14", out_tensors[12]},
            {"inputloader/convert_image_15", out_tensors[13]},
            {"inputloader/convert_image_16", out_tensors[14]},
            {"inputloader/convert_image_17", out_tensors[15]},
            {"inputloader/convert_image_18", out_tensors[16]},
            {"inputloader/convert_image_19", out_tensors[17]},
            {"inputloader/convert_image_20", out_tensors[18]},
            {"inputloader/convert_image_21", out_tensors[19]},
            {"inputloader/convert_image_22", out_tensors[10]},
            {"inputloader/convert_image_23", out_tensors[21]},
            {"inputloader/convert_image_24", out_tensors[22]}};
  }

 private:
  void Initialize_() {
    tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

    AddImageInputOp_("JPG", "ref_image", scope);
    AddImageInputOp_("JPG", "global_color", scope);
    AddImageInputOp_("PNG", "global_disparity", scope, 1);
    for (size_t i = 0; i < NUM_MOSAIC_IMAGES; ++i) {
      const std::string suffix = "_" + std::to_string(i);
      AddImageInputOp_("JPG", "colors" + suffix, scope);
      AddImageInputOp_("PNG", "disparities" + suffix, scope, 1);
      AddImageInputOp_("PNG", "input_dir" + suffix, scope);
      AddImageInputOp_("PNG", "normals" + suffix, scope);
      AddImageInputOp_("PNG", "output_dir" + suffix, scope);
    }

    tensorflow::GraphDef graph;
    scope.ToGraphDef(&graph);
    session_->Create(graph);
  }

  // Given an image file name, read in the data, try to decode it as an image,
  // resize it to the requested size, and then scale the values as desired.
  void AddImageInputOp_(const std::string& image_type,
                        const std::string& output_name, tensorflow::Scope scope,
                        const size_t num_channels = 3) {
    auto file_name = tensorflow::ops::Placeholder(
        scope.WithOpName(output_name + "_filename"),
        tensorflow::DataType::DT_STRING);

    auto file_reader = tensorflow::ops::ReadFile(scope, file_name);

    tensorflow::Output image_reader;
    if (image_type == "PNG") {
      image_reader = tensorflow::ops::DecodePng(
          scope.WithOpName("png_reader"), file_reader,
          tensorflow::ops::DecodePng::Channels(num_channels));
    } else if (image_type == "GIF") {
      // gif decoder returns 4-D tensor, remove the first dim
      image_reader = tensorflow::ops::Squeeze(
          scope.WithOpName("squeeze_first_dim"),
          tensorflow::ops::DecodeGif(scope.WithOpName("gif_reader"),
                                     file_reader));
    } else if (image_type == "BMP") {
      image_reader = tensorflow::ops::DecodeBmp(scope.WithOpName("bmp_reader"),
                                                file_reader);
    } else if (image_type == "JPG") {
      image_reader = tensorflow::ops::DecodeJpeg(
          scope.WithOpName("jpeg_reader"), file_reader,
          tensorflow::ops::DecodeJpeg::Channels(num_channels));
    }

    auto float_caster =
        tensorflow::ops::Cast(scope, image_reader, tensorflow::DT_FLOAT);
    auto scaler =
        tensorflow::ops::Multiply(scope.WithOpName(output_name), float_caster,
                                  static_cast<float>(1. / 255.));
  }

  const std::string data_folder_;
  std::ifstream fin_;

  std::unique_ptr<tensorflow::Session> session_;
};

const std::vector<std::string> OfflineDataLoader::MOSAIC_NAMES = {
    "colors", "disparities", "input_dir", "normals", "output_dir"};

#endif  // OFFLINE_DATA_LOADER_H_
