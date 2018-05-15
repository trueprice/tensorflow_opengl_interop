#ifndef OFFLINE_DATA_LOADER_H_

#include <array>
#include <fstream>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"

class OfflineDataLoader {
 public:
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

//    output_names.push_back("ref_image");
//    inputs.emplace_back(
//        output_names.back() + "_filename",
//        tensorflow::Tensor(tensorflow::DT_STRING, tensorflow::TensorShape()));
    fin_ >> filename;
//    inputs.back().second.scalar<std::string>()() =
//        tensorflow::io::JoinPath(data_folder_, filename);

    for (size_t i = 0; i < 5; ++i) {
      const std::string name = "image_" + std::to_string(i);

      output_names.push_back(name);
      inputs.emplace_back(
          name + "_filename",
          tensorflow::Tensor(tensorflow::DT_STRING, tensorflow::TensorShape()));
      fin_ >> filename;
      inputs.back().second.scalar<std::string>()() =
          tensorflow::io::JoinPath(data_folder_, filename);
    }

    // read extra filenames
    for (size_t i = 0; i < 10; ++i) {
      fin_ >> filename;
    }

    std::vector<tensorflow::Tensor> out_tensors;
    session_->Run(inputs, output_names, {}, &out_tensors);

    return {{"inputloader/convert_image_1", out_tensors[0]},
            {"inputloader/convert_image_2", out_tensors[1]},
            {"inputloader/convert_image_3", out_tensors[2]},
            {"inputloader/convert_image_4", out_tensors[3]},
            {"inputloader/convert_image_5", out_tensors[4]}};
  }

 private:
  void Initialize_() {
    tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

    for (size_t i = 0; i < 5; ++i) {
      const std::string suffix = "_" + std::to_string(i);
      AddImageInputOp_("JPG", "image" + suffix, scope);
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

#endif  // OFFLINE_DATA_LOADER_H_
