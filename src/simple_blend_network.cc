// partly adapted from:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/main.cc
// see also: https://www.tensorflow.org/versions/master/api_guides/cc/guide

#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <Eigen/Core>

#include <cuda_gl_interop.h>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/util/events_writer.h"

#include "custom_ops.h"
#include "gl_wrappers.h"

//------------------------------------------------------------------------------

void AddGraphInputsAndOutputOps(
    tensorflow::GraphDef* graph_def,
    const std::vector<std::unique_ptr<fribr::Texture>>& input_textures,
    const std::vector<std::string>& input_graph_node_names,
    const fribr::Texture& output_texture, const std::string& output_node_name,
    const GLFWwindow* window) {
  // Replace placeholder ops with TextureInput ops.
  {
    for (size_t i = 0; i < input_textures.size(); ++i) {
      for (tensorflow::NodeDef& node : *graph_def->mutable_node()) {
        if (node.name() == input_graph_node_names[i]) {
          node.set_op("TextureInput");
          node.clear_attr();

          (*node.mutable_attr())["GLFWwindow_ptr"].set_i(
              reinterpret_cast<const int64_t>(window));

          (*node.mutable_attr())["texture_id"].set_i(
              input_textures[i]->get_id());

          const auto input_shape = input_textures[i]->get_resolution();
          auto shape = (*node.mutable_attr())["shape"].mutable_shape();
          shape->add_dim()->set_size(input_shape[1]);
          shape->add_dim()->set_size(input_shape[0]);
          shape->add_dim()->set_size(3);  // RGB
        }
      }
    }
  }

  // TODO (True): potential bug in Protobuf and/or the way this project uses the
  // Protobuf library? Have to force the release of fields before setting them
  // (they otherwise point to the same memory; this doesn't happen using bazel,
  // though...).
  {
    auto node = graph_def->add_node();
    node->release_name();
    node->release_op();
    node->set_name("output");
    node->set_op("CopyToTexture");
    node->add_input(output_node_name);
    (*node->mutable_attr())["GLFWwindow_ptr"].set_i(
        reinterpret_cast<const int64_t>(window));
    (*node->mutable_attr())["texture_id"].set_i(output_texture.get_id());
  }
}

//------------------------------------------------------------------------------

// Reads a model graph definition from disk.
tensorflow::Status LoadGraph(
    std::unique_ptr<tensorflow::Session>* session,
    const std::string& graph_file_name,
    const std::vector<std::unique_ptr<fribr::Texture>>& input_textures,
    const std::vector<std::string>& input_graph_node_names,
    const fribr::Texture& output_texture,
    const std::string& output_node_name, const GLFWwindow* window) {
  tensorflow::GraphDef graph_def;
  tensorflow::Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }

  AddGraphInputsAndOutputOps(&graph_def, input_textures,
                             input_graph_node_names, output_texture,
                             output_node_name, window);

  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  return (*session)->Create(graph_def);
}

void gl_error_callback(int error, const char* description) {
  LOG(ERROR) << "GL Error " << error << " " << description;
}

bool InitializeGL(size_t width, size_t height, GLFWwindow** window) {
  glfwSetErrorCallback(gl_error_callback);
  if (!glfwInit()) {
    LOG(ERROR) << "Could not initialize glfw." << std::endl;
    return false;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_SAMPLES, 4);
  *window = glfwCreateWindow(width, height, "IBR", nullptr, nullptr);
  glfwMakeContextCurrent(*window);
  glfwSwapInterval(0);

  glewExperimental = GL_TRUE;
  GLenum glew_error = glewInit();

  if (GLEW_OK != glew_error) {
    LOG(ERROR) << "GLEW init failed: " << glewGetErrorString(glew_error);
    return false;
  }

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);
  glFrontFace(GL_CCW);
}

//------------------------------------------------------------------------------

void UploadImageToTexture(const cv::Mat& image,
                          std::unique_ptr<fribr::Texture>& texture) {
  GLuint texture_id = texture->get_id();

  glBindTexture(GL_TEXTURE_2D, texture_id);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  // use fast 4-byte alignment always
  glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

  // set length of one complete row in data (doesn't need to equal image.cols)
  glPixelStorei(GL_UNPACK_ROW_LENGTH, image.step / image.elemSize());

  cv::Mat copy(image.rows, image.cols, CV_8UC4);
  for (int y = 0; y < image.rows; ++y) {
    for (int x = 0; x < image.cols; ++x) {
      cv::Vec3b c = image.at<cv::Vec3b>(y, x);
      copy.at<cv::Vec4b>(y, x) = cv::Vec4b(c[2], c[1], c[0], 255);
    }
  }

  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, copy.cols, copy.rows, GL_BGRA,
                  GL_UNSIGNED_BYTE, copy.data);

  glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);

  glBindTexture(GL_TEXTURE_2D, 0);
}

//------------------------------------------------------------------------------

cv::Mat DownloadTexture(fribr::Texture* texture) {
  GLuint texture_id = texture->get_id();

  const auto resolution = texture->get_resolution();
  cv::Mat image(resolution.y(), resolution.x(), CV_8UC4);

  glBindTexture(GL_TEXTURE_2D, texture_id);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, image.data);
  glBindTexture(GL_TEXTURE_2D, 0);

  return image;
}

//------------------------------------------------------------------------------

int main(int argc, char** argv) {
  cudaGLSetGLDevice(0);

  // TODO (True): create command-line args
  const std::string data_folder =
      "/playpen/jtprice/research/ibr_2018/data/";
  const std::string data_file = data_folder + "test.txt";
  const std::string output_folder = data_folder + "output/";
  const std::string graph_path = data_folder + "model/model.pb";
  const std::string log_folder = data_folder + "logs/";
  const size_t width = 1296;
  const size_t height = 832;

  const std::vector<tensorflow::Flag> flag_list;

  const std::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  if (!tensorflow::Flags::Parse(&argc, argv, flag_list)) {
    LOG(ERROR) << usage;
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  // set the device to use for the CUDA/GL interop
  GLFWwindow* window;

  if (!InitializeGL(width, height, &window)) {
    return -1;
  }

  //
  // Set up input and output frame buffers.
  //

  // TODO (True): make these command-line inputs
  const std::vector<std::string> input_graph_node_names = {
      "input1", "input2", "input3", "input4", "input5"};

  const std::string output_node_name = "model/add_4";

  const auto resolution = (Eigen::Vector2i() << width, height).finished();

  fribr::Texture::Descriptor color_descriptor(GL_CLAMP_TO_EDGE, GL_NEAREST, 0,
                                              fribr::TextureFormat::RGBA8);
  std::vector<std::unique_ptr<fribr::Texture>> input_textures;
  for (size_t i = 0; i < input_graph_node_names.size(); ++i) {
    input_textures.emplace_back(
        new fribr::Texture(resolution, color_descriptor));
  }

  fribr::Texture output_texture(resolution, color_descriptor);

  //
  // Load and initialize model from Protobuf file.
  //

  std::unique_ptr<tensorflow::Session> session;
  {
    const auto load_graph_status = LoadGraph(
        &session, graph_path, input_textures, input_graph_node_names,
        output_texture, output_node_name, window);
    if (!load_graph_status.ok()) {
      LOG(ERROR) << load_graph_status;
      return -1;
    }
  }

  //
  // Run the model.
  //

  size_t output_idx = 0;

  std::ifstream fin(data_file);
  size_t dummy;

  // Create the output folder where images are saved, if it doesn't exist.
  tensorflow::Env::Default()->RecursivelyCreateDir(output_folder);


  while ((fin >> dummy >> dummy)) {
    std::string filename;

    // extra reference file
    fin >> filename;

    glfwMakeContextCurrent(window);
    for (size_t i = 0; i < 5; ++i) {
      fin >> filename;
      UploadImageToTexture(
          cv::imread(tensorflow::io::JoinPath(data_folder, filename)),
          input_textures[i]);
    }
    glfwMakeContextCurrent(0);

    // extra files
    for (size_t i = 0; i < 10; ++i) {
      fin >> filename;
    }

    {
      auto start = std::chrono::high_resolution_clock::now();
      const auto run_status = session->Run({}, {}, {"output"}, {});
      if (!run_status.ok()) {
        LOG(ERROR) << run_status;
        return -1;
      }
      const std::chrono::duration<double, std::milli> duration =
          std::chrono::high_resolution_clock::now() - start;
      LOG(INFO) << "Network ran in " << duration.count() << " ms";
    }

    // Test saving
    std::ostringstream filename_ss;
    filename_ss << std::setfill('0') << std::setw(5) << output_idx << ".png";
    filename = tensorflow::io::JoinPath(output_folder, filename_ss.str());

    glfwMakeContextCurrent(window);
    cv::Mat image = DownloadTexture(&output_texture);
    glfwMakeContextCurrent(0);
    cv::imwrite(filename, image);

    ++output_idx;
  }

  fin.close();

  return 0;
}
