// partly adapted from:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/main.cc
// see also: https://www.tensorflow.org/versions/master/api_guides/cc/guide

#include <chrono>
#include <iomanip>
#include <sstream>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/util/events_writer.h"

#include "custom_ops.h"
#include "gl_wrappers.h"
#include "OfflineDataLoader.h"

void AddGraphInputsAndOutputOps(
    tensorflow::GraphDef* graph_def,
    const std::vector<fribr::Framebuffer>& input_framebuffers,
    const std::vector<std::string>& input_graph_node_names,
    const fribr::Framebuffer& output_framebuffer,
    const std::string& output_node_name, const GLFWwindow* window) {
  // TODO (True): potential bug in Protobuf and/or the way this project uses the
  // Protobuf library? Have to force the release of fields before setting them
  // (they otherwise point to the same memory; this doesn't happen using bazel,
  // though...).
/*
  // Squeeze dim 0
  {
    auto node = graph_def->add_node();
    node->release_name();
    node->release_op();
    node->set_name("output/squeeze");
    node->set_op("Squeeze");
    node->add_input(OUTPUT_LAYER);
    (*node->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
    (*node->mutable_attr())["squeeze_dims"].mutable_list()->add_i(0);
  }

  // Create a constant value of 0
  {
    auto node = graph_def->add_node();
    node->release_name();
    node->release_op();
    node->set_name("output/const_0");
    node->set_op("Const");
    (*node->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
    auto value = (*node->mutable_attr())["value"].mutable_tensor();
    value->set_dtype(tensorflow::DT_FLOAT);
    value->set_version_number(0);
    value->mutable_tensor_shape()->add_dim()->set_size(1);
    value->add_float_val(0.f);
  }

  // Create a constant value of 255
  {
    auto node = graph_def->add_node();
    node->release_name();
    node->release_op();
    node->set_name("output/const_255");
    node->set_op("Const");
    (*node->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
    auto value = (*node->mutable_attr())["value"].mutable_tensor();
    value->set_dtype(tensorflow::DT_FLOAT);
    value->set_version_number(0);
    value->mutable_tensor_shape()->add_dim()->set_size(1);
    value->add_float_val(255.f);
  }

  // Multiply by 255
  {
    auto node = graph_def->add_node();
    node->release_name();
    node->release_op();
    node->set_name("output/mul");
    node->set_op("Mul");
    node->add_input("output/squeeze");
    node->add_input("output/const_255");
    (*node->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
  }

  // Clip to [0, 255]
  {
    auto node = graph_def->add_node();
    node->release_name();
    node->release_op();
    node->set_name("output/clip_0");
    node->set_op("Maximum");
    node->add_input("output/mul");
    node->add_input("output/const_0");
    (*node->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
  }

  {
    auto node = graph_def->add_node();
    node->release_name();
    node->release_op();
    node->set_name("output/clip_255");
    node->set_op("Minimum");
    node->add_input("output/clip_0");
    node->add_input("output/const_255");
    (*node->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
  }

  // Cast to uint8
  {
    auto node = graph_def->add_node();
    node->release_name();
    node->release_op();
    node->set_name("output/cast");
    node->set_op("Cast");
    node->add_input("output/clip_255");
    (*node->mutable_attr())["SrcT"].set_type(tensorflow::DT_FLOAT);
    (*node->mutable_attr())["DstT"].set_type(tensorflow::DT_UINT8);
  }

  // Copy to frame buffer
  {
    auto node = graph_def->add_node();
    node->release_name();
    node->release_op();
    node->set_name("output");
    node->set_op("CopyToFramebuffer");
    node->add_input("output/cast");
    (*node->mutable_attr())["framebuffer_ptr"].set_i(
        reinterpret_cast<const int64_t>(output_framebuffer));
    (*node->mutable_attr())["GLFWwindow_ptr"].set_i(
        reinterpret_cast<const int64_t>(window));
  }
*/

/*
  {
    auto node = graph_def->add_node();
    node->release_name();
    node->release_op();
    node->set_name("texture_inputs");
    node->set_op("TextureInputs");
    (*node->mutable_attr())["GLFWwindow_ptr"].set_i(
        reinterpret_cast<const int64_t>(window));

    for (size_t i = 0; i < input_framebuffers.size(); ++i) {
      const auto input_shape = input_framebuffers[i].get_resolution();

      (*node->mutable_attr())["texture_id"].add_i(
          reinterpret_cast<const int64_t>(
              input_framebuffers[i]->get_textures()[0]->get_id()));
      auto shape = (*node->mutable_attr())["output_shapes"].add_shape();
      shape->add_dim()->set_size(1);
      shape->add_dim()->set_size(input_shape[0]);
      shape->add_dim()->set_size(input_shape[1]);
      shape->add_dim()->set_size(3);  // RGB
    }
  }

  // Create a constant value of 0
  {
    auto node = graph_def->add_node();
    node->release_name();
    node->release_op();
    node->set_name("const_0");
    node->set_op("Const");
    (*node->mutable_attr())["dtype"].set_type(tensorflow::DT_INT32);
    auto value = (*node->mutable_attr())["value"].mutable_tensor();
    value->set_dtype(tensorflow::DT_INT32);
    value->set_version_number(0);
    value->mutable_tensor_shape()->add_dim()->set_size(1);
    value->add_int_val(0);
  }

  // NOTE (True): the concat could potentially be folded into the input op
  {
    auto node = graph_def->add_node();
    node->release_name();
    node->release_op();
    node->set_name("concat");
    node->set_op("Concat");
    node->add_input("const_0");
    node->add_input("texture_inputs");
    (*node->mutable_attr())["N"].set_i(input_framebuffers.size());
    (*node->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
  }
*/

  {
    auto node = graph_def->add_node();
    node->release_name();
    node->release_op();
    node->set_name("output");
    node->set_op("CopyToTexture");
    node->add_input(output_node_name);
    (*node->mutable_attr())["GLFWwindow_ptr"].set_i(
        reinterpret_cast<const int64_t>(window));
    (*node->mutable_attr())["texture_id"].set_i(
        output_framebuffer.get_textures()[0]->get_id());
  }
}

// Reads a model graph definition from disk.
tensorflow::Status LoadGraph(
    std::unique_ptr<tensorflow::Session>* session,
    const std::string& graph_file_name,
    const std::vector<fribr::Framebuffer>& input_framebuffers,
    const std::vector<std::string>& input_graph_node_names,
    const fribr::Framebuffer& output_framebuffer,
    const std::string& output_node_name, const GLFWwindow* window) {
  tensorflow::GraphDef graph_def;
  tensorflow::Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }

  AddGraphInputsAndOutputOps(&graph_def, input_framebuffers,
                             input_graph_node_names, output_framebuffer,
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

//  fribr::Texture::Descriptor color_descriptor(GL_CLAMP_TO_EDGE, GL_NEAREST, 0,
//                                              fribr::TextureFormat::RGBA8);
//  std::vector<fribr::Framebuffer> input_framebuffers;
//  for (size_t i = 0; i < 5; ++i) {
//    input_framebuffers.emplace_back(
//        {width, height},
//        std::vector<fribr::Texture::Descriptor>(1, color_descriptor));
//  }
  std::vector<fribr::Framebuffer> input_framebuffers;

  const std::vector<std::string> input_graph_node_names = {
      "imageloader/convert_image_1", "imageloader/convert_image_2",
      "imageloader/convert_image_3", "imageloader/convert_image_4",
      "imageloader/convert_image_5"};

  const std::string output_node_name = "model/add_4";

  fribr::Texture::Descriptor color_descriptor(GL_CLAMP_TO_EDGE, GL_NEAREST, 0,
                                              fribr::TextureFormat::RGBA8);
  fribr::Framebuffer output_framebuffer(
      {width, height},
      std::vector<fribr::Texture::Descriptor>(1, color_descriptor));

  //
  // Load and initialize model from Protobuf file.
  //

  std::unique_ptr<tensorflow::Session> session;
  {
    const auto load_graph_status = LoadGraph(
        &session, graph_path, input_framebuffers, input_graph_node_names,
        output_framebuffer, output_node_name, window);
    if (!load_graph_status.ok()) {
      LOG(ERROR) << load_graph_status;
      return -1;
    }
  }

  // Create the data loader.
  OfflineDataLoader data_loader(data_folder, data_file);

  //
  // Run the model.
  size_t output_idx = 0;

  // Create the output folder where images are saved, if it doesn't exist.
  tensorflow::Env::Default()->RecursivelyCreateDir(output_folder);

  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
  while (true) {
    inputs = data_loader.Next();
    if (inputs.size() == 0) {
      break;
    }

    {
      auto start = std::chrono::high_resolution_clock::now();
      const auto run_status = session->Run(inputs, {}, {"output"}, {});
      if (!run_status.ok()) {
        LOG(ERROR) << run_status;
        return -1;
      }
      const std::chrono::duration<double, std::milli> duration =
          std::chrono::high_resolution_clock::now() - start;
      LOG(INFO) << "Network ran in " << duration.count() << " ms";
    }

    // Test saving
    glfwMakeContextCurrent(window);
    std::ostringstream filename_ss;
    filename_ss << std::setfill('0') << std::setw(5) << output_idx << ".png";
    const std::string filename =
        tensorflow::io::JoinPath(output_folder, filename_ss.str());

    cv::Mat image =
        output_framebuffer.read_texture(0, fribr::ReadbackMode::ReadBGR);
    cv::flip(image, image, 0);
    cv::imwrite(filename, image);
    glfwMakeContextCurrent(0);

    ++output_idx;
  }

  return 0;
}
