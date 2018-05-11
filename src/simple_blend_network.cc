// partly adapted from:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/main.cc
// see also: https://www.tensorflow.org/versions/master/api_guides/cc/guide

#include <chrono>
#include <iomanip>
#include <sstream>
#include <utility>
#include <vector>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/util/events_writer.h"

#include "OfflineDataLoader.h"

const std::string INPUT_LAYER = "concat";
const std::string OUTPUT_LAYER = "model/add_4";

// Creates a new session with an associated graph.
tensorflow::Status CreateSessionWithGraph(
    const tensorflow::GraphDef& graph_def,
    std::unique_ptr<tensorflow::Session>* session) {
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  return (*session)->Create(graph_def);
}


// Reads a model graph definition from disk.
tensorflow::Status LoadGraph(const std::string& graph_file_name,
                             std::unique_ptr<tensorflow::Session>* session,
                             tensorflow::EventsWriter* logger) {
  tensorflow::GraphDef graph_def;
  tensorflow::Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }

  auto graph_def_str = new std::string;
  graph_def.SerializeToString(graph_def_str);
  tensorflow::Event event;
  event.set_allocated_graph_def(graph_def_str);
  logger->WriteEvent(event);

  return CreateSessionWithGraph(graph_def, session);
}

// Creates a separate graph/session for saving images.
tensorflow::Status CreateSaveImageGraph(
    std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  auto input_image = tensorflow::ops::Placeholder(scope.WithOpName("image"),
                                                  tensorflow::DT_FLOAT);
  auto input_filename = tensorflow::ops::Placeholder(
      scope.WithOpName("filename"), tensorflow::DT_STRING);

  auto squeeze = tensorflow::ops::Squeeze(scope, input_image,
                                          tensorflow::ops::Squeeze::Axis({0}));
  auto image_255 = tensorflow::ops::Multiply(scope, squeeze, 255.f);
  auto image_clip = tensorflow::ops::ClipByValue(scope, image_255, 0.f, 255.f);
  auto image_uint8 =
      tensorflow::ops::Cast(scope, image_clip, tensorflow::DT_UINT8);
  auto image_png = tensorflow::ops::EncodePng(scope, image_uint8);

  auto writer = tensorflow::ops::WriteFile(scope.WithOpName("save"),
                                           input_filename, image_png);

  tensorflow::GraphDef graph_def;
  const auto graph_def_status = scope.ToGraphDef(&graph_def);
  if (!graph_def_status.ok()) {
    return graph_def_status;
  }

  return CreateSessionWithGraph(graph_def, session);
}

int main(int argc, char** argv) {
  // TODO (True): create command-line args
  const std::string data_folder =
      "/playpen/jtprice/research/image_based_rendering/code/"
      "simple_blend_network";
  const std::string data_file =
      "/playpen/jtprice/research/image_based_rendering/code/"
      "simple_blend_network/test.txt";
  const std::string output_folder =
      "/playpen/jtprice/research/image_based_rendering/output";
  const std::string graph_path =
      "/playpen/jtprice/research/image_based_rendering/code/"
      "simple_blend_network/model/model-40192.pb";
  const std::string log_folder =
      "/playpen/jtprice/research/image_based_rendering/logs";

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

  // Logging for debug purposes.
  tensorflow::Env::Default()->RecursivelyCreateDir(log_folder);
  tensorflow::EventsWriter logger(
      tensorflow::io::JoinPath(log_folder, "events"));

  // First we load and initialize the model from the provided protobuf file.
  std::unique_ptr<tensorflow::Session> session;
  {
    const auto load_graph_status = LoadGraph(graph_path, &session, &logger);
    if (!load_graph_status.ok()) {
      LOG(ERROR) << load_graph_status;
      return -1;
    }
  }

  // Create a session for saving images.
  std::unique_ptr<tensorflow::Session> save_session;
  tensorflow::Tensor filename(tensorflow::DT_STRING, tensorflow::TensorShape());
  {
    const auto load_graph_status = CreateSaveImageGraph(&save_session);
    if (!load_graph_status.ok()) {
      LOG(ERROR) << load_graph_status;
      return -1;
    }
  }

  // Create the output folder, if it doesn't exist.
  tensorflow::Env::Default()->RecursivelyCreateDir(output_folder);

  // Create the data loader.
  OfflineDataLoader data_loader(data_folder, data_file);

  // Run the model.
  size_t output_idx = 0;

  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
  while (true) {
    inputs = data_loader.Next();
    if (inputs.size() == 0) {
      break;
    }

    std::vector<tensorflow::Tensor> outputs;
    {
      auto start = std::chrono::high_resolution_clock::now();
      const auto run_status =
          session->Run(inputs, {OUTPUT_LAYER}, {}, &outputs);
      if (!run_status.ok()) {
        LOG(ERROR) << run_status;
        return -1;
      }
      const std::chrono::duration<double, std::milli> duration =
          std::chrono::high_resolution_clock::now() - start;
      LOG(INFO) << "Network ran in " << duration.count() << " ms";
    }

    /*
    std::vector<tensorflow::Tensor> outputs;
    {
      tensorflow::RunOptions run_options;
      tensorflow::RunMetadata run_metadata;
      run_options.set_trace_level(tensorflow::RunOptions::FULL_TRACE);
      const auto run_status = session->Run(run_options, inputs, {OUTPUT_LAYER},
                                           {}, &outputs, &run_metadata);
      if (!run_status.ok()) {
        LOG(ERROR) << run_status;
        return -1;
      }
      auto tagged_metadata = new tensorflow::TaggedRunMetadata;
      tagged_metadata->set_tag(std::to_string(output_idx));
      std::string run_metadata_str;
      run_metadata.SerializeToString(&run_metadata_str);
      tagged_metadata->set_run_metadata(run_metadata_str);
      tensorflow::Event event;
      event.set_allocated_tagged_run_metadata(tagged_metadata);
      event.set_wall_time(
          std::chrono::duration<double>(
              std::chrono::high_resolution_clock::now().time_since_epoch())
              .count());
      logger.WriteEvent(event);
    }
    */

    std::ostringstream filename_ss;
    filename_ss << std::setfill('0') << std::setw(5) << output_idx << ".png";
    filename.scalar<std::string>()() =
        tensorflow::io::JoinPath(output_folder, filename_ss.str());
    {
      const auto run_status =
          save_session->Run({{"image", outputs[0]}, {"filename", filename}}, {},
                            {"save"}, nullptr);
      if (!run_status.ok()) {
        LOG(ERROR) << run_status;
        return -1;
      }
    }

    ++output_idx;
  }

  return 0;
}
