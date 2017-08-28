
#include <string>
#include <tr1/unordered_map>
#include "caffe/caffe.hpp"
#include "caffe/util/upgrade_proto.hpp"
using namespace caffe;
typedef std::tr1::unordered_map<std::string, std::string> umap;
void extract_proto(NetParameter &src_param, NetParameter &dst_param, bool for_inference = false) {
  umap c_map;
  for (int i = 0; i < src_param.layer_size(); ++i) {
    LayerParameter *layer = src_param.mutable_layer(i);
    if (layer->type() == "Split") {
      for (int j = 0; j < layer->top_size(); ++j)
        c_map[layer->top(j)] = layer->bottom(0);
    } else {
      layer->clear_blobs();
      layer->clear_phase();
      if (for_inference) {
        layer->clear_param();
        layer->clear_propagate_down();
        layer->clear_include();
        layer->clear_exclude();
        if (layer->has_convolution_param()) {
          layer->mutable_convolution_param()->clear_weight_filler();
          layer->mutable_convolution_param()->clear_bias_filler();
        }
        /*if (layer->has_batch_norm_param()) {
          layer->mutable_batch_norm_param()->clear_scale_filler();
          layer->mutable_batch_norm_param()->clear_bias_filler();
          layer->mutable_batch_norm_param()->clear_momentum();
          layer->mutable_batch_norm_param()->set_frozen(true);
        }*/
      }
      for (int j = 0; j < layer->bottom_size(); ++j)
        if (c_map.find(layer->bottom(j)) != c_map.end())
          layer->set_bottom(j, c_map[layer->bottom(j)]);
      dst_param.add_layer()->CopyFrom(*layer);
    }
  }
}
int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    LOG(ERROR) << "Usage: "
               << "extract_proto trained.caffemodel out.prototxt [--out_inference_proto]";
    return 0;
  }
  NetParameter src_param;
  ReadNetParamsFromBinaryFileOrDie(argv[1], &src_param);
  NetParameter dst_param;
  if (argc == 3)
    extract_proto(src_param, dst_param);

  else
    extract_proto(src_param, dst_param, true);
  WriteProtoToTextFile(dst_param, argv[2]);
  LOG(INFO) << "Wrote NetParameter text proto to " << argv[2];
  return 0;



 }
