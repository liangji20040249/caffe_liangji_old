/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/datadim_layer.hpp"

namespace caffe {



template <typename Dtype>
void DatadimLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      this->Forward_cpu(bottom,top);
}

template <typename Dtype>
void DatadimLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      ;
}

INSTANTIATE_LAYER_GPU_FUNCS(DatadimLayer);

}  // namespace caffe
