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
void DatadimLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ConcatParameter& concat_param = this->layer_param_.concat_param();
  CHECK(!(concat_param.has_axis() && concat_param.has_concat_dim()))
      << "Either axis or concat_dim should be specified; not both.";
}

template <typename Dtype>
void DatadimLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
 
  int n = bottom[0]->num();
  top[0]->Reshape(n,1,1,2);

}

template <typename Dtype>
void DatadimLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
 
   
  for(int i=0;i<bottom[0]->num();i++)
  {
     top_data[i*2]=bottom[0]->height();
     top_data[i*2+1]=bottom[0]->width();
  }
}

template <typename Dtype>
void DatadimLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  ;
}

#ifdef CPU_ONLY
STUB_GPU(DatadimLayer);
#endif

INSTANTIATE_CLASS(DatadimLayer);
REGISTER_LAYER_CLASS(Datadim);

}  // namespace caffe
