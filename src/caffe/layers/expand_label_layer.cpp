/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/expand_label_layer.hpp"

namespace caffe {

template <typename Dtype>
void ExpandLabelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    ignore_label_ = this->layer_param_.expand_label_param().ignore_label();
    maxlabel_ = this->layer_param_.expand_label_param().max_label();
    

}

template <typename Dtype>
void ExpandLabelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      
        int num = bottom[0]->num();
        int channels = bottom[0]->channels();
        int height = bottom[0]->height();
        int width = bottom[0]->width();
        CHECK(channels==1);
        CHECK(top.size()==1 || top.size()==2);

        switch (this->layer_param_.expand_label_param().type())
        {
                  case ExpandLabelParameter_Type_EXPAND:
                        top[0]->Reshape(num, maxlabel_+1,height, width);
                        break;
                  case ExpandLabelParameter_Type_SELECT_ONE:
                        top[0]->Reshape(num, 2,height, width);
                        break;
                  default:
                        LOG(FATAL) << "Unknown type method.";

        }

        if(top.size()>1)
        {
            top[1]->Reshape(num, 1,height, width);
        }
}

template <typename Dtype>
void ExpandLabelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      this->Forward_gpu(bottom,top);
        
}

template <typename Dtype>
void ExpandLabelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

      this->Backward_gpu(top,propagate_down,bottom);
        
}

#ifdef CPU_ONLY
STUB_GPU(ExpandLabelLayer);
#endif

INSTANTIATE_CLASS(ExpandLabelLayer);
REGISTER_LAYER_CLASS(ExpandLabel);

}  // namespace caffe
