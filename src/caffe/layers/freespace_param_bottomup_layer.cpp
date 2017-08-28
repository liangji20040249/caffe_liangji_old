#include <vector>

#include "caffe/layers/freespace_param_bottomup_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FreespaceParamBottomUpLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
        
 ;
}

template <typename Dtype>
void FreespaceParamBottomUpLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

        CHECK(top.size()==1);
        CHECK(bottom.size()==1);
        CHECK(bottom[0]->channels()==1);

	int xstage,ystage;
        switch (this->layer_param_.freespace_param_bottomup_param().type())
        {
                  case FreespaceParamBottomUpParameter_Type_ONEDIM:
			//LOG(INFO)<<"1 dim";
                        xstage = this->layer_param_.freespace_param_bottomup_param().xstage();
                        top[0]->Reshape(bottom[0]->num(), 1, 1,xstage);
                        break;
                  case FreespaceParamBottomUpParameter_Type_TWODIM:
			//LOG(INFO)<<"2 dim";
                        xstage = this->layer_param_.freespace_param_bottomup_param().xstage();
                        ystage = this->layer_param_.freespace_param_bottomup_param().ystage();
                        top[0]->Reshape(bottom[0]->num(), 1, ystage,xstage);
                        break;
		  default:
			LOG(FATAL)<<"undefine type";
			break;

        }
}

template <typename Dtype>
void FreespaceParamBottomUpLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) { 
        this->Forward_gpu(bottom,top);
  
}

template <typename Dtype>
void FreespaceParamBottomUpLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        CHECK(false);
 ;
}

#ifdef CPU_ONLY
STUB_GPU(FreespaceParamBottomUpLayer);
#endif

INSTANTIATE_CLASS(FreespaceParamBottomUpLayer);
REGISTER_LAYER_CLASS(FreespaceParamBottomUp);

}  // namespace caffe
