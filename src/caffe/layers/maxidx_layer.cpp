#include <vector>

#include "caffe/layers/maxidx_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MaxIdxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	;
}

template <typename Dtype>
void MaxIdxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	CHECK(bottom.size()==1)<<"bottom size must == 1, here get"<<bottom.size();
	CHECK(top.size()==1)<<"top size should be 1, here we get "<<top.size();

	idx_num_ = this->layer_param_.maxidx_param().idx_num();


	num_ = bottom[0]->num();
	channels_ = bottom[0]->channels();
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();

	CHECK(channels_ % idx_num_ == 0);
	top[0]->Reshape(num_, idx_num_, height_, width_);


/*
	CHECK(bottom.size()>1)<<"bottom size must > 1, here get"<<bottom.size();
	CHECK(top.size()==1)<<"top size should be 1, here we get "<<top.size();

	for(int i=1;i<bottom.size();i++)
	{
		CHECK(bottom[i]->num() == bottom[0]->num())<<"bottom num should be same, here we get "<<bottom[i]->num()<<" vs "<<bottom[0]->num();
		CHECK(bottom[i]->channels() == bottom[0]->channels())<<"bottom channels should be same, here we get "<<bottom[i]->channels()<<" vs "<<bottom[0]->channels();
		CHECK(bottom[i]->height() == bottom[0]->height())<<"bottom height should be same, here we get "<<bottom[i]->height()<<" vs "<<bottom[0]->height();
		CHECK(bottom[i]->width() == bottom[0]->width())<<"bottom width should be same, here we get "<<bottom[i]->width()<<" vs "<<bottom[0]->width();
	}

	top[0]->Reshape(num_, channels_, height_, width_);
	num_ = bottom[0]->num();
	channels_ = bottom[0]->channels();
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();
*/
}

template <typename Dtype>
void MaxIdxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
 this->Forward_gpu(bottom,top);
}

template <typename Dtype>
void MaxIdxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  ;
}

#ifdef CPU_ONLY
STUB_GPU(MaxIdxLayer);
#endif

INSTANTIATE_CLASS(MaxIdxLayer);
REGISTER_LAYER_CLASS(MaxIdx);

}  // namespace caffe
