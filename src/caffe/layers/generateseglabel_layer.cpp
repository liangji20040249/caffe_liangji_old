/*
 * Author: Liangji
 * Email: liangji20040249@gmail.com
 */
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"
#include "caffe/layers/generateseglabel_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void GenerateSeglabelLayer<Dtype>::LayerSetUp( const vector<Blob<Dtype>*> & bottom,
							    const vector<Blob<Dtype>*> & top )
	{
	
		CHECK(bottom.size()==1)<<"only support one bottom here";
                num_ = bottom[0]->num();
                channels_ = bottom[0]->channels();
                height_ = bottom[0]->height();
                width_ = bottom[0]->width();

                CHECK(channels_ == 1)<<"label should be one channels";

		resize_ratio_ = this->layer_param_.generateseglabel_param().downsample_ratio();
                seg_ignore_range_ = this->layer_param_.generateseglabel_param().seg_ignore_range();
                reg_range_ = this->layer_param_.generateseglabel_param().reg_range();
                ignore_label_ = this->layer_param_.generateseglabel_param().ignore_label();
                seg_label_shift_ = this->layer_param_.generateseglabel_param().seg_label_shift();

                out_height_ = height_ * resize_ratio_;
                out_width_ = width_ * resize_ratio_;
                out_channels_ = this->layer_param_.generateseglabel_param().valid_label_count();



                CHECK(top.size()==5)<<"top size should be 5.";

                top[0]->Reshape(num_, out_channels_, out_height_, out_width_);//expand seg label
                top[1]->Reshape(num_, 1, out_height_, out_width_);//seg mask
                top[2]->Reshape(num_, 1, out_height_, out_width_);//reg label r
                top[3]->Reshape(num_, 1, out_height_, out_width_);//reg mask
                top[4]->Reshape(num_, 2, out_height_, out_width_);//reg label dx

	}


	template <typename Dtype>
	void GenerateSeglabelLayer<Dtype>::Reshape( const vector<Blob<Dtype>*> & bottom,
							 const vector<Blob<Dtype>*> & top )
	{

		CHECK(bottom.size()==1)<<"only support one bottom here";
		num_ = bottom[0]->num();
		channels_ = bottom[0]->channels();
		height_ = bottom[0]->height();
		width_ = bottom[0]->width();
		CHECK(channels_ == 1)<<"label should be one channels";

		resize_ratio_ = this->layer_param_.generateseglabel_param().downsample_ratio();
		seg_ignore_range_ = this->layer_param_.generateseglabel_param().seg_ignore_range();
		reg_range_ = this->layer_param_.generateseglabel_param().reg_range();
		ignore_label_ = this->layer_param_.generateseglabel_param().ignore_label();
		seg_label_shift_ = this->layer_param_.generateseglabel_param().seg_label_shift();

		out_height_ = height_ / resize_ratio_;
		out_width_ = width_ / resize_ratio_;
		out_channels_ = this->layer_param_.generateseglabel_param().valid_label_count();



		CHECK(top.size()==5)<<"top size should be 5.";

		top[0]->Reshape(num_, out_channels_, out_height_, out_width_);//expand seg label
		top[1]->Reshape(num_, 1, out_height_, out_width_);//seg mask
		top[2]->Reshape(num_, 1, out_height_, out_width_);//reg label r
		top[3]->Reshape(num_, 1, out_height_, out_width_);//reg mask
		top[4]->Reshape(num_, 2, out_height_, out_width_);//reg label dx

	}


	template <typename Dtype>
	void GenerateSeglabelLayer<Dtype>::Forward_cpu( const vector<Blob<Dtype>*> & bottom,
							     const vector<Blob<Dtype>*> & top )
	{
	LOG(FATAL)<<"do not implement cpu";
	//	this->Forward_gpu(bottom,top);

	}


	template <typename Dtype>
	void GenerateSeglabelLayer<Dtype>::Backward_cpu( const vector<Blob<Dtype>*> & top,
							      const vector<bool> & propagate_down,
							      const vector<Blob<Dtype>*> & bottom )
	{

		;
	}


#ifdef CPU_ONLY
	STUB_GPU( GenerateSeglabelLayer );
#endif

	INSTANTIATE_CLASS( GenerateSeglabelLayer );
	REGISTER_LAYER_CLASS( GenerateSeglabel );
}  /* namespace caffe */
