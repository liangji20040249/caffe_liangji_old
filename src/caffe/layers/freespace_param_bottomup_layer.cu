#include <vector>

#include "caffe/layers/freespace_param_bottomup_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void FindBottomUpEdge(const int count,int num,int channels,int height,int width,int xstage,const Dtype * bottomdata, Dtype * topdata ) {
  CUDA_KERNEL_LOOP(index, count) {

    int v = index;

    int n = v / xstage;
    int c = 0;
    int xstep = width / (xstage-1);
    int current_stage = index % xstage;

    int w = current_stage * xstep;

    if(current_stage == xstage -1)
    {
      w = width -1;
    }

    int samplecount = channels * height * width;
    int spcount = height * width;

    int h=0;
    bool findedge = false;
    Dtype value = height-1;
    for(h=height-1;h>0;h-=1)
    {


      if(bottomdata[n*samplecount + c * spcount + h * width + w] ==1 &&  bottomdata[n*samplecount + c * spcount + (h-1) * width + w]==0)
      {
        findedge = true;
        value = h;
        break;
      }
    }
    topdata[n * xstage + current_stage] = height -1 - value;
  }
}


template <typename Dtype>
__global__ void FindBottomUpEdge_TowDim(const int count,int num,int channels,int height,int width,int xstage,int ystage,const Dtype * bottomdata, Dtype * topdata ) {
  CUDA_KERNEL_LOOP(index, count) {

    int v = index;

    int n = v / xstage;
    int c = 0;
    int xstep = width / (xstage-1);
    int current_stage = index % xstage;

    int w = current_stage * xstep;

    if(current_stage == xstage -1)
    {
      w = width -1;
    }

    int samplecount = channels * height * width;
    int spcount = height * width;

    int h=0;
    bool findedge = false;
    Dtype value = height-1;
    for(h=height-1;h>0;h-=1)
    {


      if(bottomdata[n*samplecount + c * spcount + h * width + w] ==1 &&  bottomdata[n*samplecount + c * spcount + (h-1) * width + w]==0)
      {
        findedge = true;
        value = h;
        break;
      }
    }

    int current_stage_x = current_stage;

    int k=0;
    int src_pos_y = 0;
    
    for(k=0;k<ystage;k++)
    {
      src_pos_y = float(k) * float(height) / float(ystage -1);;
      if(k == ystage -1)
      {
        src_pos_y = height -1;
      }
    
      topdata[n*1*xstage*ystage + k * xstage + current_stage_x] = value - src_pos_y;

    }

  }
}

template <typename Dtype>
void FreespaceParamBottomUpLayer<Dtype>::Forward_gpu_onedim(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

        int num = bottom[0]->num();
        int channels = bottom[0]->channels();
        int height = bottom[0]->height();
        int width = bottom[0]->width();
        int xstage = this->layer_param_.freespace_param_bottomup_param().xstage();
        const Dtype * bottomdata = bottom[0]->gpu_data();
        Dtype * topdata = top[0]->mutable_gpu_data();

        const int count = num * xstage;



        FindBottomUpEdge<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, num,channels,height,width,xstage,bottomdata,topdata);
        CUDA_POST_KERNEL_CHECK;

}

template <typename Dtype>
void FreespaceParamBottomUpLayer<Dtype>::Forward_gpu_twodim(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

        int num = bottom[0]->num();
        int channels = bottom[0]->channels();
        int height = bottom[0]->height();
        int width = bottom[0]->width();
        int xstage = this->layer_param_.freespace_param_bottomup_param().xstage();
	int ystage = this->layer_param_.freespace_param_bottomup_param().ystage();
        const Dtype * bottomdata = bottom[0]->gpu_data();
        Dtype * topdata = top[0]->mutable_gpu_data();

        const int count = num * xstage;



        FindBottomUpEdge_TowDim<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, num,channels,height,width,xstage,ystage,bottomdata,topdata);
        CUDA_POST_KERNEL_CHECK;

}

template <typename Dtype>
void FreespaceParamBottomUpLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

        switch (this->layer_param_.freespace_param_bottomup_param().type())
        {
                  case FreespaceParamBottomUpParameter_Type_ONEDIM:
                        this->Forward_gpu_onedim(bottom,top);
                        break;
                  case FreespaceParamBottomUpParameter_Type_TWODIM:
                        this->Forward_gpu_twodim(bottom,top);
                        break;
                  default:
                        LOG(FATAL)<<"unknown type";
                        break;

        }

}

template <typename Dtype>
void FreespaceParamBottomUpLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        this->Backward_gpu(top,propagate_down,bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(FreespaceParamBottomUpLayer);

}  // namespace caffe
