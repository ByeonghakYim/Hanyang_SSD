from __future__ import print_function
import caffe
from caffe.model_libs import *
from google.protobuf import text_format
import datetime
import math
import os
import shutil
import stat
import subprocess
import sys

# 추가 네트워크 구성
def AddExtraLayers(net, use_batchnorm=True, lr_mult=1):
    use_relu = True

    # Add additional convolutional layers.
    # 19 x 19
    from_layer = net.keys()[-1]

    # TODO(weiliu89): Construct the name using the last layer to avoid duplication.
    # 10 x 10
    out_layer = "conv6_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1,
        lr_mult=lr_mult)
    # parameter : 현재 네트워크, 입력 레이어 이름, 출력 레이어(현재 레이어) 이름, Batch normalization 사용 여부
    #             Relu 사용 여부, 필터 개수, 필터 사이즈, padding 크기, stride 크기, base learning rate scaling

    from_layer = out_layer
    out_layer = "conv6_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 2,
        lr_mult=lr_mult)

    # 5 x 5
    from_layer = out_layer
    out_layer = "conv7_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv7_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 2,
      lr_mult=lr_mult)

    # 3 x 3
    from_layer = out_layer
    out_layer = "conv8_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv8_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 0, 1,
      lr_mult=lr_mult)

    return net


# 터미널 상의 현재 디렉토리를 caffe 루트 디렉토리로 설정(상대 경로에 사용됨)
caffe_root = os.getcwd()

# 본 파이썬 코드 실행 후 sh파일을 바로 실행하는지(True), 생성 후 터미널에서 직접 입력하여 실행하는지(False) 여부 설정
run_soon = True

# 이전에 저장되었던 snapshot 부터 재개 여부 설정
resume_training = True

# caffemodel 경로에 존재하는 기존 모델 삭제 여부 설정
remove_old_models = False

# Training LMDB 경로
train_data = "examples/ALL/ALL_train_lmdb"
# Validation LMDB 경로
test_data = "examples/ALL/ALL_val_lmdb"
# resize 높이, 너비 설정
resize_width = 600
resize_height = 150
resize = "{}x{}".format(resize_width, resize_height)

# batch sampling 파라미터
# 영상 내 어떤 하나의 Ground Truth와 IOU가 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0인 경우의 최대 7개 후보를 생성하고, 그 중 하나를 랜덤으로 batch로 사용함
batch_sampler = [
        # IOU >= 0.0
        {
                'sampler': {
                        },
                'max_trials': 1,# 최대 시도 횟수 (최대 시도 횟수동안 조건을 만족하는 batch를 찾지 못할 경우 후보를 생성하지 않음)
                'max_sample': 1,# 최대 샘플링 갯수 (max_trials 이전에 조건을 만족하는 후보가 max_samples만큼 될 경우 더이상 시도하지 않음)
                'merged_data': True,
        },
        # IOU >= 0.1
        {
                'sampler': {
                        'min_scale': 0.3,# 원본 대비 최소 Sampling 크기
                        'max_scale': 1.0,# 원본 대비 최대 Sampling 크기
                        'min_aspect_ratio': 1.0,# 최소 Aspect Ratio(세로/가로)
                        'max_aspect_ratio': 1.0,# 최대 Aspect Ratio
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.1,# IOU 최소 Threshold (영상 내 어떤 한 Ground Truth와 IOU가 0.1 이상일 경우 후보로 선택 됨)
                        },
                'max_trials': 50,
                'max_sample': 1,
                'merged_data': True,
        },
        # IOU >= 0.3
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 1.0,
                        'max_aspect_ratio': 1.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.3,
                        },
                'max_trials': 50,
                'max_sample': 1,
                'merged_data': True,
        },
        # IOU >= 0.5
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.5,
                        },
                'max_trials': 50,
                'max_sample': 1,
                'merged_data': True,
        },
        # IOU >= 0.7
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.7,
                        },
                'max_trials': 50,
                'max_sample': 1,
                'merged_data': True,
        },
        # IOU >= 0.9
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.9,
                        },
                'max_trials': 50,
                'max_sample': 1,
                'merged_data': True,
        },
        # IOU >= 1.0
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'max_jaccard_overlap': 1.0,
                        },
                'max_trials': 50,
                'max_sample': 1,
                'merged_data': True,
        },
        ]
train_transform_param = {
        'mirror': True, # 입력 영상 좌우 반전 여부 설정
        'mean_value': [96, 99, 94], # Zero mean centered를 위한 Subtraction에 사용할 Mean값 [B G R]
        'resize_param': {
                'prob': 1, # resize가 일어날 확률
                'resize_mode': P.Resize.WARP, # Resize 방법
                # WARP : width, height 모두 resize 크기로 warping
                # FIT_SMALL_SIZE : aspect ratio를 유지하고 resize시 다른 축이 resize 크기보다 작아지지 않는 축을 기준으로 resize 함,
                #                  원본 영상의 aspect_ratio를 유지하기 위해서는 min/max_aspect_ratio를 1.0으로 고정해야 함,
                #                  이부분에 'height_scale : resize_height', 'width_scale' : resize_width를 추가해야 함,
                #                  batch_size는 1이어야 함(batch들의 변환된 입력 사이즈가 다를 경우 병렬 처리 불가능)
				#		ex1) 리사이즈 크기는 1242x375이고, 1280x720 영상이 입력으로 들어올 경우 1242x699의 입력 영상으로 변환됨
                #       ex2) 리사이즈 크기는 1242x750이고, 1280x720 영상이 입력으로 들어올 경우 1333x750의 입력 영상으로 변환됨
                'height': resize_height, # Resize 높이
                'width': resize_width, # Resize 너비
                'interp_mode': [ # Interpolation 방법 랜덤으로 선택됨
                        P.Resize.LINEAR,
                        P.Resize.AREA,
                        P.Resize.NEAREST,
                        P.Resize.CUBIC,
                        P.Resize.LANCZOS4,
                        ],
                },
        'distort_param': {
                'brightness_prob': 0.5,# brightness 변환이 일어날 확률
                'brightness_delta': 32,# brightness 변환 범위 -32~32
                'contrast_prob': 0.5,# contrast 변환이 일어날 확률
                'contrast_lower': 0.5,# contrast 변환 배수 최소 범위
                'contrast_upper': 1.5,# contrast 변환 배수 최대 범위
                'hue_prob': 0.5,# hue 변환이 일어날 확률
                'hue_delta': 18,# hue 변환 범위 -18~18
                'saturation_prob': 0.5,# saturation 변환이 일어날 확률
                'saturation_lower': 0.5,# saturation 변환 배수 최소 범위
                'saturation_upper': 1.5,# saturation 변환 배수 최대 범위
                'random_order_prob': 0.0,# 채널 shuffling이 일어날 확률
                },
        'expand_param': {
                'prob': 0.5,# expand 가 일어날 확률
                'max_expand_ratio': 4.0,# Expansion Canvas의 원본 대비 최대 크기 비율
                },
        'emit_constraint': {
            'emit_type': caffe_pb2.EmitConstraint.CENTER,
            }
        }
test_transform_param = {
        'mean_value': [96, 99, 94],
        'resize_param': {
                'prob': 1,
                'resize_mode': P.Resize.WARP,
                'height': resize_height,
                'width': resize_width,
                'interp_mode': [P.Resize.LINEAR],
                },
        }

# batch normalization 사용 여부(Base Network에는 적용되지 않음)
use_batchnorm = True
# 추가 네트워크의 learning rate 스케일링 factor
lr_mult = 1
# base learning rate 값 설정
if use_batchnorm:
    base_lr = 0.0004
else:
    base_lr = 0.00004

# 아래의 문자열 변수들은 생성 파일 저장 경로 설정을 위한 과정으로 본인이 파일을 찾을 때 구별하기 쉽도록 지정하면 됨
# Modify the job name if you want.
job_name = "SSD_V2{}".format(resize)
# The name of the model. Modify it if you want.
model_name = "KITTI_{}".format(job_name)

# Directory which stores the model .prototxt file.
save_dir = "models/SSD_v2/KITTI/inha/{}".format(job_name)#save_dir = "models/VGGNet/KITTI/{}".format(job_name)
# Directory which stores the snapshot of models.
snapshot_dir = "models/SSD_v2/KITTI/inha/{}".format(job_name)#snapshot_dir = "models/VGGNet/KITTI/{}".format(job_name)
# Directory which stores the job script and log file.
job_dir = "jobs/SSD_v2/KITTI/inha/{}".format(job_name)#job_dir = "jobs/VGGNet/KITTI/{}".format(job_name)
# Directory which stores the detection results.
output_result_dir = "{}/data/KITTIdevkit/results/kitti/{}/Main".format(os.environ['HOME'], job_name)#output_result_dir = "{}/data/KITTIdevkit/results/kitti/{}/Main".format(os.environ['HOME'], job_name)

# model definition files.
train_net_file = "{}/train.prototxt".format(save_dir)
test_net_file = "{}/test.prototxt".format(save_dir)
deploy_net_file = "{}/deploy.prototxt".format(save_dir)
solver_file = "{}/solver.prototxt".format(save_dir)
# snapshot prefix.
snapshot_prefix = "{}/{}".format(snapshot_dir, model_name)
# job script path.
job_file = "{}/{}.sh".format(job_dir, model_name)
##############################################################################

# validation 이미지 경로와 크기가 정리된 txt 파일 경로
name_size_file = "data/KITTI/val_name_size.txt" 
# pretrain model 경로
pretrain_model = "models/ImageNet_inception_v2_8_preActRes_iter_100000.caffemodel" #[LYW]
# 데이터 label map 경로
label_map_file = "data/KITTI/labelmap_kitti.prototxt"

# MultiBoxLoss parameters.
# 클래스 개수
num_classes = 10
# Feature map의 한 픽셀에서 모든 Class가 똑같은 bounding box 이용 여부 설정
# 즉 클래스의 개수가 10개이고, 한 픽셀에서 prior box의 aspect ratio의 개수가 4개일 경우
# True 이면 한 픽셀에서 4개의 bbox 좌표 값이 출력되고, False이면 4 x num_classes개의 값이 출력됨
share_location = True
# Background label을 mapping 설정
background_label_id=0
# difficult 데이터를 사용할지 여부 annotation 정보에 difficulty 정보가 있을 경우 사용됨
train_on_diff_gt = True
# matching 된 box의 개수만 loss normalization에 사용
normalization_mode = P.Loss.VALID
# bbox coordinate 의 저장 방법(center_size는 center의 x, y 그리고 width, height 사용)
code_type = P.PriorBox.CENTER_SIZE
# 영상 영역을 넘어가는 bbox를 matching에 사용 여부 설정, False일 경우 사용
ignore_cross_boundary_bbox = False
# negative mining 방법 설정
# MAX_NEGATIVE : neg_pos_ratio를 고려한 상위 confidence를 갖는 negative만 사용
# ex) neg_pos_ratio = 3, positive가 3개일 경우 prediction의 confidence loss 상위 9개를 negative sets로 사용
# HARD_EXAMPLE : 최대 sample_size(mutlibox_loss_param에 추가 해야 함, default 64)만큼의 negative sets을 사용
# ex) sample_size = 64, matching bbox의 background를 negative로 사용하며 최대 confidence loss 상위 64개를 negative sets을 사용
# NONE : negative mining을 하지 않음
mining_type = P.MultiBoxLoss.MAX_NEGATIVE
neg_pos_ratio = 3.
loc_weight = (neg_pos_ratio + 1.) / 4.
multibox_loss_param = {
    'loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,# localization loss 계산 방법
    'conf_loss_type': P.MultiBoxLoss.SOFTMAX,# confidence loss 계산 방법
    'loc_weight': loc_weight,
    'num_classes': num_classes,
    'share_location': share_location,
    # PER_PREDICTION : prediction bbox 당 overlap_threshold를 고려한 하나의 ground truth 매칭
    # BIPARTITE : ground truth 당 하나의 prediction bbox 매칭
    'match_type': P.MultiBoxLoss.PER_PREDICTION,
    'overlap_threshold': 0.5,# positive로 사용할 최소 overlap Threshold
    'use_prior_for_matching': True,# box matching에 prior box를 그대로 사용할지 prediction된 bbox를 사용할지 여부 설정
    'background_label_id': background_label_id,
    'use_difficult_gt': train_on_diff_gt,
    'mining_type': mining_type,
    'neg_pos_ratio': neg_pos_ratio,
    'neg_overlap': 0.5,# negative로 사용할 최대 overlap threshold
    'code_type': code_type,
    'ignore_cross_boundary_bbox': ignore_cross_boundary_bbox,
    }
loss_param = {
    'normalization': normalization_mode,
    }

# default box 사이즈 설정에 사용될 단축 크기
min_dim = 300
# conv4_3 ==> 38 x 38
# fc7 ==> 19 x 19
# conv6_2 ==> 10 x 10
# conv7_2 ==> 5 x 5
# conv8_2 ==> 3 x 3
# conv9_2 ==> 1 x 1
# 멀티박스에서 사용될 레이어 설정
mbox_source_layers = ['bn4a_root_relu', 'bn_res4b_relu', 'conv6_2', 'conv7_2', 'conv8_2']
# 각 레이어 별 최대 최소 비율 설정
min_ratio = 20
max_ratio = 90
step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2)))
#### min_sizes, max_sizes는 아래의 수식을 사용하지 않고, 각 레이어별 list의 위치에 맞게 임의로 입력이 가능함
#### min_sizes와 max_sizes는 크기가 같아야 함
min_sizes = []
max_sizes = []
for ratio in xrange(min_ratio, max_ratio + 1, step):
  min_sizes.append(min_dim * ratio / 100.)
  max_sizes.append(min_dim * (ratio + step) / 100.)
min_sizes = [min_dim * 10 / 100.] + min_sizes
max_sizes = [min_dim * 20 / 100.] + max_sizes
# 각 레이어의 크기에 따른 리사이즈된 입력 대비 default 박스의 중심 위치 interval 설정
steps = [8, 16, 32, 64, 100]
# 각 레이어 별 default box의 aspect ratio 설정
aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2]]
# 레이어 별 normalization 값 설정, -1일 경우 normalization 하지 않음
normalizations = [-1, -1, -1, -1, -1]
# variance used to encode/decode prior bboxes.
if code_type == P.PriorBox.CENTER_SIZE:
  prior_variance = [0.1, 0.1, 0.2, 0.2]
else:
  prior_variance = [0.1]
# aspect ratio이 역수 사용 여부 설정
# True 시 aspect ratio가 [2, 3]일 경우 2x1, 3x1, 1x2, 1x3 사용
# False 시                             2x1, 3x1 사용
flip = True
# 입력 영상 경계를 넘어가는 값을 입력 영상 내로 제한하는지 여부 설정
# ex) True 시 300x300 입력 영상에서 좌상단 우하단 좌표가 다음과 같을 경우
#     (-10, 0), (10, 10) --> (0, 0), (100, 100) 으로 제한
clip = False
#### ex) min_sizes = [[30 50], .....]
####     max_sizes = [[100, 150], .....]
####     aspect_ratios = [[2], .....]
####     flip이 true 일 경우 
####     일때, 첫번째 레이어의 한 픽셀은
####           한 축이 30픽셀인 1x1, 
####           한 축이 50픽셀인 1x1,
####           한 축이 sqrt(30*100)인 1x1,
####           한 축이 sqrt(50*150)인 1x1,
####           가로가 30*sqrt(2), 세로가 30/sqrt(2)인 2x1
####           가로가 50*sqrt(2), 세로가 50/sqrt(2)인 2x1
####           가로가 30/sqrt(2), 세로가 30*sqrt(2)인 1x2
####           가로가 50/sqrt(2), 세로가 50*sqrt(2)인 1x2
####     의 default box를 가짐

# 사용 GPU 설정
gpus = "0"
gpulist = gpus.split(",")
num_gpus = len(gpulist)

# batch 사이즈 설정, 실질적인 업데이트는 accum_batch_size에 도달할 경우 업데이트함
# ex) batch_size = 16, accum_batch_size = 32일 경우 
#     32/16 = 2번 forward를 통해 얻어진 loss를 이용하여 업데이트
batch_size =32
accum_batch_size = 32
iter_size = accum_batch_size / batch_size
solver_mode = P.Solver.CPU
device_id = 0
batch_size_per_device = batch_size
if num_gpus > 0:
  batch_size_per_device = int(math.ceil(float(batch_size) / num_gpus))
  iter_size = int(math.ceil(float(accum_batch_size) / (batch_size_per_device * num_gpus)))
  solver_mode = P.Solver.GPU
  device_id = int(gpulist[0])

if normalization_mode == P.Loss.NONE:
  base_lr /= batch_size_per_device
elif normalization_mode == P.Loss.VALID:
  base_lr *= 25. / loc_weight
elif normalization_mode == P.Loss.FULL:
  # Roughly there are 2000 prior bboxes per image.
  # TODO(weiliu89): Estimate the exact # of priors.
  base_lr *= 2000.

# 트레이닝 중간 평가용 데이터(처음 설정한 test_lmdb내 영상) 개수 설정
num_test_image = 3769
# 테스트시 사용할 batch size 설정
test_batch_size = 1
test_iter = int(math.ceil(float(num_test_image) / test_batch_size))

# solver 파라미터 설정
solver_param = {
    # Train parameters
    'base_lr': base_lr,
    'weight_decay': 0.0005,
    'lr_policy': "poly",
    'power':0.9,
    'gamma': 0.1,
    'momentum': 0.9,
    'iter_size': iter_size,
    'max_iter': 120000,
    'snapshot': 10000,
    'display': 10,
    'average_loss': 10,
    'type': "SGD",
    'solver_mode': solver_mode,
    'device_id': device_id,
    'debug_info': False,
    'snapshot_after_train': True,
    # Test parameters
    'test_iter': [test_iter],
    'test_interval': 50000,
    'eval_type': "detection",
    'ap_version': "11point",
    'test_initialization': False,
    }

# parameters for generating detection output.
det_out_param = {
    'num_classes': num_classes,
    'share_location': share_location,
    'background_label_id': background_label_id,
    'nms_param': {'nms_threshold': 0.45, 'top_k': 400},
    'save_output_param': {
        'output_directory': output_result_dir,
        'output_name_prefix': "comp4_det_test_",
        'output_format': "VOC",
        'label_map_file': label_map_file,
        'name_size_file': name_size_file,
        'num_test_image': num_test_image,
        },
    'keep_top_k': 200,
    'confidence_threshold': 0.01,
    'code_type': code_type,
    }

# parameters for evaluating detection results.
det_eval_param = {
    'num_classes': num_classes,
    'background_label_id': background_label_id,
    'overlap_threshold': 0.5,
    'evaluate_difficult_gt': False,
    'name_size_file': name_size_file,
    }

# 파일 유무 확인
check_if_exist(train_data)
check_if_exist(test_data)
check_if_exist(label_map_file)
check_if_exist(pretrain_model)
make_if_not_exist(save_dir)
make_if_not_exist(job_dir)
make_if_not_exist(snapshot_dir)

# Create train net.
net = caffe.NetSpec()
net.data, net.label = CreateAnnotatedDataLayer(train_data, batch_size=batch_size_per_device,
        train=True, output_label=True, label_map_file=label_map_file,
        transform_param=train_transform_param, batch_sampler=batch_sampler)

Inception_v2_8_PreActRes_Conv3x3_basic_SSD(net, from_layer='data',global_pool=False)

AddExtraLayers(net, use_batchnorm, lr_mult=lr_mult)

mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
        use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
        aspect_ratios=aspect_ratios, steps=steps, normalizations=normalizations,
        num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
        prior_variance=prior_variance, kernel_size=3, pad=1, lr_mult=lr_mult)

# Create the MultiBoxLossLayer.
name = "mbox_loss"
mbox_layers.append(net.label)
net[name] = L.MultiBoxLoss(*mbox_layers, multibox_loss_param=multibox_loss_param,
        loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')),
        propagate_down=[True, True, False, False])

with open(train_net_file, 'w') as f:
    print('name: "{}_train"'.format(model_name), file=f)
    print(net.to_proto(), file=f)
shutil.copy(train_net_file, job_dir)

# Create test net.
net = caffe.NetSpec()
net.data, net.label = CreateAnnotatedDataLayer(test_data, batch_size=test_batch_size,
        train=False, output_label=True, label_map_file=label_map_file,
        transform_param=test_transform_param)

Inception_v2_8_PreActRes_Conv3x3_basic_SSD(net, from_layer='data',global_pool=False)

AddExtraLayers(net, use_batchnorm, lr_mult=lr_mult)

mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
        use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
        aspect_ratios=aspect_ratios, steps=steps, normalizations=normalizations,
        num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
        prior_variance=prior_variance, kernel_size=3, pad=1, lr_mult=lr_mult)

conf_name = "mbox_conf"
if multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.SOFTMAX:
  reshape_name = "{}_reshape".format(conf_name)
  net[reshape_name] = L.Reshape(net[conf_name], shape=dict(dim=[0, -1, num_classes]))
  softmax_name = "{}_softmax".format(conf_name)
  net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
  flatten_name = "{}_flatten".format(conf_name)
  net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
  mbox_layers[1] = net[flatten_name]
elif multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.LOGISTIC:
  sigmoid_name = "{}_sigmoid".format(conf_name)
  net[sigmoid_name] = L.Sigmoid(net[conf_name])
  mbox_layers[1] = net[sigmoid_name]

net.detection_out = L.DetectionOutput(*mbox_layers,
    detection_output_param=det_out_param,
    include=dict(phase=caffe_pb2.Phase.Value('TEST')))
net.detection_eval = L.DetectionEvaluate(net.detection_out, net.label,
    detection_evaluate_param=det_eval_param,
    include=dict(phase=caffe_pb2.Phase.Value('TEST')))

with open(test_net_file, 'w') as f:
    print('name: "{}_test"'.format(model_name), file=f)
    print(net.to_proto(), file=f)
shutil.copy(test_net_file, job_dir)

# Create deploy net.
# Remove the first and last layer from test net.
deploy_net = net
with open(deploy_net_file, 'w') as f:
    net_param = deploy_net.to_proto()
    # Remove the first (AnnotatedData) and last (DetectionEvaluate) layer from test net.
    del net_param.layer[0]
    del net_param.layer[-1]
    net_param.name = '{}_deploy'.format(model_name)
    net_param.input.extend(['data'])
    net_param.input_shape.extend([
        caffe_pb2.BlobShape(dim=[1, 3, resize_height, resize_width])])
    print(net_param, file=f)
shutil.copy(deploy_net_file, job_dir)

# Create solver.
solver = caffe_pb2.SolverParameter(
        train_net=train_net_file,
        test_net=[test_net_file],
        snapshot_prefix=snapshot_prefix,
        **solver_param)

with open(solver_file, 'w') as f:
    print(solver, file=f)
shutil.copy(solver_file, job_dir)

max_iter = 0
# Find most recent snapshot.
for file in os.listdir(snapshot_dir):
  if file.endswith(".solverstate"):
    basename = os.path.splitext(file)[0]
    iter = int(basename.split("{}_iter_".format(model_name))[1])
    if iter > max_iter:
      max_iter = iter

train_src_param = '--weights="{}" \\\n'.format(pretrain_model)
if resume_training:
  if max_iter > 0:
    train_src_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(snapshot_prefix, max_iter)

if remove_old_models:
  # Remove any snapshots smaller than max_iter.
  for file in os.listdir(snapshot_dir):
    if file.endswith(".solverstate"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(snapshot_dir, file))
    if file.endswith(".caffemodel"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(snapshot_dir, file))

# Create job file.
with open(job_file, 'w') as f:
  f.write('cd {}\n'.format(caffe_root))
  f.write('./build/tools/caffe train \\\n')
  f.write('--solver="{}" \\\n'.format(solver_file))
  f.write(train_src_param)
  
  #jobs with time stamps
  now = datetime.datetime.now()

  if solver_param['solver_mode'] == P.Solver.GPU:
    f.write('--gpu {} 2>&1 | tee {}/{}_{}-{}-{}-{}:{}:{}.log\n'.format(gpus, job_dir, model_name,now.year,now.month, now.day, now.hour, now.minute, now.second))
  else:
    f.write('2>&1 | tee {}/{}.log\n'.format(job_dir, model_name))


# Copy the python script to job_dir.
py_file = os.path.abspath(__file__)
shutil.copy(py_file, job_dir)

# Run the job.
os.chmod(job_file, stat.S_IRWXU)
if run_soon:
  subprocess.call(job_file, shell=True)
