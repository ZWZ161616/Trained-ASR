#!/bin/bash
. ./path.sh
. ./cmd.sh
# Copyright 
# Apache 2.0
#檢查kaldi所需的東西
#cd kaldi/tools
#extras/check_dependencies.sh

# Caution: some of the graph creation steps use quite a bit of memory, so you
# should run this on a machine that has sufficient memory.

#路徑cd到S5

data=data/train
data2=data/train/iv
data3=data/train/iv_nopitch

d_data=data/inside/
d_data3=data/inside/iv_nopitch

test_name=Ho3
test_data=data/$test_name
test_data2=data/$test_name/iv
test_data3=data/$test_name/iv_nopitch

nj=8 #CPU 線程
passward=c7789520 #使用者密碼

dos2unix $data/text
dos2unix $d_data/text
dos2unix $test_data/text
### features extraction###
echo -e "#####Compute MFCC#####"
#steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj $nj --cmd run.pl $data $data/log $data/feats || exit 1
echo -e "#####Compute CMVN#####"
#steps/compute_cmvn_stats.sh $data $data/log $data/feats || exit 1 #正規化
echo -e "#####Finish feature extracion#####"
### features extraction end###

### features extraction###
#echo -e "#####Compute MFCC#####"
#steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj $nj --cmd run.pl $d_data $d_data/log $d_data/feats || exit 1
#echo -e "#####Compute CMVN#####"
#steps/compute_cmvn_stats.sh $d_data $d_data/log $d_data/feats || exit 1 #正規化
#echo -e "#####Finish feature extracion#####"
### features extraction end###


echo -e "### Monophone start###"
#steps/train_mono.sh --cmd run.pl --nj $nj $data data/lang_test_learning exp_learning/inside/mono || exit 1  #與train_mono.sh差別在beam (當音檔較大時調整)
#utils/mkgraph.sh data/lang_test_learning exp_learning/inside/mono exp_learning/inside/mono/graph || exit 1
#steps/decode.sh --cmd run.pl --nj 1 --config conf/decode.config exp_learning/inside/mono/graph $d_data exp_learning/inside/mono/decode_mono || exit 1
#steps/align_si.sh --cmd run.pl --nj $nj $data data/lang_test_learning exp_learning/inside/mono exp_learning/inside/mono_ali || exit 1
echo -e "### Monophone End###"

echo -e "##Triphone DELTA start###"
#steps/train_deltas.sh --cmd run.pl 2500 20000 $data data/lang_test_learning exp_learning/inside/mono_ali exp_learning/inside/DELTA_tri1 || exit 1
#utils/mkgraph.sh data/lang_test_learning exp_learning/inside/DELTA_tri1 exp_learning/inside/DELTA_tri1/graph || exit 1
#steps/decode.sh --cmd run.pl --nj 1 --config conf/decode.config exp_learning/inside/DELTA_tri1/graph $d_data exp_learning/inside/DELTA_tri1/decode_DELTA || exit 1
#steps/align_si.sh --cmd run.pl --nj $nj $data data/lang_test_learning exp_learning/inside/DELTA_tri1 exp_learning/inside/DELTA_tri1_ali || exit 1
echo -e "##Triphone DELTA End###"

echo -e "##Triphone DELTA+DELTA+DELTA start##"
#steps/train_deltas.sh --cmd run.pl 2500 20000 $data data/lang_test_learning exp_learning/inside/DELTA_tri1_ali exp_learning/inside/DELTA3_tri2 || exit 1
#utils/mkgraph.sh data/lang_test_learning exp_learning/inside/DELTA3_tri2 exp_learning/inside/DELTA3_tri2/graph || exit 1
#steps/decode.sh --cmd run.pl --nj 1 --config conf/decode.config  exp_learning/inside/DELTA3_tri2/graph $d_data exp_learning/inside/DELTA3_tri2/decode_DELTA3 || exit 1
#steps/align_si.sh --cmd run.pl --nj $nj $data data/lang_test_learning exp_learning/inside/DELTA3_tri2 exp_learning/inside/DELTA3_tri2_ali || exit 1
echo -e "##Triphone DELTA+DELTA+DELTA End###"

echo -e "##Triphone LDA+MLLT##"
#steps/train_lda_mllt.sh --cmd run.pl 2500 20000 $data data/lang_test_learning exp_learning/inside/DELTA3_tri2_ali exp_learning/inside/LDA_tri3 || exit 1
#utils/mkgraph.sh data/lang_test_learning exp_learning/inside/LDA_tri3 exp_learning/inside/LDA_tri3/graph || exit 1
#steps/decode.sh --cmd run.pl --nj 1 --config conf/decode.config exp_learning/inside/LDA_tri3/graph $d_data exp_learning/inside/LDA_tri3/decode_LDA || exit 1
#steps/align_fmllr.sh --cmd run.pl --nj $nj $data data/lang_test_learning exp_learning/inside/LDA_tri3 exp_learning/inside/LDA_tri3_ali || exit 1
echo -e "##Triphone LDA+MLLT End##"

echo -e "##Triphone SAT##"
#steps/train_sat.sh --cmd run.pl 3500 100000 $data data/lang_test_learning exp_learning/inside/LDA_tri3_ali exp_learning/inside/SAT_tri4 || exit 1
#utils/mkgraph.sh data/lang_test_learning exp_learning/inside/SAT_tri4 exp_learning/inside/SAT_tri4/graph || exit 1
#steps/decode_fmllr.sh --cmd run.pl --nj 1 --config conf/decode.config exp_learning/inside/SAT_tri4/graph $d_data exp_learning/inside/SAT_tri4/decode_SAT || exit 1
#steps/align_fmllr.sh --cmd run.pl --nj $nj $data data/lang_test_learning exp_learning/inside/SAT_tri4 exp_learning/inside/SAT_tri4_ali || exit 1
echo -e "##Triphone SAT End##"

%%%%%%%%i-vector

echo -e "##Generate feature##"
echo -e "#train"
#steps/make_mfcc_pitch.sh --cmd run.pl --nj $nj --mfcc-config conf/mfcc_hires.conf $data $data2/log $data2/feats  || exit 1
#steps/compute_cmvn_stats.sh $data $data2/log $data2/feats  || exit 1
#utils/data/limit_feature_dim.sh 0:39 $data $data3  || exit 1
#steps/compute_cmvn_stats.sh $data3 $data3/log $data3/feats  || exit 1

echo -e "#inside_test"
#steps/make_mfcc_pitch.sh --cmd run.pl --nj $nj --mfcc-config conf/mfcc_hires.conf $d_data $d_data/log $d_data/feats  || exit 1
#steps/compute_cmvn_stats.sh $d_data $d_data/log $d_data/feats  || exit 1
#utils/data/limit_feature_dim.sh 0:39 $d_data $d_data3  || exit 1
#steps/compute_cmvn_stats.sh $d_data3 $d_data3/log $d_data3/feats  || exit 1
 
echo -e "#outside_test"
#steps/make_mfcc_pitch.sh --cmd run.pl --nj 1 --mfcc-config conf/mfcc_hires.conf $test_data $test_data/log $test_data/feats  || exit 1
#steps/compute_cmvn_stats.sh $test_data $test_data/log $test_data/feats  || exit 1
#utils/data/limit_feature_dim.sh 0:39 $test_data $test_data3  || exit 1
#steps/compute_cmvn_stats.sh $test_data $test_data3/log $test_data3/feats  || exit 1


echo -e "##Train i-Vector"
echo -e "###Computing a PCA transform from the hires data."
#steps/online/nnet2/get_pca_transform.sh --cmd run.pl --splice-opts "--left-context=3 --right-context=3" --max-utts 10000 --subsample 2 $data3 exp_learning/inside/nnet3/pca_transform  || exit 1

echo -e "###Training the diagonal UBM. Use 512 Gaussians in the UBM."  
#steps/online/nnet2/train_diag_ubm.sh --cmd run.pl --nj $nj --num-frames 700000 --num-threads 16 $data3 512 exp_learning/inside/nnet3/pca_transform exp_learning/inside/nnet3/diag_ubm  || exit 1

echo -e "####Training the iVector extractor"
#steps/online/nnet2/train_ivector_extractor.sh --cmd run.pl --nj $nj --num-processes 1 $data3 exp_learning/inside/nnet3/diag_ubm exp_learning/inside/nnet3/extractor  || exit 1

echo -e "####Extracting iVectors for training"
#steps/online/nnet2/extract_ivectors_online.sh --cmd run.pl --nj $nj $data3 exp_learning/inside/nnet3/extractor $data3/ivectors  || exit 1

echo -e "####Extracting iVectors for inside"
steps/online/nnet2/extract_ivectors_online.sh --cmd run.pl --nj $nj $d_data3 exp_learning/inside/nnet3/extractor $d_data3/ivectors  || exit 1

echo -e "##Extracting iVectors for testing"
steps/online/nnet2/extract_ivectors_online.sh --cmd run.pl --nj 1 $test_data3 exp_learning/inside/nnet3/extractor $test_data3/ivectors  || exit 1

echo -e "##Prepare for chain model training"
echo -e "###Create topology"

cp -r data/lang_test_learning data/lang_chain
silphonelist=$(cat data/lang_chain/phones/silence.csl)
nonsilphonelist=$(cat data/lang_chain/phones/nonsilence.csl)
echo -e "## Use our special topology... note that later on may have to tune this"
steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist > data/lang_chain/topo

echo -e "#Get alignment as lattice"
steps/align_fmllr_lats.sh --nj $nj --cmd run.pl $data3 data/lang_test_learning exp_learning/inside/SAT_tri4 exp_learning/inside/lats  || exit 1

echo -e "##Build tree"
steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 --context-opts "--context-width=2 --central-position=1" --cmd run.pl 4200 $data3 data/lang_chain exp_learning/inside/SAT_tri4_ali exp_learning/inside/nnet3/tri4_biphone_tree  || exit 1

echo -e "##Creating neural network config using the xconfig"
. path.sh
xent_regularize=0.1
num_targets=$(tree-info exp_learning/inside/nnet3/tri4_biphone_tree/tree | grep num-pdfs | awk '{print $2}')
learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
affine_opts="l2-regularize=0.0005 dropout-proportion=0.5 dropout-per-dim=true dropout-per-dim-continuous=true"
tdnnf_opts="l2-regularize=0.0005 dropout-proportion=0.5 bypass-scale=0.75"
linear_opts="l2-regularize=0.005 orthonormal-constraint=-1.0"
prefinal_opts="l2-regularize=0.0005"
output_opts="l2-regularize=0.00005"


#dir=exp_learning/inside/chain/tdnnf_sp_13layers_128n_10epochs_lr0.015_xent0.0005
#dir=exp_learning/inside/chain/tdnnf_0105_Augment_Fake_256_128_9_yukang
dir=exp_learning/inside/chain/tdnnf_Augment_128_64_14_0.2_yukang
mkdir -p $dir/configs

cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat
  #idct-layer name=idct input=input dim=40 cepstral-lifter=22 affine-transform-file=$dir/configs/idct.mat
  #batchnorm-component name=batchnorm0 input=idct
  #spec-augment-layer name=spec-augment freq-max-proportion=0.5 time-zeroed-proportion=0.2 time-mask-max-frames=20
  #delta-layer name=delta input=spec-augment
  #no-op-component name=input2 input=Append(delta, Scale(0.4, ReplaceIndex(ivector, t, 0)))

  
  relu-batchnorm-dropout-layer name=tdnn1 $affine_opts dim=128
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=128 bottleneck-dim=64 time-stride=14
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=128 bottleneck-dim=64 time-stride=14
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=128 bottleneck-dim=64 time-stride=14
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=128 bottleneck-dim=64 time-stride=14
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=128 bottleneck-dim=64 time-stride=14
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=128 bottleneck-dim=64 time-stride=14
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=128 bottleneck-dim=64 time-stride=14
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=128 bottleneck-dim=64 time-stride=14
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=128 bottleneck-dim=64 time-stride=14
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=128 bottleneck-dim=64 time-stride=14
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=128 bottleneck-dim=64 time-stride=14
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=128 bottleneck-dim=64 time-stride=14
  

  linear-component name=prefinal-l dim=64 $linear_opts
  
  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=128 small-dim=64
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts
  
  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=128 small-dim=64
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF

steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/  || exit 1

train_stage=-10
train_ivector_dir=$data3/ivectors
common_egs_dir=
get_egs_stage=-10
frames_per_eg=50
dropout_schedule='0,0@0.20,0.5@0.50,0'
remove_egs=true
train_data_dir=$data3
tree_dir=exp_learning/inside/nnet3/tri4_biphone_tree
lat_dir=exp_learning/inside/lats
steps/nnet3/chain/train.py --stage $train_stage \
    --cmd run.pl \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0 --constrained false" \
    --egs.chunk-width $frames_per_eg \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 2500000 \
    --trainer.num-epochs 5 \
    --trainer.optimization.num-jobs-initial 1 \
    --trainer.optimization.num-jobs-final 1 \
    --trainer.optimization.initial-effective-lrate 0.015 \
    --trainer.optimization.final-effective-lrate 0.0015 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir

echo -e "#Build graph for decoding"
utils/mkgraph.sh --self-loop-scale 1.0 data/lang_chain $dir $dir/graph  || exit 1

echo -e "#Prepare online decoding directory"
steps/online/nnet3/prepare_online_decoding.sh --mfcc-config conf/mfcc_hires.conf --add-pitch false data/lang_chain exp_learning/inside/nnet3/extractor ${dir} ${dir}_online  || exit 1

echo -e "#Online inside decoding"
steps/online/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --nj 1 --cmd run.pl $dir/graph $d_data3 ${dir}_online/decode_inside  || exit 1    #####inside

echo -e "#Online outside decoding"
steps/online/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --nj 1 --cmd run.pl $dir/graph $test_data3 ${dir}_online/decode_$test_name  || exit 1     #####outside
