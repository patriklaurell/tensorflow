/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cmath>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace elementwise {
namespace {

bool IsNumericSupportedType(const TfLiteType type) {
  return type == kTfLiteFloat32;
}

bool IsLogicalSupportedType(const TfLiteType type) {
  return type == kTfLiteBool;
}

typedef bool (*IsSupportedType)(TfLiteType);
template <IsSupportedType>
TfLiteStatus GenericPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  TF_LITE_ENSURE_EQ(context, input->type, output->type);
  if (!IsSupportedType(input->type)) {
    context->ReportError(context, "Current data type %d is not supported.",
                         input->type);
    return kTfLiteError;
  }
  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

template <typename T>
inline TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node,
                             T func(T), TfLiteType expected_type) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  TF_LITE_ENSURE_EQ(context, input->type, expected_type);
  const int64_t num_elements = NumElements(input);
  const T* in_data = GetTensorData<T>(input);
  T* out_data = GetTensorData<T>(output);
  for (int64_t i = 0; i < num_elements; ++i) {
    out_data[i] = func(in_data[i]);
  }
  return kTfLiteOk;
}

inline TfLiteStatus EvalNumeric(TfLiteContext* context, TfLiteNode* node,
                                float float_func(float)) {
  return EvalImpl<float>(context, node, float_func, kTfLiteFloat32);
}

inline TfLiteStatus EvalLogical(TfLiteContext* context, TfLiteNode* node,
                                bool bool_func(bool)) {
  return EvalImpl<bool>(context, node, bool_func, kTfLiteBool);
}

template <typename T>
TfLiteStatus AbsEvalQuantized(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  // Support only case where input and output have same quantization
  // scale and zero_point for output is smaller than input. This ensures
  // that output does not overflow limits of quantization.
  TF_LITE_ENSURE(context, input->params.scale == output->params.scale);

  const int num_elements =
      MatchingElementsSize(GetTensorShape(input), GetTensorShape(output));
  const T* in_data = tflite::GetTensorData<T>(input);
  T* out_data = tflite::GetTensorData<T>(output);
  int in_zp = static_cast<T>(input->params.zero_point);
  int out_zp = static_cast<T>(output->params.zero_point);
  for (int i = 0; i < num_elements; i++) {
    int val = in_data[i] - in_zp;
    if (val < 0) {
      val = out_zp - val;
    } else {
      val = out_zp + val;
    }
    T clamped_val = static_cast<T>(
        std::min(std::max(val, static_cast<int>(std::numeric_limits<T>::min())),
                 static_cast<int>(std::numeric_limits<T>::max())));
    out_data[i] = clamped_val;
  }
  return kTfLiteOk;
}

TfLiteStatus AbsEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  switch (input->type) {
    case kTfLiteFloat32: {
      return EvalNumeric(context, node, std::abs);
    } break;
    case kTfLiteInt8: {
      return AbsEvalQuantized<int8_t>(context, node);
    } break;
    default: {
      printf("Only float32 and int8 types are supported.");
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus SinEval(TfLiteContext* context, TfLiteNode* node) {
  return EvalNumeric(context, node, std::sin);
}

TfLiteStatus CosEval(TfLiteContext* context, TfLiteNode* node) {
  return EvalNumeric(context, node, std::cos);
}

TfLiteStatus LogEval(TfLiteContext* context, TfLiteNode* node) {
  return EvalNumeric(context, node, std::log);
}

TfLiteStatus SqrtEval(TfLiteContext* context, TfLiteNode* node) {
  return EvalNumeric(context, node, std::sqrt);
}

TfLiteStatus RsqrtEval(TfLiteContext* context, TfLiteNode* node) {
  return EvalNumeric(context, node, [](float f) { return 1.f / std::sqrt(f); });
}

TfLiteStatus SquareEval(TfLiteContext* context, TfLiteNode* node) {
  return EvalNumeric(context, node, [](float f) { return f * f; });
}

TfLiteStatus LogicalNotEval(TfLiteContext* context, TfLiteNode* node) {
  return EvalLogical(context, node, [](bool v) { return !v; });
}

}  // namespace
}  // namespace elementwise

TfLiteRegistration* Register_ABS() {
  static TfLiteRegistration r = {
      /*init=*/nullptr, /*free=*/nullptr,
      elementwise::GenericPrepare<elementwise::IsNumericSupportedType>,
      elementwise::AbsEval};
  return &r;
}

TfLiteRegistration* Register_SIN() {
  static TfLiteRegistration r = {
      /*init=*/nullptr, /*free=*/nullptr,
      elementwise::GenericPrepare<elementwise::IsNumericSupportedType>,
      elementwise::SinEval};
  return &r;
}

TfLiteRegistration* Register_COS() {
  static TfLiteRegistration r = {
      /*init=*/nullptr, /*free=*/nullptr,
      elementwise::GenericPrepare<elementwise::IsNumericSupportedType>,
      elementwise::CosEval};
  return &r;
}

TfLiteRegistration* Register_LOG() {
  static TfLiteRegistration r = {
      /*init=*/nullptr, /*free=*/nullptr,
      elementwise::GenericPrepare<elementwise::IsNumericSupportedType>,
      elementwise::LogEval};
  return &r;
}

TfLiteRegistration* Register_SQRT() {
  static TfLiteRegistration r = {
      /*init=*/nullptr, /*free=*/nullptr,
      elementwise::GenericPrepare<elementwise::IsNumericSupportedType>,
      elementwise::SqrtEval};
  return &r;
}

TfLiteRegistration* Register_RSQRT() {
  static TfLiteRegistration r = {
      /*init=*/nullptr, /*free=*/nullptr,
      elementwise::GenericPrepare<elementwise::IsNumericSupportedType>,
      elementwise::RsqrtEval};
  return &r;
}

TfLiteRegistration* Register_SQUARE() {
  static TfLiteRegistration r = {
      /*init=*/nullptr, /*free=*/nullptr,
      elementwise::GenericPrepare<elementwise::IsNumericSupportedType>,
      elementwise::SquareEval};
  return &r;
}

TfLiteRegistration* Register_LOGICAL_NOT() {
  static TfLiteRegistration r = {
      /*init=*/nullptr, /*free=*/nullptr,
      elementwise::GenericPrepare<elementwise::IsLogicalSupportedType>,
      elementwise::LogicalNotEval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
