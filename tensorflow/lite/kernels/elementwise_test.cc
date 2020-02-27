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

#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class ElementWiseOpBaseModel : public SingleOpModel {
 public:
  int input() const { return input_; }
  int output() const { return output_; }

 protected:
  int input_;
  int output_;
};

class ElementWiseOpFloatModel : public ElementWiseOpBaseModel {
 public:
  ElementWiseOpFloatModel(BuiltinOperator op,
                          std::initializer_list<int> input_shape) {
    input_ = AddInput(TensorType_FLOAT32);
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(op, BuiltinOptions_NONE, 0);
    BuildInterpreter({input_shape});
  }
};

class ElementWiseOpInt8Model : public ElementWiseOpBaseModel {
 public:
  ElementWiseOpInt8Model(const TensorData& input, BuiltinOperator op) {
    input_ = AddInput(input);
    output_ = AddOutput(input);
    SetBuiltinOp(op, BuiltinOptions_NONE, 0);
    BuildInterpreter({GetShape(input_)});
  }

  void SetInput(std::initializer_list<int8_t> data) {
    PopulateTensor(input_, data);
  }

  std::vector<int8_t> GetOutput() { return ExtractVector<int8_t>(output_); }

  std::vector<float> GetDequantizedOutput() {
    return Dequantize<int8_t>(ExtractVector<int8_t>(output_), GetScale(output_),
                              GetZeroPoint(output_));
  }
};

class ElementWiseOpBoolModel : public ElementWiseOpBaseModel {
 public:
  ElementWiseOpBoolModel(BuiltinOperator op,
                         std::initializer_list<int> input_shape) {
    input_ = AddInput(TensorType_BOOL);
    output_ = AddOutput(TensorType_BOOL);
    SetBuiltinOp(op, BuiltinOptions_NONE, 0);
    BuildInterpreter({input_shape});
  }
};

TEST(ElementWise, Sin) {
  ElementWiseOpFloatModel m(BuiltinOperator_SIN, {1, 1, 4, 1});
  m.PopulateTensor<float>(m.input(), {0, 3.1415926, -3.1415926, 1});
  m.Invoke();
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray(ArrayFloatNear({0, 0, 0, 0.84147})));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({1, 1, 4, 1}));
}

TEST(ElementWise, Cos) {
  ElementWiseOpFloatModel m(BuiltinOperator_COS, {1, 1, 4, 1});
  m.PopulateTensor<float>(m.input(), {0, 3.1415926, -3.1415926, 1});
  m.Invoke();
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray(ArrayFloatNear({1, -1, -1, 0.54030})));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({1, 1, 4, 1}));
}

TEST(ElementWise, Log) {
  ElementWiseOpFloatModel m(BuiltinOperator_LOG, {1, 1, 4, 1});
  m.PopulateTensor<float>(m.input(), {1, 3.1415926, 1, 1});
  m.Invoke();
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray(ArrayFloatNear({0, 1.14473, 0, 0})));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({1, 1, 4, 1}));
}

TEST(FloatActivationsOpTest, Abs) {
  ElementWiseOpFloatModel m(BuiltinOperator_ABS, {1, 2, 4, 1});
  m.PopulateTensor<float>(m.input(), {
                                         0.f, -6.2f, 2.f, 4.f,  //
                                         3.f, -2.f, 10.f, 1.f,  //
                                     });
  m.Invoke();
  EXPECT_THAT(m.ExtractVector<float>(m.output()), ElementsAreArray({
                                                      0.f, 6.2f, 2.f, 4.f,  //
                                                      3.f, 2.f, 10.f, 1.f,  //
                                                  }));
}

TEST(ElementWise, Sqrt) {
  ElementWiseOpFloatModel m(BuiltinOperator_SQRT, {1, 1, 4, 1});
  m.PopulateTensor<float>(m.input(), {0, 1, 2, 4});
  m.Invoke();
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray(ArrayFloatNear({0, 1, 1.41421, 2})));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({1, 1, 4, 1}));
}

TEST(ElementWise, Rsqrt) {
  ElementWiseOpFloatModel m(BuiltinOperator_RSQRT, {1, 1, 4, 1});
  m.PopulateTensor<float>(m.input(), {1, 2, 4, 9});
  m.Invoke();
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray(ArrayFloatNear({1, 0.7071, 0.5, 0.33333})));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({1, 1, 4, 1}));
}

TEST(ElementWise, Square) {
  ElementWiseOpFloatModel m(BuiltinOperator_SQUARE, {1, 1, 4, 1});
  m.PopulateTensor<float>(m.input(), {1, 2, 0.5, -3.0});
  m.Invoke();
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray(ArrayFloatNear({1, 4.0, 0.25, 9.0})));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({1, 1, 4, 1}));
}

TEST(ElementWise, LogicalNot) {
  ElementWiseOpBoolModel m(BuiltinOperator_LOGICAL_NOT, {1, 1, 4, 1});
  m.PopulateTensor<bool>(m.input(), {true, false, true, false});
  m.Invoke();
  EXPECT_THAT(m.ExtractVector<bool>(m.output()),
              ElementsAreArray({false, true, false, true}));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({1, 1, 4, 1}));
}

TEST(Int8ActivationsOpTest, Abs1) {
  // Calculate quantized abs
  TensorData td;
  td.type = TensorType_INT8;
  td.shape = {1, 1, 4, 1};
  td.scale = 0.01;
  td.zero_point = 0;
  ElementWiseOpInt8Model m(td, BuiltinOperator_ABS);
  m.SetInput({-127, 0, -0, 127});
  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAreArray({127, 0, 0, 127}));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray(td.shape));
}

TEST(Int8ActivationsOpTest, Abs2) {
  // Calculate quantized abs
  TensorData td;
  td.type = TensorType_INT8;
  td.shape = {1, 1, 4, 1};
  td.scale = 0.01;
  td.zero_point = 0;
  ElementWiseOpInt8Model m(td, BuiltinOperator_ABS);
  m.SetInput({-128, 0, -0, 127});
  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAreArray({127, 0, 0, 127}));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray(td.shape));
}

TEST(Int8ActivationsOpTest, Abs3) {
  // Calculate quantized abs
  TensorData td;
  td.type = TensorType_INT8;
  td.shape = {1, 1, 4, 1};
  td.scale = 0.01;
  td.zero_point = -10;
  ElementWiseOpInt8Model m(td, BuiltinOperator_ABS);
  m.SetInput({-128, 0, -0, 12});
  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAreArray({108, 0, 0, 12}));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray(td.shape));
}

}  // namespace
}  // namespace tflite
