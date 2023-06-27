/*
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "main_functions.h"

#include "constants.h"
#include "model.hpp"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <tensorflow/lite/micro/kernels/micro_ops.h>

#include <math.h>
#include <stdint.h>

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/usb/usb_device.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/drivers/gpio.h>

#define LED0_NODE DT_ALIAS(led0)
static const struct gpio_dt_spec led = GPIO_DT_SPEC_GET(LED0_NODE, gpios);



/* Globals, used for compatibility with Arduino-style sketches. */
namespace {
	// tflite::ErrorReporter *error_reporter = nullptr;
	const tflite::Model *model = nullptr;
	tflite::MicroInterpreter *interpreter = nullptr;
	TfLiteTensor *input = nullptr;
	TfLiteTensor *output = nullptr;

	//constexpr int kTensorArenaSize = 170 * 1024;
	constexpr int kTensorArenaSize = 230 * 1024;
	uint8_t tensor_arena[kTensorArenaSize];
}  /* namespace */

uint8_t setup_failed = 1;


/* The name of this function is important for Arduino compatibility. */
void setup(void)
{
	/* Initialize LED. */
	if (!device_is_ready(led.port)) {
		return;
	}
	gpio_pin_configure_dt(&led, GPIO_OUTPUT_ACTIVE);

	/* Initialize UART USB communication. */
	const struct device *dev = DEVICE_DT_GET(DT_CHOSEN(zephyr_console));
	uint32_t dtr = 0;

	if (usb_enable(NULL)) {
		return;
	}

	/* Poll if the DTR (Data Terminal Ready) flag was set, and only then continue. */
	while (!dtr) {
		uart_line_ctrl_get(dev, UART_LINE_CTRL_DTR, &dtr);
		/* Toggle LED until DTR flag was set. */
		gpio_pin_toggle_dt(&led);		
		/* Give CPU resources to low priority threads. */
		k_sleep(K_MSEC(100));
	}


	model = tflite::GetModel(g_model);
	if (model->version() != TFLITE_SCHEMA_VERSION) {
		printk("Version error");
		return;
	}

	tflite::MicroMutableOpResolver<95> op_resolver;
	op_resolver.AddAbs();
	op_resolver.AddAdd();
	op_resolver.AddAddN();	
	op_resolver.AddArgMax();
	op_resolver.AddArgMin();
	op_resolver.AddAssignVariable();
	op_resolver.AddAveragePool2D();
	op_resolver.AddBatchToSpaceNd();
	op_resolver.AddBroadcastArgs();
	op_resolver.AddBroadcastTo();
	op_resolver.AddCallOnce();
	op_resolver.AddCast();
	op_resolver.AddCeil();
	op_resolver.AddCircularBuffer();
	op_resolver.AddConcatenation();
	op_resolver.AddConv2D();
	op_resolver.AddCos();
	op_resolver.AddCumSum();
	op_resolver.AddDepthToSpace();
	op_resolver.AddDepthwiseConv2D();
	op_resolver.AddDequantize();
	op_resolver.AddDetectionPostprocess();
	op_resolver.AddDiv();
	op_resolver.AddElu();
	op_resolver.AddEqual();
	op_resolver.AddExp();
	op_resolver.AddExpandDims();
	op_resolver.AddFill();
	op_resolver.AddFloor();
	op_resolver.AddFloorDiv();
	op_resolver.AddFloorMod();
	op_resolver.AddFullyConnected();
	op_resolver.AddGather();
	op_resolver.AddGatherNd();
	op_resolver.AddGreater();
	op_resolver.AddGreaterEqual();
	op_resolver.AddHardSwish();
	op_resolver.AddIf();
	op_resolver.AddL2Normalization();
	op_resolver.AddL2Pool2D();
	op_resolver.AddLeakyRelu();
	op_resolver.AddLess();
	op_resolver.AddLessEqual();
	op_resolver.AddLog();
	op_resolver.AddLogicalAnd();
	op_resolver.AddLogicalNot();
	op_resolver.AddLogicalOr();
	op_resolver.AddLogistic();
	op_resolver.AddLogSoftmax();
	op_resolver.AddMaximum();
	op_resolver.AddMaxPool2D();
	op_resolver.AddMirrorPad();
	op_resolver.AddMean();
	op_resolver.AddMinimum();
	op_resolver.AddMul();
	op_resolver.AddNeg();
	op_resolver.AddNotEqual();
	op_resolver.AddPack();
	op_resolver.AddPad();
	op_resolver.AddPadV2();
	op_resolver.AddPrelu();
	op_resolver.AddQuantize();
	op_resolver.AddReadVariable();
	op_resolver.AddReduceMax();
	op_resolver.AddRelu();
	op_resolver.AddRelu6();
	op_resolver.AddReshape();
	op_resolver.AddResizeBilinear();
	op_resolver.AddResizeNearestNeighbor();
	op_resolver.AddRound();
	op_resolver.AddRsqrt();
	op_resolver.AddSelectV2();
	op_resolver.AddShape();
	op_resolver.AddSin();
	op_resolver.AddSlice();
	op_resolver.AddSoftmax();
	op_resolver.AddSpaceToBatchNd();
	op_resolver.AddSpaceToDepth();
	op_resolver.AddSplit();
	op_resolver.AddSplitV();
	op_resolver.AddSqueeze();
	op_resolver.AddSqrt();
	op_resolver.AddSquare();
	op_resolver.AddSquaredDifference();
	op_resolver.AddStridedSlice();
	op_resolver.AddSub();
	op_resolver.AddSum();
	op_resolver.AddSvdf();
	op_resolver.AddTanh();
	op_resolver.AddTransposeConv();
	op_resolver.AddTranspose();
	op_resolver.AddUnpack();
	op_resolver.AddVarHandle();
	op_resolver.AddWhile();
	op_resolver.AddZerosLike();


	static tflite::MicroInterpreter static_interpreter(model, op_resolver, tensor_arena, kTensorArenaSize, nullptr, nullptr);
	interpreter = &static_interpreter;

	/* Allocate memory from the tensor_arena for the model's tensors. */
	interpreter->AllocateTensors();

	/* Obtain pointers to the model's input and output tensors. */
	input = interpreter->input(0);
	output = interpreter->output(0);

	setup_failed = 0;
}

/* The name of this function is important for Arduino compatibility. */
void loop(void)
{	
	if (setup_failed){
		printk("Setup failed\n");
		return;
	}

	short iterations = 5;

	int64_t all_times = 0;
	int64_t time_stamp;
	int64_t milliseconds_spent;

  	for(int i=0; i < iterations; i++){;
		// start time measurement
		time_stamp = k_uptime_get();

		TfLiteStatus invoke_status = interpreter->Invoke();
		if (invoke_status != kTfLiteOk) {
			printk("Invoke error\n");
			return;
		}

		milliseconds_spent = k_uptime_delta(&time_stamp);
		all_times += milliseconds_spent;

		printf("InfTime: %d%d\n", (int32_t)(milliseconds_spent >> 32), (int32_t)(milliseconds_spent));
		k_sleep(K_SECONDS(5));


	}
}
