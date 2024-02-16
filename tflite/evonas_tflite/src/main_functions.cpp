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

	//constexpr int kTensorArenaSize = 170 * 1024;
	constexpr int kTensorArenaSize = 155 * 1024;
	uint8_t tensor_arena[kTensorArenaSize];
}  /* namespace */

uint32_t MeasureTfLiteEvalTensors(const tflite::Model* model) {
	uint32_t total_size = 0;
	for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs()->size();
		subgraph_idx++) {
		const tflite::SubGraph* subgraph = model->subgraphs()->Get(subgraph_idx);

		size_t alloc_count = subgraph->tensors()->size();
		total_size += sizeof(TfLiteEvalTensor) * alloc_count;
		
	}
	return total_size;
}

uint32_t MeasureNodeAndRegistrations(const tflite::Model* model) {
	uint32_t total_size = 0;
	for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs()->size();
		subgraph_idx++) {
		const tflite::SubGraph* subgraph = model->subgraphs()->Get(subgraph_idx);
		uint32_t operators_size = tflite::NumSubgraphOperators(subgraph);
		uint32_t size= sizeof(tflite::NodeAndRegistration) * operators_size;
		total_size += size;
	}
	return total_size;
}


uint32_t MeasureTotalSize(const tflite::Model* model) {
	uint32_t total_size = 0;

	total_size += sizeof(tflite::internal::ScratchBufferRequest) * 12;

	//total_size += tflite::getMicroBuiltinDataAllocatorSize();

	// Allocate struct to store eval tensors, nodes and registrations.
	total_size += sizeof(tflite::SubgraphAllocations) * model->subgraphs()->size();
	
	// nodes and registrations get allocated extra and are referred by SubgraphAllocations struct
	total_size += MeasureNodeAndRegistrations(model);
	total_size += MeasureTfLiteEvalTensors(model);

	uint32_t input_size = sizeof(TfLiteTensor*) * model->subgraphs()->Get(0)->inputs()->size();
	total_size += input_size;

	uint32_t output_size = sizeof(TfLiteTensor*) * model->subgraphs()->Get(0)->outputs()->size();
	total_size += output_size;

	return total_size;		
}
uint8_t setup_failed = 1;




/* The name of this function is important for Arduino compatibility. */
__attribute__((optimize(0))) void setup(void)
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

	/* Turn LED on to indicate that DTR flag was set */
	gpio_pin_set_dt(&led, 1);	

	printk("connected \n");

	model = tflite::GetModel(g_model);
	if (model->version() != TFLITE_SCHEMA_VERSION) {
		printk("Version error");
		return;
	}

	uint32_t measured = MeasureTotalSize(model);
	printk("measured used bytes : %d \n", measured);


	// Create OpResolver class with up to 26 kernel support.
	using KeywordOpResolver = tflite::MicroMutableOpResolver<38>;

	KeywordOpResolver* op_resolver = new KeywordOpResolver();
	op_resolver->AddFullyConnected();
	op_resolver->AddReshape();
	op_resolver->AddSoftmax();
	op_resolver->AddLeakyRelu();
	op_resolver->AddRelu();
	op_resolver->AddTanh();
	op_resolver->AddTranspose();
	op_resolver->AddSlice();
	op_resolver->AddGather();
	op_resolver->AddMul();
	op_resolver->AddPack();
	op_resolver->AddSum();
	op_resolver->AddQuantize();
	op_resolver->AddDequantize();
	op_resolver->AddDepthwiseConv2D();
	op_resolver->AddMaxPool2D();
	op_resolver->AddAdd();
	op_resolver->AddConv2D();
	op_resolver->AddSquaredDifference();
	op_resolver->AddRsqrt();
	op_resolver->AddSub();
	op_resolver->AddSqrt();
	op_resolver->AddSquare();
	op_resolver->AddMean();
	op_resolver->AddReduceMax();
	op_resolver->AddAveragePool2D();
	op_resolver->AddExpandDims();
	op_resolver->AddShape();
	op_resolver->AddConcatenation();
	op_resolver->AddLogistic();

	op_resolver->AddRange();
	op_resolver->AddPad();
	op_resolver->AddSplit();
	op_resolver->AddLog();

	op_resolver->AddSplitV();
	op_resolver->AddFloorDiv();
	op_resolver->AddStridedSlice();
	op_resolver->AddMaximum();
	//op_resolver->AddRfft2D();

	printk("added operations\n");
	static tflite::MicroInterpreter static_interpreter(model, *op_resolver, tensor_arena, kTensorArenaSize, nullptr, nullptr);
	interpreter = &static_interpreter;
	printk("init interpreter done\n");

	/* Allocate memory from the tensor_arena for the model's tensors. */
	TfLiteStatus allocate_status = interpreter->AllocateTensors();
	size_t size = interpreter->arena_used_bytes();
	printk("tensorarena used bytes : %d \n", size);

	if (allocate_status != kTfLiteOk) {
		printk("AllocateTensors() failed\n");
		return;
	}
	printk("allocating done\n");
	k_sleep(K_MSEC(100));
	setup_failed = 0;

	if (setup_failed){
		printk("Setup failed\n");
		return;
	}

	int64_t all_times = 0;
	int64_t time_stamp;
	int64_t milliseconds_spent;

	k_sleep(K_MSEC(100));
	printk("now invoking...\n");

	// run invoke one time
	// start time measurement
	time_stamp = k_uptime_get();
	
	TfLiteStatus invoke_status = interpreter->Invoke();
	if (invoke_status != kTfLiteOk) {
		printk("Invoke error\n");
		return;
	}

	milliseconds_spent = k_uptime_delta(&time_stamp);

	if (milliseconds_spent >= 200) {
		// just pass by as we need no averaging
	}
	else{
		size_t n_iterations = 10;

		for (size_t i = 0; i < n_iterations; i++){
			time_stamp = k_uptime_get();	

			TfLiteStatus invoke_status = interpreter->Invoke();
			if (invoke_status != kTfLiteOk) {
				printk("Invoke error\n");
				return;
			}
			milliseconds_spent = k_uptime_delta(&time_stamp);
			all_times += milliseconds_spent;
		}

		// divide by 10 to get average time
		milliseconds_spent = all_times / n_iterations;
	}

	printk("Time: %d\n", (int32_t)(milliseconds_spent));
	printk("InfTime: %d%d\n", (int32_t)(milliseconds_spent >> 32), (int32_t)(milliseconds_spent));
	k_sleep(K_SECONDS(5));
}

/* The name of this function is important for Arduino compatibility. */
void loop(void)
{	
	k_sleep(K_FOREVER);
}
