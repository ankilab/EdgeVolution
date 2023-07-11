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


	// Create OpResolver class with up to 26 kernel support.
	using KeywordOpResolver = tflite::MicroMutableOpResolver<26>;

	KeywordOpResolver* op_resolver = new KeywordOpResolver();
	op_resolver->AddFullyConnected();
	op_resolver->AddReshape();
	op_resolver->AddSoftmax();
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


	static tflite::MicroInterpreter static_interpreter(model, *op_resolver, tensor_arena, kTensorArenaSize, nullptr, nullptr);
	interpreter = &static_interpreter;

	/* Allocate memory from the tensor_arena for the model's tensors. */
	interpreter->AllocateTensors();

	size_t size = interpreter->arena_used_bytes();
	printk("size : %d \n", size);

	setup_failed = 0;

	if (setup_failed){
		printk("Setup failed\n");
		return;
	}

	int64_t all_times = 0;
	int64_t time_stamp;
	int64_t milliseconds_spent;


	// start time measurement
	time_stamp = k_uptime_get();
	printk("now invoking...\n");

	TfLiteStatus invoke_status = interpreter->Invoke();
	if (invoke_status != kTfLiteOk) {
			printk("Invoke error\n");
			return;
	}

	milliseconds_spent = k_uptime_delta(&time_stamp);
	all_times += milliseconds_spent;
	printk("Time: %d\n", (int32_t)(milliseconds_spent));
	printk("InfTime: %d%d\n", (int32_t)(milliseconds_spent >> 32), (int32_t)(milliseconds_spent));
	k_sleep(K_SECONDS(5));



	delete[] op_resolver;


}

/* The name of this function is important for Arduino compatibility. */
void loop(void)
{	



}
