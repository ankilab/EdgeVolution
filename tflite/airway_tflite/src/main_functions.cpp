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

#include <tensorflow/lite/micro/all_ops_resolver.h>
#include "constants.h"
#include "model.hpp"
#include "output_handler.hpp"
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>

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
	tflite::ErrorReporter *error_reporter = nullptr;
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

	/*unsigned char *received = NULL;
	int poll_in = 0;

	while (received == NULL) {
		while (!poll_in) {
			poll_in = uart_poll_in(dev, &received);
		}
		printk("waiting\n");
		k_sleep(K_MSEC(100));
	} 

	while (true) {
		printk("Received stop\n");
	}*/
	

	/* Set up logging. Google style is to avoid globals or statics because of
	 * lifetime uncertainty, but since this has a trivial destructor it's okay.
	 * NOLINTNEXTLINE(runtime-global-variables)
	 */

	static tflite::MicroErrorReporter micro_error_reporter;

	error_reporter = &micro_error_reporter;

	/* Map the model into a usable data structure. This doesn't involve any
	 * copying or parsing, it's a very lightweight operation.
	 */
	model = tflite::GetModel(g_model);
	if (model->version() != TFLITE_SCHEMA_VERSION) {
		TF_LITE_REPORT_ERROR(error_reporter,
						"Model provided is schema version %d not equal "
						"to supported version %d.",
						model->version(), TFLITE_SCHEMA_VERSION);
		printk("Version error");
		return;
	}

	/* This pulls in all the operation implementations we need.
	 * NOLINTNEXTLINE(runtime-global-variables)
	 */
	static tflite::AllOpsResolver resolver;

	/* Build an interpreter to run the model with. */
	static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, nullptr, nullptr);
	interpreter = &static_interpreter;

	/* Allocate memory from the tensor_arena for the model's tensors. */
	TfLiteStatus allocate_status = interpreter->AllocateTensors();
	if (allocate_status != kTfLiteOk) {
		TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
		printk("AllocateTensors error\n");
		return;
	}

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

  	for(int i=0; i < iterations; i++){ 
		// start time measurement
		time_stamp = k_uptime_get();

    	// fill input array with data
    	/*for (int j = 0; j < 16000; j++){
			//printf("%d\n", j);
      		input->data.f[j] = ((float)rand()/(float)(RAND_MAX)) * 1.0;
		}*/


		/* Run inference, and report any error */
		TfLiteStatus invoke_status = interpreter->Invoke();

		if (invoke_status != kTfLiteOk) {
			printk("Invoke error\n");
			return;
		}

		milliseconds_spent = k_uptime_delta(&time_stamp);
		all_times += milliseconds_spent;

		printf("InfTime: %d%d\n", (int32_t)(milliseconds_spent >> 32), (int32_t)(milliseconds_spent));
		k_sleep(K_SECONDS(5));
	
		/* Obtain the quantized output from model's output tensor */
		/*int8_t y_quantized = output->data.int8[0];

		/* Dequantize the output from integer to floating-point */
		/*float y = (y_quantized - output->params.zero_point) * output->params.scale;

		const char *ySign = (y < 0) ? "-" : "";
		float yVal = (y < 0) ? -y : y;
		int yInt1 = yVal;
		float yFrac = yVal - yInt1;
		int yInt2 = trunc(yFrac * 1000);

		printf("%s%d.%04d\n", ySign, yInt1, yInt2);*/
	}
	//all_times = all_times / iterations;
	//printf("%d%d\n", (int32_t)(all_times >> 32), (int32_t)(all_times));

	/*float yVal = (all_times < 0) ? -all_times : all_times;
	int yInt1 = yVal;
	float yFrac = yVal - yInt1;
	int yInt2 = trunc(yFrac * 1000);

	printf("%d.%04d\n", yInt1, yInt2);*/
}
