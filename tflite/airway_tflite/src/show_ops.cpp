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

#include "show_ops.h"

#include "model.hpp"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <tensorflow/lite/micro/kernels/micro_ops.h>
#include "tensorflow/lite/micro/micro_graph.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "flatbuffers/flatbuffers.h" // from @flatbuffers
#include "tensorflow/lite/micro/flatbuffer_utils.h"
#include "tensorflow/lite/schema/schema_utils.h"

#include <math.h>
#include <stdint.h>

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/usb/usb_device.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/drivers/gpio.h>

#define LED0_NODE DT_ALIAS(led0)
static const struct gpio_dt_spec led = GPIO_DT_SPEC_GET(LED0_NODE, gpios);

namespace tflite
{

	// tflite::ErrorReporter *error_reporter = nullptr;
	const Model *model = nullptr;

	__attribute__((optimize(0))) void print_ops(void)
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

		// // /* Poll if the DTR (Data Terminal Ready) flag was set, and only then continue. */
		while (!dtr) {
			uart_line_ctrl_get(dev, UART_LINE_CTRL_DTR, &dtr);
			/* Toggle LED until DTR flag was set. */
			gpio_pin_toggle_dt(&led);		
			/* Give CPU resources to low priority threads. */
			k_sleep(K_MSEC(100));
		}	
		printk("connected");

		model = tflite::GetModel(g_model);

		for (unsigned int subgraph_idx = 0; subgraph_idx < model->subgraphs()->size(); subgraph_idx++)
		{
			const SubGraph *subgraph = model->subgraphs()->Get(subgraph_idx);
			TFLITE_DCHECK(subgraph != nullptr);

			auto *opcodes = model->operator_codes();
			uint32_t operators_size = NumSubgraphOperators(subgraph);

			// print the built in codes for the operations
			for (size_t i = 0; i < operators_size; ++i)
			{
				const auto *op = subgraph->operators()->Get(i);
				const size_t index = op->opcode_index();
				const auto *opcode = opcodes->Get(index);
				auto builtin_code = GetBuiltinCode(opcode);
				printk("opcode %i \n", builtin_code);
			}
		}
	}

}
void print_ops(void)
{
	tflite::print_ops();
}
