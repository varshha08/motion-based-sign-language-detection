/*
  This is the Edge Impulse inference sketch from the code shared in our
  conversation. It reproduces the attempted sketch, not a verified final
  working version. The input buffer repeats one sensor reading across the
  full frame; this may not match the trained model's required sampling window.

  Required:
  - Export the Arduino library from your Edge Impulse project.
  - Install the MPU6050 library used by your setup.
  - Confirm the impulse input format and sampling settings.
*/

#include <ei-varshaar-project-1_inferencing.h>
#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;

void setup() {
  Serial.begin(115200);
  delay(2000);

  Wire.begin(21, 22);
  Wire.setClock(100000);

  Serial.println("Initializing MPU...");
  mpu.initialize();

  if (!mpu.testConnection()) {
    Serial.println("MPU6050 FAILED");
    delay(2000);
  } else {
    Serial.println("MPU6050 READY");
  }
}

void loop() {
  int16_t ax, ay, az;
  mpu.getAcceleration(&ax, &ay, &az);

  float input[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];

  for (int i = 0; i < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; i += 3) {
    input[i] = ax / 16384.0;
    input[i + 1] = ay / 16384.0;
    input[i + 2] = az / 16384.0;
  }

  signal_t signal;
  numpy::signal_from_buffer(
    input,
    EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE,
    &signal
  );

  ei_impulse_result_t result;
  EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);

  if (err != EI_IMPULSE_OK) {
    Serial.print("Classifier failed: ");
    Serial.println(err);
    delay(1000);
    return;
  }

  Serial.println("Prediction:");
  for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
    Serial.print(result.classification[i].label);
    Serial.print(": ");
    Serial.println(result.classification[i].value);
  }

  Serial.println("------------------");
  delay(1000);
}
