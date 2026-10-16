# Motion-Based Sign Language Detection Using TinyML

## Project Overview
This project aims to recognize motion-based gestures using an ESP32 microcontroller, an MPU6050 accelerometer and gyroscope sensor, and a TinyML model.

The system is designed to classify four gesture categories:
- FOOD
- WATER
- HELP
- MEDICINE

## Hardware Requirements
- ESP32 development board
- MPU6050 accelerometer and gyroscope sensor
- Jumper wires
- USB cable

## Software and Technologies
- Arduino IDE
- Python
- TensorFlow / Keras
- TinyML
- MPU6050 sensor library

## Hardware Connections

| MPU6050 | ESP32 |
|---|---|
| VCC | 3.3V |
| GND | GND |
| SDA | GPIO 21 |
| SCL | GPIO 22 |

## Working Principle
1. The MPU6050 captures motion data.
2. The ESP32 reads sensor measurements through I2C.
3. The collected sensor data is processed into features required by the model.
4. The TinyML model predicts the corresponding gesture.
5. The predicted gesture is displayed through the serial output.

## Project Structure
- `Arduino_Code/` – ESP32 firmware
- `Model_Training/` – Python model training code
- `Dataset/` – Sensor recordings, if available
- `Images/` – Project photographs and output screenshots

## Results
Add the actual model accuracy and hardware testing results after verifying them.

## Limitations
The prediction performance depends on sensor data quality, training data, and consistency between training and deployment preprocessing.

## Future Improvements
- Improve gesture recognition accuracy.
- Test with more users and different motion patterns.
- Optimize the model for efficient on-device inference.

## Author
Varsha
