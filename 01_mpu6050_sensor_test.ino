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
    while (true) {
      delay(1000);
    }
  }

  Serial.println("MPU6050 READY");
}

void loop() {
  int16_t ax, ay, az;
  mpu.getAcceleration(&ax, &ay, &az);

  Serial.print(ax / 16384.0);
  Serial.print(",");
  Serial.print(ay / 16384.0);
  Serial.print(",");
  Serial.println(az / 16384.0);

  delay(50);
}
