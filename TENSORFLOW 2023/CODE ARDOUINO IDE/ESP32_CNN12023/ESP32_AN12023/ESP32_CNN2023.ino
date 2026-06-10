#include <Arduino.h>
#include <stdint.h>
#include <Wire.h>
#include <Adafruit_INA219.h>
#include <ArduTFLite.h>
#include "cnn_model_data.h"

// ================= CONFIG =================
const int N_FEATURES = 8;
const int N_CLASSES  = 4;

// Mémoire réservée pour TensorFlow Lite Micro
constexpr int kTensorArenaSize = 32 * 1024;
byte tensorArena[kTensorArenaSize];

// ================= INA219 =================
Adafruit_INA219 ina219;

// ================= NORMALISATION =================

float means[N_FEATURES] = {
  82946227.54495819f,
  69268.14200088085f,
  7.406373252679435f,
  2.5964629165718227f,
  65.39542863461945f,
  84.20318805384402f,
  11.032745280479302f,
  83.85474111781087f
};

float scales[N_FEATURES] = {
  20847540.938972678f,
  405467.3164622806f,
  6.709516490165713f,
  26.477828227761f,
  398.28310678053236f,
  228.0282374074317f,
  6.597566103763726f,
  221.6471413772571f
};

// ================= CLASSES =================

const char* CLASS_NAMES[N_CLASSES] = {
  "BenignTraffic",
  "DDoS-ICMP_Flood",
  "DDoS-SYN_Flood",
  "DDoS-UDP_Flood"
};

// ================= PARSE CSV =================

bool parse8Floats(String line, float values[N_FEATURES]) {
  int start = 0;

  for (int i = 0; i < N_FEATURES; i++) {
    int commaIndex;

    if (i < N_FEATURES - 1) {
      commaIndex = line.indexOf(',', start);
      if (commaIndex == -1) return false;

      String token = line.substring(start, commaIndex);
      token.trim();
      values[i] = token.toFloat();
      start = commaIndex + 1;
    } else {
      String token = line.substring(start);
      token.trim();
      values[i] = token.toFloat();
    }
  }

  return true;
}

// ================= NORMALISATION =================

void normalizeFeatures(float raw[N_FEATURES], float norm[N_FEATURES]) {
  for (int i = 0; i < N_FEATURES; i++) {
    if (scales[i] == 0.0f) {
      norm[i] = 0.0f;
    } else {
      norm[i] = (raw[i] - means[i]) / scales[i];
    }
  }
}

// ================= ARGMAX =================

int argmax(float arr[N_CLASSES]) {
  int idx = 0;
  float maxVal = arr[0];

  for (int i = 1; i < N_CLASSES; i++) {
    if (arr[i] > maxVal) {
      maxVal = arr[i];
      idx = i;
    }
  }

  return idx;
}

// ================= DEBUG =================

void printVector(const char* title, float v[N_FEATURES]) {
  Serial.println(title);

  for (int i = 0; i < N_FEATURES; i++) {
    Serial.print("x");
    Serial.print(i + 1);
    Serial.print(" = ");
    Serial.println(v[i], 6);
  }
}

void printOutputs(float outputs[N_CLASSES], int pred) {
  Serial.println("===== OUTPUTS =====");

  for (int i = 0; i < N_CLASSES; i++) {
    Serial.print(CLASS_NAMES[i]);
    Serial.print(" : ");
    Serial.println(outputs[i], 6);
  }

  Serial.println("===== RESULT =====");
  Serial.print("Prediction: ");
  Serial.println(CLASS_NAMES[pred]);

  Serial.print("Class ID: ");
  Serial.println(pred);

  Serial.print("Confidence: ");
  Serial.println(outputs[pred], 6);

  if (pred == 0) {
    Serial.println("=> NORMAL TRAFFIC");
  } else {
    Serial.print("=> ATTACK DETECTED: ");
    Serial.println(CLASS_NAMES[pred]);
  }
}

// ================= MEMOIRE =================

void printMemoryInfo() {
  uint32_t heapTotal = ESP.getHeapSize();
  uint32_t freeHeap  = ESP.getFreeHeap();
  uint32_t usedHeap  = heapTotal - freeHeap;
  uint32_t minFreeHeap = ESP.getMinFreeHeap();

  Serial.println("===== MEMORY INFO =====");

  Serial.print("Heap total (bytes): ");
  Serial.println(heapTotal);

  Serial.print("Free heap (bytes): ");
  Serial.println(freeHeap);

  Serial.print("Used heap (bytes): ");
  Serial.println(usedHeap);

  Serial.print("Min free heap (bytes): ");
  Serial.println(minFreeHeap);

  Serial.print("TensorArena (bytes): ");
  Serial.println(kTensorArenaSize);

  Serial.println("=======================");
}

// ================= SETUP =================

void setup() {
  Serial.begin(115200);
  delay(2000);

  Wire.begin(21, 22);

  Serial.println("==================================");
  Serial.println("CNN MODEL LOADED");
  Serial.println("MODEL_TAG: CNN_4CLASSES_8FEATURES");
  Serial.println("ENERGY_MEASUREMENT: INA219");
  Serial.println("ENERGY_MODE: INFERENCE_ONLY");
  Serial.println("==================================");

  if (!ina219.begin()) {
    Serial.println("ERREUR: INA219 non detecte !");
    Serial.println("Verifier VCC, GND, SDA, SCL");
    while (1) {
      delay(10);
    }
  }

  Serial.println("INA219 detecte avec succes !");

  Serial.println("Memory before modelInit:");
  printMemoryInfo();

  Serial.print("MEAN0 = ");
  Serial.println(means[0], 6);

  Serial.print("SCALE0 = ");
  Serial.println(scales[0], 6);

  if (!modelInit(cnn_model_tflite, tensorArena, kTensorArenaSize)) {
    Serial.println("Erreur modelInit");
    Serial.println("Essaye kTensorArenaSize = 48 * 1024 ou 64 * 1024");

    while (1) {
      delay(1000);
    }
  }

  Serial.println("Model init OK");

  Serial.println("Memory after modelInit:");
  printMemoryInfo();

  Serial.println("ESP32 CNN 4 CLASSES READY");
  Serial.println("Format attendu :");
  Serial.println("IAT,Header_Length,Protocol Type,flow_duration,rst_count,Tot size,Magnitue,AVG");
  Serial.println("Exemple :");
  Serial.println("655.569893,28,1,4.970551,0,50.871651,7.132436,50.871651");
  Serial.println("----------------------------------");
}

// ================= LOOP =================

void loop() {
  if (Serial.available() == 0) return;

  String line = Serial.readStringUntil('\n');
  line.trim();

  if (line.length() == 0) return;

  float raw[N_FEATURES];
  float norm[N_FEATURES];
  float outputs[N_CLASSES];

  if (!parse8Floats(line, raw)) {
    Serial.println("ERROR: bad format");
    Serial.println("Utilise : x1,x2,x3,x4,x5,x6,x7,x8");
    Serial.println("PRED:PARSE_ERROR,CONF:0.0,TIME_US:0,VOLTAGE_V:0,CURRENT_MA:0,POWER_MW:0,ENERGY_J:0,ENERGY_MJ:0,HEAP_TOTAL:0,FREE_HEAP:0,USED_HEAP:0,MIN_FREE_HEAP:0,TENSOR_ARENA:32768");
    Serial.println("----------------------------------");
    return;
  }

  printVector("===== RAW FEATURES =====", raw);

  normalizeFeatures(raw, norm);

  printVector("===== NORMALIZED FEATURES =====", norm);

  for (int i = 0; i < N_FEATURES; i++) {
    if (!modelSetInput(norm[i], i)) {
      Serial.print("Erreur modelSetInput indice ");
      Serial.println(i);
      return;
    }
  }

  // ================= MESURE AVANT INFERENCE =================

  float voltageBefore_V = ina219.getBusVoltage_V();
  float currentBefore_mA = ina219.getCurrent_mA();
  float powerBefore_mW = ina219.getPower_mW();

  Serial.println("Avant inference");

  // ================= INFERENCE CNN =================

  uint32_t t1 = micros();

  if (!modelRunInference()) {
    Serial.println("Erreur inference");
    Serial.println("PRED:INFERENCE_ERROR,CONF:0.0,TIME_US:0,VOLTAGE_V:0,CURRENT_MA:0,POWER_MW:0,ENERGY_J:0,ENERGY_MJ:0,HEAP_TOTAL:0,FREE_HEAP:0,USED_HEAP:0,MIN_FREE_HEAP:0,TENSOR_ARENA:32768");
    Serial.println("----------------------------------");
    return;
  }

  uint32_t t2 = micros();
  uint32_t inferenceUs = t2 - t1;

  Serial.println("Apres inference");

  // ================= MESURE APRES INFERENCE =================

  float voltageAfter_V = ina219.getBusVoltage_V();
  float currentAfter_mA = ina219.getCurrent_mA();
  float powerAfter_mW = ina219.getPower_mW();

  float powerAvg_mW = (powerBefore_mW + powerAfter_mW) / 2.0;
  float inference_s = inferenceUs / 1000000.0;
  float powerAvg_W = powerAvg_mW / 1000.0;

  float energy_J = powerAvg_W * inference_s;
  float energy_mJ = energy_J * 1000.0;

  for (int i = 0; i < N_CLASSES; i++) {
    outputs[i] = modelGetOutput(i);
  }

  int pred = argmax(outputs);

  // ================= CALCUL HEAP =================

  uint32_t heapTotal = ESP.getHeapSize();
  uint32_t freeHeap  = ESP.getFreeHeap();
  uint32_t usedHeap  = heapTotal - freeHeap;
  uint32_t minFreeHeap = ESP.getMinFreeHeap();

  printOutputs(outputs, pred);

  Serial.print("Inference time (us): ");
  Serial.println(inferenceUs);

  Serial.println("===== ENERGY MEASUREMENT INA219 =====");

  Serial.print("Voltage before (V): ");
  Serial.println(voltageBefore_V, 6);

  Serial.print("Current before (mA): ");
  Serial.println(currentBefore_mA, 6);

  Serial.print("Power before (mW): ");
  Serial.println(powerBefore_mW, 6);

  Serial.print("Voltage after (V): ");
  Serial.println(voltageAfter_V, 6);

  Serial.print("Current after (mA): ");
  Serial.println(currentAfter_mA, 6);

  Serial.print("Power after (mW): ");
  Serial.println(powerAfter_mW, 6);

  Serial.print("Average power (mW): ");
  Serial.println(powerAvg_mW, 6);

  Serial.print("Energy inference (J): ");
  Serial.println(energy_J, 9);

  Serial.print("Energy inference (mJ): ");
  Serial.println(energy_mJ, 6);

  Serial.println("===== MEMORY INFO =====");

  Serial.print("Heap total (bytes): ");
  Serial.println(heapTotal);

  Serial.print("Free heap (bytes): ");
  Serial.println(freeHeap);

  Serial.print("Used heap (bytes): ");
  Serial.println(usedHeap);

  Serial.print("Min free heap (bytes): ");
  Serial.println(minFreeHeap);

  Serial.print("TensorArena (bytes): ");
  Serial.println(kTensorArenaSize);

  Serial.println("----------------------------------");

  // Ligne finale facile à récupérer dans Kali Linux / CSV
  Serial.print("PRED:");
  Serial.print(CLASS_NAMES[pred]);

  Serial.print(",CONF:");
  Serial.print(outputs[pred], 6);

  Serial.print(",TIME_US:");
  Serial.print(inferenceUs);

  Serial.print(",VOLTAGE_V:");
  Serial.print(voltageAfter_V, 6);

  Serial.print(",CURRENT_MA:");
  Serial.print(currentAfter_mA, 6);

  Serial.print(",POWER_MW:");
  Serial.print(powerAvg_mW, 6);

  Serial.print(",ENERGY_J:");
  Serial.print(energy_J, 9);

  Serial.print(",ENERGY_MJ:");
  Serial.print(energy_mJ, 6);

  Serial.print(",HEAP_TOTAL:");
  Serial.print(heapTotal);

  Serial.print(",FREE_HEAP:");
  Serial.print(freeHeap);

  Serial.print(",USED_HEAP:");
  Serial.print(usedHeap);

  Serial.print(",MIN_FREE_HEAP:");
  Serial.print(minFreeHeap);

  Serial.print(",TENSOR_ARENA:");
  Serial.println(kTensorArenaSize);
}