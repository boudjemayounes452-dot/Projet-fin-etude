#include <Arduino.h>
#include <stdint.h>
#include <Wire.h>
#include <Adafruit_INA219.h>
#include "rf_model_esp32.h"

// ================= CONFIG =================
const int N_FEATURES = 8;
const int N_CLASSES = 4;

const char* CLASS_NAMES[N_CLASSES] = {
  "BenignTraffic",
  "DDoS-ICMP_Flood",
  "DDoS-SYN_Flood",
  "DDoS-UDP_Flood"
};

// ================= INA219 =================
Adafruit_INA219 ina219;

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

// ================= CONVERSION FLOAT -> INT16 =================

int16_t safeFloatToInt16(float x) {
  long v = lround(x);

  if (v > 32767) v = 32767;
  if (v < -32768) v = -32768;

  return (int16_t)v;
}

// ================= DEBUG =================

void printRawFeatures(float raw[N_FEATURES]) {
  Serial.println("===== RAW INPUT =====");
  for (int i = 0; i < N_FEATURES; i++) {
    Serial.print("x");
    Serial.print(i + 1);
    Serial.print(" = ");
    Serial.println(raw[i], 6);
  }
}

void printIntFeatures(int16_t values[N_FEATURES]) {
  Serial.println("===== INT16 INPUT =====");
  for (int i = 0; i < N_FEATURES; i++) {
    Serial.print("z");
    Serial.print(i + 1);
    Serial.print(" = ");
    Serial.println(values[i]);
  }
}

// ================= SETUP =================

void setup() {
  Serial.begin(115200);
  delay(2000);

  // I2C ESP32 : SDA = GPIO 21, SCL = GPIO 22
  Wire.begin(21, 22);

  Serial.println("==================================");
  Serial.println("ESP32 RANDOM FOREST 4 CLASSES");
  Serial.println("MODEL_TAG: RF_4CLASSES_8FEATURES");
  Serial.println("ENERGY_MEASUREMENT: INA219");
  Serial.println("ENERGY_MODE: INFERENCE_ONLY");
  Serial.println("==================================");

  // Initialisation INA219
  if (!ina219.begin()) {
    Serial.println("ERREUR: INA219 non detecte !");
    Serial.println("Verifier VCC, GND, SDA, SCL");
    while (1) {
      delay(10);
    }
  }

  Serial.println("INA219 detecte avec succes !");

  Serial.println("Format attendu :");
  Serial.println("IAT,Header_Length,Protocol Type,flow_duration,rst_count,Tot size,Magnitue,AVG");
  Serial.println("Exemple :");
  Serial.println("655.569885,28,1,4.970551,0,50.871651,7.132436,50.871651");
  Serial.println("----------------------------------");

  Serial.print("HEAP_TOTAL_START:");
  Serial.println(ESP.getHeapSize());

  Serial.print("FREE_HEAP_START:");
  Serial.println(ESP.getFreeHeap());

  Serial.println("----------------------------------");
}

// ================= LOOP =================

void loop() {
  // Si Kali n'envoie rien, ESP32 ne fait aucune mesure
  if (Serial.available() == 0) return;

  String line = Serial.readStringUntil('\n');
  line.trim();

  if (line.length() == 0) return;

  float raw[N_FEATURES];
  int16_t features[N_FEATURES];

  if (!parse8Floats(line, raw)) {
    Serial.println("FORMAT ERROR");
    Serial.println("PRED:PARSE_ERROR,CONF:0.0,TIME_US:0");
    Serial.println("----------------------------------");
    return;
  }

  printRawFeatures(raw);

  for (int i = 0; i < N_FEATURES; i++) {
    features[i] = safeFloatToInt16(raw[i]);
  }

  printIntFeatures(features);

  // ================= MESURE AVANT INFERENCE =================

  float voltageBefore_V = ina219.getBusVoltage_V();
  float currentBefore_mA = ina219.getCurrent_mA();
  float powerBefore_mW = ina219.getPower_mW();

  // ================= INFERENCE IA =================

  unsigned long t0 = micros();

  int pred = rf_model_predict(features, N_FEATURES);

  unsigned long t1 = micros();
  unsigned long inferenceUs = t1 - t0;

  // ================= MESURE APRES INFERENCE =================

  float voltageAfter_V = ina219.getBusVoltage_V();
  float currentAfter_mA = ina219.getCurrent_mA();
  float powerAfter_mW = ina219.getPower_mW();

  // Puissance moyenne pendant l'inférence
  float powerAvg_mW = (powerBefore_mW + powerAfter_mW) / 2.0;

  // Temps d'inférence en secondes
  float inference_s = inferenceUs / 1000000.0;

  // Conversion puissance mW vers W
  float powerAvg_W = powerAvg_mW / 1000.0;

  // Energie en Joule : E = P x t
  float energy_J = powerAvg_W * inference_s;

  // Energie en mJ
  float energy_mJ = energy_J * 1000.0;

  if (pred < 0 || pred >= N_CLASSES) {
    Serial.println("RF INFERENCE ERROR");
    Serial.println("PRED:INFERENCE_ERROR,CONF:0.0,TIME_US:0");
    Serial.println("----------------------------------");
    return;
  }

  uint32_t total_heap = ESP.getHeapSize();
  uint32_t free_heap = ESP.getFreeHeap();
  uint32_t used_heap = total_heap - free_heap;

  // ================= AFFICHAGE RESULTATS =================

  Serial.println("===== RESULT =====");
  Serial.print("Class ID: ");
  Serial.println(pred);

  Serial.print("Class: ");
  Serial.println(CLASS_NAMES[pred]);

  if (pred == 0) {
    Serial.println("=> NORMAL TRAFFIC");
  } else {
    Serial.print("=> ATTACK DETECTED: ");
    Serial.println(CLASS_NAMES[pred]);
  }

  Serial.print("Inference time (us): ");
  Serial.println(inferenceUs);

  Serial.print("Inference time (s): ");
  Serial.println(inference_s, 9);

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

  Serial.println("===== MEMORY =====");

  Serial.print("Heap total (bytes): ");
  Serial.println(total_heap);

  Serial.print("Free heap (bytes): ");
  Serial.println(free_heap);

  Serial.print("Used heap (bytes): ");
  Serial.println(used_heap);

  Serial.println("----------------------------------");

  // Ligne compacte pour Kali / CSV
  Serial.print("PRED:");
  Serial.print(CLASS_NAMES[pred]);

  Serial.print(",CONF:");
  Serial.print("1.000000");

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
  Serial.print(total_heap);

  Serial.print(",FREE_HEAP:");
  Serial.print(free_heap);

  Serial.print(",USED_HEAP:");
  Serial.println(used_heap);
}