#include <Arduino.h>
#include <stdint.h>
#include <Wire.h>
#include <Adafruit_INA219.h>
#include <ArduTFLite.h>
#include <math.h>
#include "cnn_model_data.h"

// =========================================================
// ESP32 - CNN STABLE BINAIRE CICIDS2018
// Classes : 0 = NORMAL, 1 = ATTACK
// Sortie : sigmoid unique = probabilite ATTACK
// Energie : INA219, inference only
// =========================================================

const int N_FEATURES = 8;
const float THRESHOLD = 0.5f;

constexpr int kTensorArenaSize = 64 * 1024;
byte tensorArena[kTensorArenaSize];

// ================= INA219 =================
Adafruit_INA219 ina219;

// =========================================================
// NORMALISATION STANDARD SCALER
// =========================================================

float means[N_FEATURES] = {
  11426.32299983408f,
  483.9186195453791f,
  238.50767860423724f,
  45204.55583750888f,
  115634.06475713904f,
  325708.22096282896f,
  160.96377434745483f,
  680.4812078977933f
};

float scales[N_FEATURES] = {
  17075.737219914463f,
  301.54054726248233f,
  121.47205843352386f,
  1015317.8927028695f,
  1231045.1973182205f,
  602644.515605166f,
  46.90564382500144f,
  9122.483740050544f
};

// =========================================================
// CLASSES
// =========================================================

const char* CLASS_NORMAL = "NORMAL";
const char* CLASS_ATTACK = "ATTACK";

// =========================================================
// PARSER 8 VALEURS CSV
// =========================================================

bool parse8Floats(String line, float values[N_FEATURES]) {
  int start = 0;

  for (int i = 0; i < N_FEATURES; i++) {
    int commaIndex;

    if (i < N_FEATURES - 1) {
      commaIndex = line.indexOf(',', start);

      if (commaIndex == -1) {
        return false;
      }

      String token = line.substring(start, commaIndex);
      token.trim();

      if (token.length() == 0) {
        return false;
      }

      values[i] = token.toFloat();
      start = commaIndex + 1;
    } else {
      String token = line.substring(start);
      token.trim();

      if (token.length() == 0) {
        return false;
      }

      values[i] = token.toFloat();
    }
  }

  return true;
}

// =========================================================
// VERIFICATION RAW FEATURES
// =========================================================

bool checkRawFeatures(const float raw[N_FEATURES]) {
  for (int i = 0; i < N_FEATURES; i++) {
    if (isnan(raw[i]) || isinf(raw[i])) {
      Serial.print("ERREUR: RAW feature NaN/Inf indice ");
      Serial.println(i);
      return false;
    }
  }

  return true;
}

// =========================================================
// NORMALISATION SECURISEE
// =========================================================

bool normalizeFeatures(const float raw[N_FEATURES], float norm[N_FEATURES]) {
  for (int i = 0; i < N_FEATURES; i++) {
    if (isnan(raw[i]) || isinf(raw[i])) {
      Serial.print("ERREUR: raw feature NaN/Inf indice ");
      Serial.println(i);
      return false;
    }

    if (scales[i] == 0.0f || isnan(scales[i]) || isinf(scales[i])) {
      Serial.print("ERREUR: scale invalide indice ");
      Serial.println(i);
      return false;
    }

    norm[i] = (raw[i] - means[i]) / scales[i];

    if (isnan(norm[i]) || isinf(norm[i])) {
      Serial.print("ERREUR: normalized feature NaN/Inf indice ");
      Serial.println(i);
      return false;
    }
  }

  return true;
}

// =========================================================
// AFFICHAGE VECTEURS
// =========================================================

void printVector(const char* title, const float v[N_FEATURES]) {
  Serial.println(title);

  for (int i = 0; i < N_FEATURES; i++) {
    Serial.print("  [");
    Serial.print(i);
    Serial.print("] = ");
    Serial.println(v[i], 6);
  }
}

// =========================================================
// MEMOIRE
// =========================================================

void printMemoryInfo() {
  uint32_t heap_total = ESP.getHeapSize();
  uint32_t free_heap = ESP.getFreeHeap();
  uint32_t used_heap = heap_total - free_heap;
  uint32_t min_free_heap = ESP.getMinFreeHeap();

  Serial.println("===== MEMORY INFO =====");

  Serial.print("Heap total (bytes): ");
  Serial.println(heap_total);

  Serial.print("Free heap (bytes): ");
  Serial.println(free_heap);

  Serial.print("Used heap (bytes): ");
  Serial.println(used_heap);

  Serial.print("Min free heap (bytes): ");
  Serial.println(min_free_heap);

  Serial.print("TensorArena (bytes): ");
  Serial.println(kTensorArenaSize);

  Serial.println("=======================");
}

// =========================================================
// SETUP
// =========================================================

void setup() {
  Serial.begin(115200);
  delay(2000);

  // I2C ESP32 : SDA = GPIO21, SCL = GPIO22
  Wire.begin(21, 22);

  Serial.println();
  Serial.println("===== ESP32 CNN STABLE BINAIRE CICIDS2018 START =====");
  Serial.println("Modele : cnn_final_stable_seed42.tflite SANS optimisation");
  Serial.println("Header : cnn_model_data.h");
  Serial.println("Classes : 0=NORMAL, 1=ATTACK");
  Serial.println("Sortie modele : sigmoid unique = probabilite ATTACK");
  Serial.println("ENERGY_MEASUREMENT: INA219");
  Serial.println("ENERGY_MODE: INFERENCE_ONLY");

  Serial.print("Threshold : ");
  Serial.println(THRESHOLD, 2);

  if (!ina219.begin()) {
    Serial.println("ERREUR: INA219 non detecte !");
    Serial.println("Verifier VCC, GND, SDA, SCL");
    while (1) {
      delay(10);
    }
  }

  Serial.println("INA219 detecte avec succes !");

  Serial.println();
  Serial.println("Memory before modelInit:");
  printMemoryInfo();

  if (!modelInit(cnn_model_tflite, tensorArena, kTensorArenaSize)) {
    Serial.println("Erreur modelInit");
    Serial.println("Essaie kTensorArenaSize = 80 * 1024");

    while (1) {
      delay(1000);
    }
  }

  Serial.println("Model init OK");

  Serial.println();
  Serial.println("Memory after modelInit:");
  printMemoryInfo();

  Serial.println();
  Serial.println("Format attendu :");
  Serial.println("Init Bwd Win Bytes,Fwd Packet Length Max,Fwd Packet Length Std,Fwd IAT Min,Flow IAT Mean,Flow Bytes/s,Avg Packet Size,Total Fwd Packets");

  Serial.println();
  Serial.println("Exemple :");
  Serial.println("0,60,0,1000,2000,5000,60,10");
  Serial.println("----------------------------------");
}

// =========================================================
// LOOP
// =========================================================

void loop() {
  if (Serial.available() == 0) {
    return;
  }

  String line = Serial.readStringUntil('\n');
  line.trim();

  if (line.length() == 0) {
    return;
  }

  float raw[N_FEATURES];
  float norm[N_FEATURES];

  // =====================================================
  // 1) PARSE
  // =====================================================

  if (!parse8Floats(line, raw)) {
    Serial.println("FORMAT ERROR");
    Serial.println("Utilise : x1,x2,x3,x4,x5,x6,x7,x8");
    Serial.println("PRED:PARSE_ERROR,CONF:0.000000,PROB_ATTACK:0.000000,TIME_US:0,VOLTAGE_V:0,CURRENT_MA:0,POWER_MW:0,ENERGY_J:0,ENERGY_MJ:0,HEAP_TOTAL:0,FREE_HEAP:0,USED_HEAP:0,MIN_FREE_HEAP:0,TENSOR_ARENA:65536");
    Serial.println("----------------------------------");
    return;
  }

  printVector("===== RAW INPUT =====", raw);

  // =====================================================
  // 2) CHECK RAW
  // =====================================================

  if (!checkRawFeatures(raw)) {
    Serial.println("PRED:RAW_NAN,CONF:0.000000,PROB_ATTACK:0.000000,TIME_US:0,VOLTAGE_V:0,CURRENT_MA:0,POWER_MW:0,ENERGY_J:0,ENERGY_MJ:0,HEAP_TOTAL:0,FREE_HEAP:0,USED_HEAP:0,MIN_FREE_HEAP:0,TENSOR_ARENA:65536");
    Serial.println("----------------------------------");
    return;
  }

  // =====================================================
  // 3) NORMALISATION
  // =====================================================

  if (!normalizeFeatures(raw, norm)) {
    Serial.println("PRED:NORM_NAN,CONF:0.000000,PROB_ATTACK:0.000000,TIME_US:0,VOLTAGE_V:0,CURRENT_MA:0,POWER_MW:0,ENERGY_J:0,ENERGY_MJ:0,HEAP_TOTAL:0,FREE_HEAP:0,USED_HEAP:0,MIN_FREE_HEAP:0,TENSOR_ARENA:65536");
    Serial.println("----------------------------------");
    return;
  }

  printVector("===== NORMALIZED INPUT =====", norm);

  // =====================================================
  // 4) SET INPUT
  // =====================================================

  for (int i = 0; i < N_FEATURES; i++) {
    if (!modelSetInput(norm[i], i)) {
      Serial.print("Erreur modelSetInput indice ");
      Serial.println(i);
      Serial.println("PRED:SETINPUT_ERROR,CONF:0.000000,PROB_ATTACK:0.000000,TIME_US:0,VOLTAGE_V:0,CURRENT_MA:0,POWER_MW:0,ENERGY_J:0,ENERGY_MJ:0,HEAP_TOTAL:0,FREE_HEAP:0,USED_HEAP:0,MIN_FREE_HEAP:0,TENSOR_ARENA:65536");
      Serial.println("----------------------------------");
      return;
    }
  }

  // =====================================================
  // 5) MESURE AVANT INFERENCE
  // =====================================================

  float voltageBefore_V = ina219.getBusVoltage_V();
  float currentBefore_mA = ina219.getCurrent_mA();
  float powerBefore_mW = ina219.getPower_mW();

  // =====================================================
  // 6) INFERENCE CNN
  // =====================================================

  uint32_t t0 = micros();

  if (!modelRunInference()) {
    Serial.println("INFERENCE ERROR");
    Serial.println("PRED:INFERENCE_ERROR,CONF:0.000000,PROB_ATTACK:0.000000,TIME_US:0,VOLTAGE_V:0,CURRENT_MA:0,POWER_MW:0,ENERGY_J:0,ENERGY_MJ:0,HEAP_TOTAL:0,FREE_HEAP:0,USED_HEAP:0,MIN_FREE_HEAP:0,TENSOR_ARENA:65536");
    Serial.println("----------------------------------");
    return;
  }

  uint32_t t1 = micros();
  uint32_t inference_time_us = t1 - t0;

  // =====================================================
  // 7) MESURE APRES INFERENCE
  // =====================================================

  float voltageAfter_V = ina219.getBusVoltage_V();
  float currentAfter_mA = ina219.getCurrent_mA();
  float powerAfter_mW = ina219.getPower_mW();

  float powerAvg_mW = (powerBefore_mW + powerAfter_mW) / 2.0;
  float inference_s = inference_time_us / 1000000.0;
  float powerAvg_W = powerAvg_mW / 1000.0;

  float energy_J = powerAvg_W * inference_s;
  float energy_mJ = energy_J * 1000.0;

  // =====================================================
  // 8) OUTPUT
  // =====================================================

  float prob_attack = modelGetOutput(0);

  if (isnan(prob_attack) || isinf(prob_attack)) {
    Serial.println("ERREUR: sortie modele NaN ou Inf");
    Serial.print("PRED:OUTPUT_NAN,CONF:0.000000,PROB_ATTACK:nan,TIME_US:");
    Serial.print(inference_time_us);
    Serial.println(",VOLTAGE_V:0,CURRENT_MA:0,POWER_MW:0,ENERGY_J:0,ENERGY_MJ:0,HEAP_TOTAL:0,FREE_HEAP:0,USED_HEAP:0,MIN_FREE_HEAP:0,TENSOR_ARENA:65536");
    Serial.println("----------------------------------");
    return;
  }

  if (prob_attack < 0.0f) {
    prob_attack = 0.0f;
  }

  if (prob_attack > 1.0f) {
    prob_attack = 1.0f;
  }

  int pred_id;
  const char* pred_label;
  float confidence;

  if (prob_attack >= THRESHOLD) {
    pred_id = 1;
    pred_label = CLASS_ATTACK;
    confidence = prob_attack;
  } else {
    pred_id = 0;
    pred_label = CLASS_NORMAL;
    confidence = 1.0f - prob_attack;
  }

  // =====================================================
  // 9) CALCUL HEAP
  // =====================================================

  uint32_t heap_total = ESP.getHeapSize();
  uint32_t free_heap = ESP.getFreeHeap();
  uint32_t used_heap = heap_total - free_heap;
  uint32_t min_free_heap = ESP.getMinFreeHeap();

  // =====================================================
  // 10) AFFICHAGE
  // =====================================================

  Serial.println("===== OUTPUT =====");

  Serial.print("Prob ATTACK : ");
  Serial.println(prob_attack, 6);

  Serial.println("===== RESULT =====");

  Serial.print("Class ID   : ");
  Serial.println(pred_id);

  Serial.print("Class      : ");
  Serial.println(pred_label);

  Serial.print("Confidence : ");
  Serial.println(confidence, 6);

  Serial.print("Inference time us : ");
  Serial.println(inference_time_us);

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

  Serial.print("Heap total bytes  : ");
  Serial.println(heap_total);

  Serial.print("Free heap bytes   : ");
  Serial.println(free_heap);

  Serial.print("Used heap bytes   : ");
  Serial.println(used_heap);

  Serial.print("Min free heap bytes : ");
  Serial.println(min_free_heap);

  Serial.print("TensorArena bytes : ");
  Serial.println(kTensorArenaSize);

  if (pred_id == 1) {
    Serial.println("=> ATTACK DETECTED");
  } else {
    Serial.println("=> NORMAL TRAFFIC");
  }

  Serial.println("----------------------------------");

  // =====================================================
  // 11) LIGNE PARSEABLE PAR PYTHON KALI
  // =====================================================

  Serial.print("PRED:");
  Serial.print(pred_label);

  Serial.print(",CONF:");
  Serial.print(confidence, 6);

  Serial.print(",PROB_ATTACK:");
  Serial.print(prob_attack, 6);

  Serial.print(",TIME_US:");
  Serial.print(inference_time_us);

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
  Serial.print(heap_total);

  Serial.print(",FREE_HEAP:");
  Serial.print(free_heap);

  Serial.print(",USED_HEAP:");
  Serial.print(used_heap);

  Serial.print(",MIN_FREE_HEAP:");
  Serial.print(min_free_heap);

  Serial.print(",TENSOR_ARENA:");
  Serial.println(kTensorArenaSize);
}