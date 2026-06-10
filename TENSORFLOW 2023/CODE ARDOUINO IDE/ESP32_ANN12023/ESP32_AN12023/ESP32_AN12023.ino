#include <Arduino.h>
#include <ArduTFLite.h>
#include "ann_model_data.h"

// ================= CONFIG =================
const int N_FEATURES = 8;
const int N_CLASSES  = 4;

constexpr int kTensorArenaSize = 32 * 1024;
byte tensorArena[kTensorArenaSize];

// ================= NORMALISATION =================
// Depuis ann_normalization.json

float means[N_FEATURES] = {
  82946227.54495819f,     // IAT
  69268.14200088085f,     // Header_Length
  7.406373252679435f,     // Protocol Type
  2.5964629165718227f,    // flow_duration
  65.39542863461945f,     // rst_count
  84.20318805384402f,     // Tot size
  11.032745280479302f,    // Magnitue
  83.85474111781087f      // AVG
};

float scales[N_FEATURES] = {
  20847540.938972678f,    // IAT
  405467.3164622806f,     // Header_Length
  6.709516490165713f,     // Protocol Type
  26.477828227761f,       // flow_duration
  398.28310678053236f,    // rst_count
  228.0282374074317f,     // Tot size
  6.597566103763726f,     // Magnitue
  221.6471413772571f      // AVG
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

void normalizeFeatures(const float raw[N_FEATURES], float norm[N_FEATURES]) {
  for (int i = 0; i < N_FEATURES; i++) {
    if (scales[i] == 0.0f) {
      norm[i] = 0.0f;
    } else {
      norm[i] = (raw[i] - means[i]) / scales[i];
    }
  }
}

// ================= ARGMAX =================

int argmax(const float arr[], int n) {
  int idx = 0;
  float maxVal = arr[0];

  for (int i = 1; i < n; i++) {
    if (arr[i] > maxVal) {
      maxVal = arr[i];
      idx = i;
    }
  }

  return idx;
}

// ================= DEBUG =================

void printVector(const char* title, const float v[], int n) {
  Serial.println(title);

  for (int i = 0; i < n; i++) {
    Serial.print("  [");
    Serial.print(i);
    Serial.print("] = ");
    Serial.println(v[i], 6);
  }
}

void printOutputs(const float outputs[], int pred) {
  Serial.println("===== OUTPUTS =====");

  for (int i = 0; i < N_CLASSES; i++) {
    Serial.print(CLASS_NAMES[i]);
    Serial.print(" : ");
    Serial.println(outputs[i], 6);
  }

  Serial.println("===== RESULT =====");
  Serial.print("Class ID: ");
  Serial.println(pred);

  Serial.print("Class: ");
  Serial.println(CLASS_NAMES[pred]);

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
  uint32_t heap_total = ESP.getHeapSize();
  uint32_t free_heap  = ESP.getFreeHeap();
  uint32_t used_heap  = heap_total - free_heap;

  Serial.println("===== MEMORY INFO =====");

  Serial.print("Heap total (bytes): ");
  Serial.println(heap_total);

  Serial.print("Free heap (bytes): ");
  Serial.println(free_heap);

  Serial.print("Used heap (bytes): ");
  Serial.println(used_heap);

  Serial.print("TensorArena (bytes): ");
  Serial.println(kTensorArenaSize);

  Serial.println("=======================");
}

// ================= SETUP =================

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println();
  Serial.println("===== ESP32 ANN 4 CLASSES START =====");

  Serial.println("Memory before modelInit:");
  printMemoryInfo();

  if (!modelInit(ann_model_tflite, tensorArena, kTensorArenaSize)) {
    Serial.println("Erreur modelInit");
    Serial.println("Essaye d'augmenter kTensorArenaSize a 48*1024 ou 64*1024");

    while (1) {
      delay(1000);
    }
  }

  Serial.println("Model init OK");

  Serial.println("Memory after modelInit:");
  printMemoryInfo();

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
    Serial.println("FORMAT ERROR");
    Serial.println("PRED:PARSE_ERROR,CONF:0.0,TIME_US:0,HEAP_TOTAL:0,FREE_HEAP:0,USED_HEAP:0,TENSOR_ARENA:32768");
    Serial.println("----------------------------------");
    return;
  }

  normalizeFeatures(raw, norm);

  printVector("===== RAW INPUT =====", raw, N_FEATURES);
  printVector("===== NORMALIZED INPUT =====", norm, N_FEATURES);

  for (int i = 0; i < N_FEATURES; i++) {
    modelSetInput(norm[i], i);
  }

  uint32_t t0 = micros();

  if (!modelRunInference()) {
    Serial.println("INFERENCE ERROR");
    Serial.println("PRED:INFERENCE_ERROR,CONF:0.0,TIME_US:0,HEAP_TOTAL:0,FREE_HEAP:0,USED_HEAP:0,TENSOR_ARENA:32768");
    Serial.println("----------------------------------");
    return;
  }

  uint32_t t1 = micros();
  uint32_t inference_time_us = t1 - t0;

  for (int i = 0; i < N_CLASSES; i++) {
    outputs[i] = modelGetOutput(i);
  }

  int pred = argmax(outputs, N_CLASSES);

  // ================= CALCUL HEAP =================
  uint32_t heap_total = ESP.getHeapSize();
  uint32_t free_heap  = ESP.getFreeHeap();
  uint32_t used_heap  = heap_total - free_heap;

  printOutputs(outputs, pred);

  Serial.print("Inference time (us): ");
  Serial.println(inference_time_us);

  Serial.print("Heap total (bytes): ");
  Serial.println(heap_total);

  Serial.print("Free heap (bytes): ");
  Serial.println(free_heap);

  Serial.print("Used heap (bytes): ");
  Serial.println(used_heap);

  Serial.print("TensorArena (bytes): ");
  Serial.println(kTensorArenaSize);

  Serial.println("----------------------------------");

  // Ligne finale facile à récupérer dans Kali Linux / CSV
  Serial.print("PRED:");
  Serial.print(CLASS_NAMES[pred]);

  Serial.print(",CONF:");
  Serial.print(outputs[pred], 6);

  Serial.print(",TIME_US:");
  Serial.print(inference_time_us);

  Serial.print(",HEAP_TOTAL:");
  Serial.print(heap_total);

  Serial.print(",FREE_HEAP:");
  Serial.print(free_heap);

  Serial.print(",USED_HEAP:");
  Serial.print(used_heap);

  Serial.print(",TENSOR_ARENA:");
  Serial.println(kTensorArenaSize);
}