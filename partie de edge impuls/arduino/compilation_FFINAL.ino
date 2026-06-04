#include <WiFi.h>
#include "esp_wifi.h"
#include "esp_wifi_types.h"
#include <vector>
#include <boudjema_inferencing.h>

// ===== CONFIGURATION =====
#define CHANNEL         6
#define TARGET_MAC      {0x7C, 0xB0, 0x73, 0x2C, 0x6D, 0x7B}
#define WINDOW_MS       1000

uint8_t targetMAC[6] = TARGET_MAC;

// ===== VARIABLES SNIFF =====
volatile uint32_t cntTotal = 0, cntTarget = 0, bytesTotal = 0;
std::vector<uint64_t> seenMACs;
volatile uint32_t iatCount = 0;
volatile double iatMean = 0, iatM2 = 0;
volatile unsigned long lastTsMicros = 0;
portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;
unsigned long windowStart = 0;

// ===== 7 FEATURES =====
constexpr size_t kNumFeatures = 7;
static float features[kNumFeatures];

// ===== FONCTIONS SNIFF =====
bool isNewMAC(const uint8_t *mac) {
    uint64_t key = 0;
    for (int i = 0; i < 6; i++) key = (key << 8) | mac[i];
    for (auto &k : seenMACs) if (k == key) return false;
    seenMACs.push_back(key);
    return true;
}

void updateIAT(unsigned long delta) {
    iatCount++;
    double d = delta - iatMean;
    iatMean += d / iatCount;
    iatM2 += d * (delta - iatMean);
}

void sniffer_cb(void* buf, wifi_promiscuous_pkt_type_t type) {
    if (type != WIFI_PKT_DATA && type != WIFI_PKT_MGMT) return;
    auto pkt = (wifi_promiscuous_pkt_t*)buf;
    uint8_t *hdr = pkt->payload;
    uint16_t len = pkt->rx_ctrl.sig_len;
    uint8_t *src = hdr + 10;  // Adresse MAC source
    unsigned long now = micros();

    portENTER_CRITICAL_ISR(&mux);
    cntTotal++;
    bytesTotal += len;
    if (lastTsMicros) updateIAT(now - lastTsMicros);
    lastTsMicros = now;
    isNewMAC(src);

    bool match = true;
    for (int i = 0; i < 6; i++) if (src[i] != targetMAC[i]) { match = false; break; }
    if (match) cntTarget++;
    portEXIT_CRITICAL_ISR(&mux);
}

// ===== CALLBACK POUR L'INFÉRENCE =====
static int data_feed_cb(size_t offset, size_t length, float *out_ptr) {
    if (offset + length > kNumFeatures) return -1;
    for (size_t i = 0; i < length; i++) out_ptr[i] = features[offset + i];
    return 0;
}

// ===== SETUP =====
void setup() {
    Serial.begin(115200);
    delay(100);
    Serial.println("=== ESP32 IDS (7 features) ===");
    
    WiFi.mode(WIFI_STA);
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);
    esp_wifi_set_storage(WIFI_STORAGE_RAM);
    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_start();
    esp_wifi_set_channel(CHANNEL, WIFI_SECOND_CHAN_NONE);
    esp_wifi_set_promiscuous_rx_cb(&sniffer_cb);
    esp_wifi_set_promiscuous(true);
    
    windowStart = millis();
    Serial.println("Sniffer actif - detection (7 features)");
}

// ===== LOOP =====
void loop() {
    unsigned long now = millis();
    if (now - windowStart < WINDOW_MS) return;

    // Variables locales pour sortir de la section critique
    uint32_t T, M, B, U;
    double meanIAT = 0, varIAT = 0;

    portENTER_CRITICAL(&mux);
    T = cntTotal;
    M = cntTarget;
    B = bytesTotal;
    U = seenMACs.size();
    uint32_t N = iatCount;
    meanIAT = iatMean;
    varIAT = (N > 1) ? (iatM2 / (N - 1)) : 0.0;
    // Réinitialiser pour la prochaine fenêtre
    cntTotal = 0;
    cntTarget = 0;
    bytesTotal = 0;
    iatCount = 0;
    iatMean = 0;
    iatM2 = 0;
    lastTsMicros = 0;
    seenMACs.clear();
    portEXIT_CRITICAL(&mux);

    // Calcul des 7 caractéristiques
    float pktRatio = (T > 0) ? ((float)M / T) : 0.0f;
    float avgPktSize = (T > 0) ? ((float)B / T) : 0.0f;

    features[0] = (float)T;          // total_pkts
    features[1] = (float)M;          // target_pkts
    features[2] = pktRatio;          // pkt_ratio
    features[3] = avgPktSize;        // avg_pkt_size
    features[4] = (float)U;          // unique_MACs
    features[5] = (float)meanIAT;    // mean_IAT_us
    features[6] = (float)varIAT;     // IAT_variance

    // Inférence Edge Impulse
    signal_t signal;
    signal.total_length = kNumFeatures;
    signal.get_data = &data_feed_cb;
    ei_impulse_result_t result;
    EI_IMPULSE_ERROR res = run_classifier(&signal, &result, true);

    if (res != EI_IMPULSE_OK) {
        Serial.printf("ERR:%d\n", res);
    } else {
        size_t best_i = 0;
        float best_v = result.classification[0].value;
        for (size_t i = 1; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
            if (result.classification[i].value > best_v) {
                best_v = result.classification[i].value;
                best_i = i;
            }
        }
        Serial.printf("PRED:%s,%.3f\n", result.classification[best_i].label, best_v);
    }

    windowStart = now;
}
