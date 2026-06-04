#include <WiFi.h>
#include "esp_wifi.h"
#include "esp_wifi_types.h"
#include <vector>

#define CHANNEL         1
#define TARGET_MAC      {0x7C, 0xB0, 0x73, 0x2C, 0x6D, 0x7B}
#define WINDOW_MS       1000

uint8_t targetMAC[6] = TARGET_MAC;

volatile uint32_t cntTotal = 0, cntTarget = 0, bytesTotal = 0;
std::vector<uint64_t> seenMACs;
volatile uint32_t iatCount = 0;
volatile double iatMean = 0, iatM2 = 0;
volatile unsigned long lastTsMicros = 0;
portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;
unsigned long windowStart = 0;

bool isNewMAC(const uint8_t *mac) {
    uint64_t key = 0;
    for (int i = 0; i < 6; i++) key = (key << 😎 | mac[i];
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
    uint8_t *src = hdr + 10;
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

void setup() {
    Serial.begin(115200);
    delay(100);
    Serial.println("=== ESP32 Collecte ===");
    
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
}

void loop() {
    unsigned long now = millis();
    if (now - windowStart < WINDOW_MS) return;

    portENTER_CRITICAL(&mux);
    uint32_t T = cntTotal, M = cntTarget, B = bytesTotal;
    uint32_t U = seenMACs.size();
    uint32_t N = iatCount;
    double meanIAT = iatMean;
    double varIAT = (N > 1) ? (iatM2 / (N - 1)) : 0.0;
    
    cntTotal = 0; cntTarget = 0; bytesTotal = 0;
    iatCount = 0; iatMean = 0; iatM2 = 0;
    lastTsMicros = 0;
    seenMACs.clear();
    portEXIT_CRITICAL(&mux);

    float pktRatio = (T > 0) ? ((float)M / T) : 0.0f;
    float avgPktSize = (T > 0) ? ((float)B / T) : 0.0f;

    Serial.printf("%u,%u,%.3f,%.1f,%u,%.1f,%.1f\n", 
                  T, M, pktRatio, avgPktSize, U, meanIAT, varIAT);

    windowStart = now;
}
