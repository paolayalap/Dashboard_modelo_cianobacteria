/* =========================================================
   Mega 2560 + pH SEN0161-V2 (A2) + DS18B20 (D2)
   Librería: DFRobot_PH (calibración/EEPROM incluidas)
   Comandos Serial (115200):
     enterph  -> entrar a modo calibración
     calph    -> calibrar en buffer actual (auto 7.00 / 4.00)
     exitph   -> guardar en EEPROM y salir
   ========================================================= */

#include "DFRobot_PH.h"
#include <EEPROM.h>
#include <OneWire.h>
#include <DallasTemperature.h>

#define PH_PIN A2         // <--- pH en A2
#define ONE_WIRE_BUS 2    // <--- DS18B20 en D2

DFRobot_PH ph;
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature ds18b20(&oneWire);

float temperatureC = 25.0;
float phValue = 7.0;

// Promedia lecturas y devuelve mV
float leerVoltajePH_mV(uint8_t muestras = 10) {
  long acum = 0;
  for (uint8_t i = 0; i < muestras; i++) {
    acum += analogRead(PH_PIN);   // Mega: 10 bits, 0..1023
    delay(10);
  }
  float adcProm = (float)acum / muestras;
  // Referencia por defecto: 5.00 V → 5000 mV
  float mV = adcProm * (5000.0 / 1023.0);
  return mV;
}

float leerTemperaturaC() {
  ds18b20.requestTemperatures();
  float t = ds18b20.getTempCByIndex(0);
  if (t > -55 && t < 125) return t;
  return 25.0; // respaldo si no responde
}

void setup() {
  Serial.begin(115200);
  delay(800);
  ds18b20.begin();
  ph.begin();   // carga calibración previa desde EEPROM (si existe)
  Serial.println(F("Listo. Use: enterph, calph, exitph"));
}

void loop() {
  static unsigned long t0 = millis();
  if (millis() - t0 >= 1000) { // cada 1 s
    t0 = millis();
    temperatureC = leerTemperaturaC();
    float mV = leerVoltajePH_mV(10);
    phValue = ph.readPH(mV, temperatureC); // compensación por T

    Serial.print(F("T[°C]=")); Serial.print(temperatureC, 2);
    Serial.print(F(" | Vph[mV]=")); Serial.print(mV, 1);
    Serial.print(F(" | pH=")); Serial.println(phValue, 2);
  }

  // Manejo de comandos de calibración: enterph, calph, exitph
  float mVinst = leerVoltajePH_mV(5);
  ph.calibration(mVinst, temperatureC);
}
