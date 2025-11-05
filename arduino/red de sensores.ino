#include <OneWire.h>
#include <DallasTemperature.h>
#include "DFRobot_PH.h"
#include "DFRobot_EC.h"
#include <EEPROM.h>

// ==== DEFINICIÓN DE PINES ====
#define ONE_WIRE_BUS 2     // DS18B20 (temperatura)
#define TURBIDITY_PIN A1   // Sensor de turbidez
#define PH_PIN A2          // Sensor de pH
#define DO_PIN A3          // Sensor de oxígeno disuelto
//#define EC_PIN A4          // Sensor de conductividad

// ==== OBJETOS Y CONSTANTES ====
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);
DFRobot_PH ph;
//DFRobot_EC ec;

float temperature = 25.0;

// ==== Calibración pH ====
const float m_ph = 9.836;
const float b_ph = -11.91;

// ==== Calibración oxígeno disuelto ====
#define VREF 5000
#define ADC_RES 1024
#define CAL1_V 1245
#define CAL1_T 25.75

const float DO_Table[41] = {
  14.46,14.22,13.99,13.77,13.56,13.36,13.17,12.98,12.80,12.63,
  12.46,12.30,12.14,11.99,11.84,11.69,11.55,11.41,11.28,11.15,
  11.02,10.90,10.78,10.66,10.55,10.44,10.33,10.22,10.12,10.02,
  9.92,9.82,9.73,9.63,9.54,9.45,9.37,9.28,9.20,9.12,9.04
};

void setup() {
  Serial.begin(115200);
  sensors.begin();
  ph.begin();
  //ec.begin();
  delay(1000);
}

void loop() {
  static unsigned long t_prev = 0;
  if (millis() - t_prev >= 1000) {
    t_prev = millis();

    // === TEMPERATURA ===
    sensors.requestTemperatures();
    temperature = sensors.getTempCByIndex(0);
    Serial.print("\nTemperatura: ");
    Serial.print(temperature, 2);
    Serial.println(" °C");

    // === CONDUCTIVIDAD ===
    //int rawEC = analogRead(EC_PIN);
    //float voltageEC = rawEC * (5000.0 / 1024.0);
    //Serial.print("Conductividad - Voltaje: ");
    //Serial.print(voltageEC, 0);
    //if (voltageEC > 5) {
      //float ecValue = ec.readEC(voltageEC, temperature);
      //Serial.print(" mV | EC: ");
      //Serial.print(ecValue, 2);
      //Serial.println(" ms/cm");
    //} else {
      //Serial.println(" mV | EC: N/A (voltaje bajo)");
    //}
    //ec.calibration(voltageEC, temperature);

    // === pH ===
    float voltagePH = analogRead(PH_PIN) * (5.0 / 1024.0);
    float phValue = m_ph * voltagePH + b_ph;
    Serial.print("pH - Voltaje: ");
    Serial.print(voltagePH, 3);
    Serial.print(" V | pH: ");
    Serial.println(phValue, 2);

    // === OXÍGENO DISUELTO ===
    int adcDO = analogRead(DO_PIN);
    float voltageDO = adcDO * (float)VREF / ADC_RES;
    int tempIndex = constrain((int)temperature, 0, 40);
    float doValue = (voltageDO / CAL1_V) * DO_Table[tempIndex];
    Serial.print("Oxígeno Disuelto - Voltaje: ");
    Serial.print(voltageDO, 1);
    Serial.print(" mV | DO: ");
    Serial.print(doValue, 2);
    Serial.println(" mg/L");

    // === TURBIDEZ ===
    float voltageTurb = analogRead(TURBIDITY_PIN) * (5.0 / 1024.0);
    float NTU;
    if (voltageTurb >= 4.21) {
      NTU = 0.5;
    } else if (voltageTurb <= 3.27) {
      NTU = 500.0;
    } else if (voltageTurb > 4.10) {
      float m = (50.0 - 0.5) / (4.10 - 4.21);
      float b = 0.5 - m * 4.21;
      NTU = m * voltageTurb + b;
    } else {
      float m = (500.0 - 50.0) / (3.27 - 4.10);
      float b = 50.0 - m * 4.10;
      NTU = m * voltageTurb + b;
    }

    Serial.print("Turbidez - Voltaje: ");
    Serial.print(voltageTurb, 3);
    Serial.print(" V | NTU: ");
    Serial.print(NTU, 1);
    Serial.print(" → ");

    if (voltageTurb >= 4.21)
      Serial.println("Agua limpia.");
    else if (voltageTurb >= 4.10)
      Serial.println("Agua con partículas.");
    else if (voltageTurb >= 3.27)
      Serial.println("Agua muy turbia.");
    else
      Serial.println("Fuera del rango calibrado.");
  }
}
