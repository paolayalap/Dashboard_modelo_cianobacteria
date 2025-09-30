//CÓDIGO SEGURO PARA VER QUE EL SENSOR ESTÉ FUNCIONANDO CORRECTAMENTE
//void setup() {
//  Serial.begin(115200);
//}


//CÓDIGO PARA BORRAR LA EEPROM
//void loop() {
//  int raw = analogRead(A4);
//  float voltage = raw * (5.0 / 1024.0);

//  Serial.print("Raw analog: ");
//  Serial.print(raw);
//  Serial.print(" | Voltage: ");
//  Serial.print(voltage * 1000); // en milivoltios
//  Serial.println(" mV");

//  delay(1000);
//}

//#include <EEPROM.h>

//void setup() {
//  for (int i = 0; i < EEPROM.length(); i++) {
//    EEPROM.write(i, 0);
//  }
//  Serial.begin(115200);
//  Serial.println("EEPROM limpiada. Ahora vuelve a cargar el código principal.");
//}

//void loop() {}

#include "DFRobot_EC.h"
#include <EEPROM.h>
#include <OneWire.h>
#include <DallasTemperature.h>

#define EC_PIN A4
#define ONE_WIRE_BUS 2

OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);
DFRobot_EC ec;

float voltage, ecValue, temperature = 25.0;

// Cambia a true SOLO cuando quieras calibrar manualmente
bool modo_calibracion = true;

void setup() {
  Serial.begin(115200);
  sensors.begin();  
  ec.begin();
  delay(1000);
}

void loop() {
  static unsigned long timepoint = millis();
  if (millis() - timepoint > 1000U) {
    timepoint = millis();

    // Leer temperatura
    sensors.requestTemperatures();
    temperature = sensors.getTempCByIndex(0);
    if (temperature < -50 || temperature > 125 || isnan(temperature)) {
      Serial.println("Error: Temperatura inválida");
      return;
    }

    // Leer voltaje
    int raw = analogRead(EC_PIN);
    voltage = raw / 1024.0 * 5000;

    Serial.print("Raw analog: ");
    Serial.print(raw);
    Serial.print(" | Voltage: ");
    Serial.print(voltage);
    Serial.print(" mV | Temp: ");
    Serial.print(temperature, 1);
    Serial.print(" °C");

    // Calcular EC
    if (voltage > 5) {
      ecValue = ec.readEC(voltage, temperature);
      Serial.print(" | EC: ");
      Serial.print(ecValue, 2);
      Serial.println(" ms/cm");
    } else {
      Serial.println(" | EC: N/A (voltaje bajo)");
    }
  }

  // Solo permite calibración si tú activas la bandera
  if (modo_calibracion) {
    ec.calibration(voltage, temperature);
  }
}
