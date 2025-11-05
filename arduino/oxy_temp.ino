#include <OneWire.h>
#include <DallasTemperature.h>

// ==== CONFIGURACIÓN PARA SENSOR DE TEMPERATURA DS18B20 ====
#define ONE_WIRE_BUS 2 // Pin digital para el DS18B20
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

// ==== CONFIGURACIÓN PARA SENSOR DE OXÍGENO DISUELTO ====
#define DO_PIN A3     // ← AQUÍ se usa A3 en el Arduino MEGA
#define VREF 5000     // Voltaje de referencia en mV
#define ADC_RES 1024  // Resolución del ADC del Arduino MEGA

// Calibración de un solo punto
//#define CAL1_V 1600   // ← Reemplaza este valor con el que obtuviste al calibrar al aire
//#define CAL1_T 25     // ← Temperatura ambiente durante la calibración

#define CAL1_V 1245
#define CAL1_T 25.75
//oxigeno disuelto de calibración ideal 8.12 mg/L

// Tabla de oxígeno disuelto (mg/L) a saturación 100%, desde 0°C hasta 40°C
const float DO_Table[41] = {
  14.46,14.22,13.99,13.77,13.56,13.36,13.17,12.98,12.80,12.63,
  12.46,12.30,12.14,11.99,11.84,11.69,11.55,11.41,11.28,11.15,
  11.02,10.90,10.78,10.66,10.55,10.44,10.33,10.22,10.12,10.02,
  9.92,9.82,9.73,9.63,9.54,9.45,9.37,9.28,9.20,9.12,9.04
};

void setup() {
  Serial.begin(115200);
  sensors.begin(); // Inicializa el sensor de temperatura
}

void loop() {
  // === 1. Leer temperatura del DS18B20 ===
  sensors.requestTemperatures();
  float temperature = sensors.getTempCByIndex(0);

  // Limita el índice a valores válidos de 0 a 40
  int tempIndex = constrain((int)temperature, 0, 40);

  // === 2. Leer voltaje del sensor de oxígeno disuelto ===
  int adcValue = analogRead(DO_PIN);
  float voltage = (float)adcValue / ADC_RES * VREF;

  // === 3. Calcular oxígeno disuelto compensado por temperatura ===
  float doValue = (voltage / CAL1_V) * DO_Table[tempIndex];

  // === 4. Mostrar resultados ===
  Serial.print("Temperatura: ");
  Serial.print(temperature, 2);
  Serial.print(" °C\tVoltaje: ");
  Serial.print(voltage, 2);
  Serial.print(" mV\tOxígeno disuelto: ");
  Serial.print(doValue, 2);
  Serial.println(" mg/L");

  delay(1000);
}
