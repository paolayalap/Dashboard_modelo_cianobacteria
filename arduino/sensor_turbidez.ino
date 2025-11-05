void setup() {
  Serial.begin(115200); // Comunicación serial a 115200 baudios
}

void loop() {
  int sensorValue = analogRead(A1); // Leer señal del sensor
  float voltage = sensorValue * (5.0 / 1024.0); // Convertir a voltaje (0–5V)
  float NTU;

  // Interpolación basada en 3 puntos calibrados:
  // (4.21V, 0.5 NTU), (4.10V, 50 NTU), (3.27V, 500 NTU)
  if (voltage >= 4.21) {
    NTU = 0.5;
  } else if (voltage <= 3.27) {
    NTU = 500.0;
  } else if (voltage > 4.10) {
    // Interpolar entre 4.21V y 4.10V
    float m = (50.0 - 0.5) / (4.10 - 4.21);
    float b = 0.5 - m * 4.21;
    NTU = m * voltage + b;
  } else {
    // Interpolar entre 4.10V y 3.27V
    float m = (500.0 - 50.0) / (3.27 - 4.10);
    float b = 50.0 - m * 4.10;
    NTU = m * voltage + b;
  }

  // Imprimir resultados
  Serial.print("Voltaje: ");
  Serial.print(voltage, 3);
  Serial.print(" V | NTU: ");
  Serial.print(NTU, 1);
  Serial.print(" -> ");

  // Clasificación por voltaje (como pediste)
  if (voltage >= 4.21) {
    Serial.println("Agua limpia (muy pocas partículas).");
  } else if (voltage >= 4.10) {
    Serial.println("Agua con presencia de partículas.");
  } else if (voltage >= 3.27) {
    Serial.println("Agua con muchas partículas.");
  } else {
    Serial.println("Lectura fuera del rango calibrado.");
  }

  delay(500);
}

