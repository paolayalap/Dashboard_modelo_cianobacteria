# 🔧 Código Arduino — Red de Sensores

Esta carpeta contiene los **programas en Arduino** desarrollados para la adquisición de datos de los sensores utilizados en la **boya multisensorial** del proyecto.  
Cada script permite la lectura individual de un sensor, así como la integración de todos ellos en un sistema de medición conjunto.

---

## 📂 Archivos incluidos

| Archivo | Descripción |
|----------|-------------|
| `sensor_pH.ino` | Lectura y calibración del sensor de **pH**. |
| `sensor_turbidez.ino` | Lectura de la **turbidez del agua (NTU)**. |
| `sensor_oxigeno_disuelto.ino` | Medición de **oxígeno disuelto (mg/L)**. |
| `sensor_conductividad.ino` | Lectura de **conductividad eléctrica (µS/cm)**. |
| `red_de_sensores.ino` | Código que **integra los cinco sensores** para registrar los parámetros simultáneamente. |

---

## ⚠️ Observaciones

Durante las pruebas del código `red_de_sensores.ino`, se observó que **el sensor de conductividad no funcionaba correctamente** al ejecutarse junto con los demás sensores.  
Se recomienda probarlo **de forma individual o en combinación parcial**, ya que podría existir un conflicto de comunicación o interferencia en el canal analógico.

---

## ⚙️ Propósito

Estos programas permiten:
- Realizar la **adquisición y calibración de datos** de cada sensor.  
- Obtener mediciones experimentales utilizadas en los **dataframes de las pruebas piloto**.  
- Preparar la integración completa de la **boya multisensorial automatizada** con comunicación hacia la interfaz de predicción.

---
