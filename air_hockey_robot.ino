/*
  Air Hockey Robot — Arduino Motor Controller
  ============================================
  Receives angle commands from Raspberry Pi over Serial.
  Drives two stepper or servo motors for the 5-bar linkage.

  Protocol (from Pi):
    "T1:{angle1},T2:{angle2}\n"
    e.g. "T1:87.3,T2:102.1\n"

  Also accepts:
    "HOME\n"  — moves to home position (90, 90)

  ── WIRING (adjust pins to your hardware) ──────────────────────
  Servo mode:
    Motor 1 signal → PIN 9
    Motor 2 signal → PIN 10

  Stepper mode (TMC2209 / A4988):
    Motor 1 STEP → PIN 2,  DIR → PIN 3
    Motor 2 STEP → PIN 4,  DIR → PIN 5

  Set USE_SERVOS below to true (servo) or false (stepper)
  ──────────────────────────────────────────────────────────────
*/

#include <Servo.h>

// ── Configuration ──────────────────────────────────────────────
#define USE_SERVOS true         // true = servos, false = steppers

#define SERVO1_PIN  9
#define SERVO2_PIN  10
#define HOME_ANGLE1 90.0f
#define HOME_ANGLE2 90.0f

// Stepper pins (only used if USE_SERVOS = false)
#define STEP1_PIN   2
#define DIR1_PIN    3
#define STEP2_PIN   4
#define DIR2_PIN    5
#define STEPS_PER_REV  3200     // microsteps per revolution (adjust for driver setting)
#define MAX_STEP_SPEED 4000     // steps/sec

// Safety: clamp angles to this range
#define MIN_ANGLE   10.0f
#define MAX_ANGLE   170.0f
// ───────────────────────────────────────────────────────────────

Servo servo1, servo2;

float currentAngle1 = HOME_ANGLE1;
float currentAngle2 = HOME_ANGLE2;
float targetAngle1  = HOME_ANGLE1;
float targetAngle2  = HOME_ANGLE2;

String inputBuffer = "";

// ── Stepper state (if USE_SERVOS = false) ──────────────────────
long stepPos1 = 0, stepPos2 = 0;
long stepTarget1 = 0, stepTarget2 = 0;

float angleToSteps(float angle) {
  return (angle / 360.0f) * STEPS_PER_REV;
}

void stepMotors() {
  // Non-blocking single-step toward target for each motor
  if (stepPos1 != stepTarget1) {
    bool dir = (stepTarget1 > stepPos1);
    digitalWrite(DIR1_PIN, dir ? HIGH : LOW);
    digitalWrite(STEP1_PIN, HIGH);
    delayMicroseconds(50);
    digitalWrite(STEP1_PIN, LOW);
    stepPos1 += dir ? 1 : -1;
  }
  if (stepPos2 != stepTarget2) {
    bool dir = (stepTarget2 > stepPos2);
    digitalWrite(DIR2_PIN, dir ? HIGH : LOW);
    digitalWrite(STEP2_PIN, HIGH);
    delayMicroseconds(50);
    digitalWrite(STEP2_PIN, LOW);
    stepPos2 += dir ? 1 : -1;
  }
}

// ── Parse "T1:87.3,T2:102.1" ──────────────────────────────────
bool parseCommand(String cmd, float &a1, float &a2) {
  cmd.trim();
  if (cmd == "HOME") {
    a1 = HOME_ANGLE1;
    a2 = HOME_ANGLE2;
    return true;
  }
  int t1idx = cmd.indexOf("T1:");
  int t2idx = cmd.indexOf("T2:");
  if (t1idx < 0 || t2idx < 0) return false;

  int commaIdx = cmd.indexOf(',');
  if (commaIdx < 0) return false;

  a1 = cmd.substring(t1idx + 3, commaIdx).toFloat();
  a2 = cmd.substring(t2idx + 3).toFloat();

  // Clamp to safe range
  a1 = constrain(a1, MIN_ANGLE, MAX_ANGLE);
  a2 = constrain(a2, MIN_ANGLE, MAX_ANGLE);
  return true;
}

void setMotors(float a1, float a2) {
  if (USE_SERVOS) {
    servo1.write((int)a1);
    servo2.write((int)a2);
  } else {
    stepTarget1 = (long)angleToSteps(a1);
    stepTarget2 = (long)angleToSteps(a2);
  }
  currentAngle1 = a1;
  currentAngle2 = a2;
}

// ── Setup ──────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  inputBuffer.reserve(64);

  if (USE_SERVOS) {
    servo1.attach(SERVO1_PIN);
    servo2.attach(SERVO2_PIN);
    servo1.write((int)HOME_ANGLE1);
    servo2.write((int)HOME_ANGLE2);
  } else {
    pinMode(STEP1_PIN, OUTPUT);
    pinMode(DIR1_PIN,  OUTPUT);
    pinMode(STEP2_PIN, OUTPUT);
    pinMode(DIR2_PIN,  OUTPUT);
  }

  Serial.println("AirHockeyRobot READY");
}

// ── Main loop ─────────────────────────────────────────────────
void loop() {
  // Read incoming serial data character by character
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      // Process complete line
      float a1, a2;
      if (parseCommand(inputBuffer, a1, a2)) {
        setMotors(a1, a2);
        // Optional ack — comment out if it causes serial congestion
        // Serial.print("OK T1:"); Serial.print(a1); Serial.print(" T2:"); Serial.println(a2);
      }
      inputBuffer = "";
    } else {
      inputBuffer += c;
    }
  }

  // Run stepper motors toward target (non-blocking)
  if (!USE_SERVOS) {
    stepMotors();
  }
}
