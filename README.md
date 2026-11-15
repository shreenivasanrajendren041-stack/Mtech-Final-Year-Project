# Mtech-Final-Year-Project

# Identity-Aware Cross-Device Multi-Factor Authentication via Continuous Facial Biometrics and Smart Device Escalation

An advanced multi-factor authentication (MFA) and access control system built with Flask, PyTorch, and ArcFace embeddings. Features include unconstrained facial verification, anti-fraud liveness & mask-detection checks, role-based access control (RBAC), and persistent user management.

## Key Features
* **Biometric Authentication:** Powered by an ArcFace backbone and MTCNN facial alignment.
* **Anti-Fraud Mask Detection:** HSV lower-face skin-tone pixel ratio checks to prevent spoofing under unconstrained lighting.
* **Role-Based Access Control (RBAC):** Dedicated administrative controls for the website owner (`pers1`) to manage users securely.
* **Permanent User Deletion & Blocklists:** Securely wipes persistent disk data while adding blocked accounts to a permanent JSON registry.
