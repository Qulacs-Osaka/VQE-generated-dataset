OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-1.3942499386985165) q[0];
ry(0.9194243164263836) q[1];
cx q[0],q[1];
ry(-2.062459030768059) q[0];
ry(-0.44680828030195574) q[1];
cx q[0],q[1];
ry(2.5173135663627804) q[2];
ry(-2.074855264019221) q[3];
cx q[2],q[3];
ry(-1.0249836287502554) q[2];
ry(-0.9582276424898355) q[3];
cx q[2],q[3];
ry(-2.130217994232445) q[1];
ry(1.6131193276779947) q[2];
cx q[1],q[2];
ry(-0.40822832257266567) q[1];
ry(1.9196304251459542) q[2];
cx q[1],q[2];
ry(2.981246470983599) q[0];
ry(-1.774375749388051) q[1];
cx q[0],q[1];
ry(-0.9050546188078181) q[0];
ry(1.5289225564380136) q[1];
cx q[0],q[1];
ry(1.0233240196099682) q[2];
ry(-0.0711752823095777) q[3];
cx q[2],q[3];
ry(-1.5878948620831157) q[2];
ry(-2.2782390853376344) q[3];
cx q[2],q[3];
ry(-0.714601516819946) q[1];
ry(-2.20023178235499) q[2];
cx q[1],q[2];
ry(-1.7160724441678452) q[1];
ry(-0.2977652932186263) q[2];
cx q[1],q[2];
ry(1.496206747205691) q[0];
ry(-1.4928202254476945) q[1];
cx q[0],q[1];
ry(-2.287939029744533) q[0];
ry(-2.5938793070399844) q[1];
cx q[0],q[1];
ry(2.8357728484437366) q[2];
ry(2.4530454654582843) q[3];
cx q[2],q[3];
ry(3.0327068801535955) q[2];
ry(2.5364628615718092) q[3];
cx q[2],q[3];
ry(1.9432638181084148) q[1];
ry(0.9234579515487154) q[2];
cx q[1],q[2];
ry(-1.5909052311091072) q[1];
ry(1.530050149425631) q[2];
cx q[1],q[2];
ry(0.9697317618736534) q[0];
ry(1.6645402732043983) q[1];
cx q[0],q[1];
ry(-0.8997482843689683) q[0];
ry(-0.28624949418248935) q[1];
cx q[0],q[1];
ry(1.277031697349268) q[2];
ry(1.9184848720277845) q[3];
cx q[2],q[3];
ry(-1.3705975351876898) q[2];
ry(0.7944585858760744) q[3];
cx q[2],q[3];
ry(-2.8253132665059857) q[1];
ry(1.3290033282767497) q[2];
cx q[1],q[2];
ry(0.5166453477259614) q[1];
ry(-1.2389234366057955) q[2];
cx q[1],q[2];
ry(0.49668730427840563) q[0];
ry(0.6382881190899192) q[1];
ry(-0.7161176465766822) q[2];
ry(-2.223535284545865) q[3];