OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(2.3011989393424894) q[0];
ry(-1.4998293443236521) q[1];
cx q[0],q[1];
ry(-2.052348160130603) q[0];
ry(0.17560058897708597) q[1];
cx q[0],q[1];
ry(-0.547234027579) q[2];
ry(-0.21248818683341497) q[3];
cx q[2],q[3];
ry(1.1811277239732176) q[2];
ry(-2.600372237147544) q[3];
cx q[2],q[3];
ry(1.378292821510379) q[0];
ry(-3.03318924049315) q[2];
cx q[0],q[2];
ry(1.4987540487435462) q[0];
ry(1.4694354666672054) q[2];
cx q[0],q[2];
ry(2.9586742609791066) q[1];
ry(-0.02400023385905925) q[3];
cx q[1],q[3];
ry(1.7210617585652832) q[1];
ry(1.184049330460914) q[3];
cx q[1],q[3];
ry(-1.2739587687871574) q[0];
ry(0.44530146988174696) q[1];
cx q[0],q[1];
ry(-2.143808961495421) q[0];
ry(2.378822363144219) q[1];
cx q[0],q[1];
ry(0.8184963150636886) q[2];
ry(3.072831073568772) q[3];
cx q[2],q[3];
ry(-1.0792813925124274) q[2];
ry(1.4388810841241617) q[3];
cx q[2],q[3];
ry(0.3117042569806276) q[0];
ry(1.517097424597178) q[2];
cx q[0],q[2];
ry(-0.5758823268470072) q[0];
ry(-3.0291443372780336) q[2];
cx q[0],q[2];
ry(-2.9753330317832827) q[1];
ry(-1.0720232815353619) q[3];
cx q[1],q[3];
ry(1.8604849092465647) q[1];
ry(0.6591663258870993) q[3];
cx q[1],q[3];
ry(-0.964947203745222) q[0];
ry(0.9894373579020878) q[1];
cx q[0],q[1];
ry(1.538923871874271) q[0];
ry(0.9401784435734845) q[1];
cx q[0],q[1];
ry(-0.10838215272531238) q[2];
ry(0.6675355258301979) q[3];
cx q[2],q[3];
ry(1.4563100564897078) q[2];
ry(-2.9316138782825063) q[3];
cx q[2],q[3];
ry(3.0759380200357826) q[0];
ry(-0.27445719298285715) q[2];
cx q[0],q[2];
ry(-1.7012252563765013) q[0];
ry(2.357011191200361) q[2];
cx q[0],q[2];
ry(-1.8348218355241843) q[1];
ry(1.8653989171431569) q[3];
cx q[1],q[3];
ry(-0.8092314427207116) q[1];
ry(-0.5394554635001274) q[3];
cx q[1],q[3];
ry(0.6759752624729067) q[0];
ry(2.316899567918187) q[1];
ry(0.6912015028905348) q[2];
ry(-0.9945461300810381) q[3];