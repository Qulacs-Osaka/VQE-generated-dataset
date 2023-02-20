OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.08784849953431711) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
h q[0];
h q[6];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(0.6996930696300896) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[6];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.005792561863843847) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
sdg q[0];
h q[0];
sdg q[6];
h q[6];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(0.8384285765748403) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[6];
s q[6];
cx q[0],q[1];
rz(-0.27991355451266) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.5546588404868473) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
h q[1];
h q[7];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.08676801859063708) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[7];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.6841857037122793) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
sdg q[1];
h q[1];
sdg q[7];
h q[7];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(-1.8844418489871848) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[7];
s q[7];
h q[2];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(-2.0213934965098015) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[4];
h q[2];
h q[8];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.021490774330019696) q[8];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[8];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(0.4864420494733584) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
sdg q[2];
h q[2];
sdg q[8];
h q[8];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
rz(0.21374853678737865) q[8];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[8];
s q[8];
cx q[2],q[3];
rz(0.017276557183362146) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(0.3725195680159375) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[5];
h q[3];
h q[9];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
rz(-1.5441234971929778) q[9];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[9];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(1.7465584263053675) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[5];
s q[5];
sdg q[3];
h q[3];
sdg q[9];
h q[9];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
rz(-1.6796226334594624) q[9];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[9];
s q[9];
h q[4];
h q[10];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
rz(0.22958603285298243) q[10];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[10];
sdg q[4];
h q[4];
sdg q[10];
h q[10];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.032069842030012057) q[10];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[10];
s q[10];
cx q[4],q[5];
rz(-0.030648038783432693) q[5];
cx q[4],q[5];
h q[5];
h q[11];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
rz(1.5893762822778463) q[11];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[11];
sdg q[5];
h q[5];
sdg q[11];
h q[11];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
rz(0.8173326609128557) q[11];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[11];
s q[11];
h q[6];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.1264640501487739) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
h q[8];
sdg q[6];
h q[6];
sdg q[8];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(0.12796162673390496) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(-0.14030701822034788) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-1.6107975795924463) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
h q[9];
sdg q[7];
h q[7];
sdg q[9];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-1.0931036276721133) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
s q[7];
h q[9];
s q[9];
h q[8];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(0.6671937121637366) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
h q[10];
sdg q[8];
h q[8];
sdg q[10];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(0.07253734770674679) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(-0.23913372083301573) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(0.20377786321491606) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
h q[11];
sdg q[9];
h q[9];
sdg q[11];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(0.25424861244056346) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.030270663982344837) q[11];
cx q[10],q[11];
rx(-1.5764204937298056) q[0];
rz(0.00029048111906731914) q[0];
rx(-0.018006503725625084) q[1];
rz(0.08611802764606019) q[1];
rx(-1.6009571559145799) q[2];
rz(-0.0010091610849480917) q[2];
rx(-0.004585865900642751) q[3];
rz(0.02165289023186147) q[3];
rx(0.004486642605020388) q[4];
rz(0.3712381742461695) q[4];
rx(-0.004499557094724861) q[5];
rz(0.2909264024220054) q[5];
rx(-0.4555115955596856) q[6];
rz(0.0004317322124897623) q[6];
rx(0.0005334657892680414) q[7];
rz(0.04799311666772477) q[7];
rx(-0.00013132778651379827) q[8];
rz(-0.0026917942029363547) q[8];
rx(0.00031966075210249664) q[9];
rz(-2.879796142011434) q[9];
rx(2.391860863651171e-05) q[10];
rz(-0.2441819718358948) q[10];
rx(8.654261688929893e-05) q[11];
rz(0.17310086455861817) q[11];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.062192559084266026) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
h q[0];
h q[6];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(5.5346766107337544e-05) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[6];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.0949174326886885) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
sdg q[0];
h q[0];
sdg q[6];
h q[6];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.0030349264572068825) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[6];
s q[6];
cx q[0],q[1];
rz(0.0005195166393552235) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-4.067012608035222e-05) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
h q[1];
h q[7];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(5.110801478586335e-05) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[7];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-1.206160586126386e-05) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
sdg q[1];
h q[1];
sdg q[7];
h q[7];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(-3.3298601511749717e-05) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[7];
s q[7];
h q[2];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.07906013056706294) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[4];
h q[2];
h q[8];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
rz(0.00018324809503891484) q[8];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[8];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(0.0012772621169753267) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
sdg q[2];
h q[2];
sdg q[8];
h q[8];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
rz(8.659359188295393e-05) q[8];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[8];
s q[8];
cx q[2],q[3];
rz(0.0012881002366255754) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.03775590422928152) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[5];
h q[3];
h q[9];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.0001133472785868361) q[9];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[9];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.08449299215650162) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[5];
s q[5];
sdg q[3];
h q[3];
sdg q[9];
h q[9];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
rz(0.00036929316445499213) q[9];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[9];
s q[9];
h q[4];
h q[10];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
rz(0.0004948040007031708) q[10];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[10];
sdg q[4];
h q[4];
sdg q[10];
h q[10];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.0010724641824830575) q[10];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[10];
s q[10];
cx q[4],q[5];
rz(0.03165108507439173) q[5];
cx q[4],q[5];
h q[5];
h q[11];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
rz(0.00037133939361067367) q[11];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[11];
sdg q[5];
h q[5];
sdg q[11];
h q[11];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
rz(0.0008200348750734883) q[11];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[11];
s q[11];
h q[6];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(-2.9567289108410613) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
h q[8];
sdg q[6];
h q[6];
sdg q[8];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(0.0003082310854269684) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(0.0028919122125703562) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-2.5519996001753507) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
h q[9];
sdg q[7];
h q[7];
sdg q[9];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.29835748524740946) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
s q[7];
h q[9];
s q[9];
h q[8];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(0.07787316270356329) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
h q[10];
sdg q[8];
h q[8];
sdg q[10];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(0.048353570126782625) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(0.011381192284833157) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-1.2499117281102794) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
h q[11];
sdg q[9];
h q[9];
sdg q[11];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(1.2749288363371185) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.031436054064319575) q[11];
cx q[10],q[11];
rx(-1.5633229083220903) q[0];
rz(-0.19129781101278617) q[0];
rx(0.017889995200158285) q[1];
rz(-0.28713656921825065) q[1];
rx(-1.54155078593102) q[2];
rz(0.0015051537028305088) q[2];
rx(0.004733047482251189) q[3];
rz(0.06463750203893429) q[3];
rx(-0.004393660857531599) q[4];
rz(-0.2609070224806032) q[4];
rx(-3.1370712389523696) q[5];
rz(0.2874484024047853) q[5];
rx(-2.6860745802110406) q[6];
rz(-0.03478284083412207) q[6];
rx(1.7131943761286063e-05) q[7];
rz(-1.5441262820737574) q[7];
rx(-1.463624792153622e-05) q[8];
rz(0.03569882260459695) q[8];
rx(-3.141301975001212) q[9];
rz(-0.218635698654895) q[9];
rx(-0.00022469861789653443) q[10];
rz(0.31378957963815085) q[10];
rx(3.1411694058829123) q[11];
rz(-0.04235185457151566) q[11];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.07358404137405365) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
h q[0];
h q[6];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.5879807066289684) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[6];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.2619972435878984) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
sdg q[0];
h q[0];
sdg q[6];
h q[6];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.3843606296105882) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[6];
s q[6];
cx q[0],q[1];
rz(-0.009895187971412003) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.021777964126431827) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
h q[1];
h q[7];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(0.03258866576234917) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[7];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.05287846072872223) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
sdg q[1];
h q[1];
sdg q[7];
h q[7];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.05962507093733783) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[7];
s q[7];
h q[2];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(0.48275185616253835) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[4];
h q[2];
h q[8];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
rz(1.3662869805756217) q[8];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[8];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(0.45519419894193125) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
sdg q[2];
h q[2];
sdg q[8];
h q[8];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
rz(1.3834381385840615) q[8];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[8];
s q[8];
cx q[2],q[3];
rz(2.4023375499662554) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-1.2532121442833415) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[5];
h q[3];
h q[9];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
rz(-1.0044587115916892) q[9];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[9];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.18005925173534512) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[5];
s q[5];
sdg q[3];
h q[3];
sdg q[9];
h q[9];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
rz(0.1721387382856645) q[9];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[9];
s q[9];
h q[4];
h q[10];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.1087809859130488) q[10];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[10];
sdg q[4];
h q[4];
sdg q[10];
h q[10];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.09941660037121632) q[10];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[10];
s q[10];
cx q[4],q[5];
rz(0.17815528072387418) q[5];
cx q[4],q[5];
h q[5];
h q[11];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.27897482248391964) q[11];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[11];
sdg q[5];
h q[5];
sdg q[11];
h q[11];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
rz(0.9905299317073851) q[11];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[11];
s q[11];
h q[6];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(0.7374964134937807) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
h q[8];
sdg q[6];
h q[6];
sdg q[8];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(0.6846920113215461) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(0.026470805723327998) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-1.5286235661788161) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
h q[9];
sdg q[7];
h q[7];
sdg q[9];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.03451188143808758) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
s q[7];
h q[9];
s q[9];
h q[8];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.201492506795833) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
h q[10];
sdg q[8];
h q[8];
sdg q[10];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.2008346360231442) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(2.9768703449088236e-05) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.12435709889343165) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
h q[11];
sdg q[9];
h q[9];
sdg q[11];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.010370696558595191) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.05502990944183935) q[11];
cx q[10],q[11];
rx(-0.003323268803840848) q[0];
rz(0.0025944370386386116) q[0];
rx(-0.0003249943176902037) q[1];
rz(0.28265804127959343) q[1];
rx(-9.208160826617841e-05) q[2];
rz(-2.4243956343434094) q[2];
rx(0.00011240837313242952) q[3];
rz(-0.051429521176321505) q[3];
rx(-3.653331620060841e-05) q[4];
rz(-0.083854799350957) q[4];
rx(-1.6408659478029988e-05) q[5];
rz(1.559946799217087) q[5];
rx(2.012826873862766e-05) q[6];
rz(0.028019494667175075) q[6];
rx(0.000139546166688742) q[7];
rz(0.22374838129658203) q[7];
rx(2.544718710179978e-05) q[8];
rz(0.005080321173883458) q[8];
rx(0.0001358553633558015) q[9];
rz(0.36030588541873426) q[9];
rx(0.00025906434290089267) q[10];
rz(-0.05617388234996166) q[10];
rx(2.039999683365885e-05) q[11];
rz(1.5999474763323391) q[11];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.5164875090219887) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
h q[0];
h q[6];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.4053199542091824) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[6];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.5732759095607607) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
sdg q[0];
h q[0];
sdg q[6];
h q[6];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.42301291853781453) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[6];
s q[6];
cx q[0],q[1];
rz(0.13109566523953678) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.9701266413708638) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
h q[1];
h q[7];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(-1.576562102794495) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[7];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.09134143924437718) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
sdg q[1];
h q[1];
sdg q[7];
h q[7];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.2449355997701196) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[7];
s q[7];
h q[2];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.3788794971315651) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[4];
h q[2];
h q[8];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
rz(-1.6350765705796024) q[8];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[8];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.3522748264707081) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
sdg q[2];
h q[2];
sdg q[8];
h q[8];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
rz(-1.6325807304971132) q[8];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[8];
s q[8];
cx q[2],q[3];
rz(0.07651893604696047) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-1.5781608990914442) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[5];
h q[3];
h q[9];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.7676872288171739) q[9];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[9];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(0.0015106021922465201) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[5];
s q[5];
sdg q[3];
h q[3];
sdg q[9];
h q[9];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.19927015117662314) q[9];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[9];
s q[9];
h q[4];
h q[10];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.2230291047448356) q[10];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[10];
sdg q[4];
h q[4];
sdg q[10];
h q[10];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.1866593113972639) q[10];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[10];
s q[10];
cx q[4],q[5];
rz(0.0017199234069206564) q[5];
cx q[4],q[5];
h q[5];
h q[11];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
rz(-1.3973097653368172) q[11];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[11];
sdg q[5];
h q[5];
sdg q[11];
h q[11];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.0020742860729950864) q[11];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[11];
s q[11];
h q[6];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(0.04075143751768734) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
h q[8];
sdg q[6];
h q[6];
sdg q[8];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(0.04939873657610709) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(0.1133522915094701) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-1.56128286331771) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
h q[9];
sdg q[7];
h q[7];
sdg q[9];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.16478955125234082) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
s q[7];
h q[9];
s q[9];
h q[8];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.45977508316372356) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
h q[10];
sdg q[8];
h q[8];
sdg q[10];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.3670504456072945) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(0.07510834844630686) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-1.5687226382449004) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
h q[11];
sdg q[9];
h q[9];
sdg q[11];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(0.014354550776654186) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.0001201737018680462) q[11];
cx q[10],q[11];
rx(0.0020915695181246367) q[0];
rz(2.3319119032176645) q[0];
rx(0.00030596273135732774) q[1];
rz(-0.030045280064209) q[1];
rx(5.3089822190532026e-05) q[2];
rz(0.7721443005761329) q[2];
rx(0.0002489116464167631) q[3];
rz(-0.052751694434048174) q[3];
rx(2.2634028552717184e-05) q[4];
rz(-0.8103070907709363) q[4];
rx(-5.5728936563084836e-05) q[5];
rz(-0.08268892500811113) q[5];
rx(7.372990858001859e-05) q[6];
rz(0.7404043094565965) q[6];
rx(7.437064648854643e-05) q[7];
rz(-0.0976559924160689) q[7];
rx(-4.409394920839375e-05) q[8];
rz(-0.7949665096988114) q[8];
rx(-0.0001766453261445202) q[9];
rz(0.09450706209129289) q[9];
rx(-8.864335730247114e-05) q[10];
rz(-2.426121114802811) q[10];
rx(-0.0006981054873907166) q[11];
rz(-0.14465976967414124) q[11];