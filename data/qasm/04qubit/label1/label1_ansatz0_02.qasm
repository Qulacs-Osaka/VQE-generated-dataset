OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.22543267675215972) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.2964742417104893) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.09068335635491134) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-1.1704389949248342) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.8614088847520456) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.10235214409625872) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.6176909357094105) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-1.0979470873012398) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.5678838639184104) q[3];
cx q[2],q[3];
rz(-0.10139421457597479) q[0];
rz(0.5327774694839492) q[1];
rz(0.11313711217103782) q[2];
rz(-0.09249290297715893) q[3];
rx(-0.4099546778871893) q[0];
rx(-1.164446982646942) q[1];
rx(-0.31409026519799066) q[2];
rx(-0.8916350297618709) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.44158024810704116) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.5717220987025552) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.01903848345715325) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-1.2624763641218428) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(1.4313389300818962) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.2502496651763703) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.6893108181037795) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.7895040748523199) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.24189455517329506) q[3];
cx q[2],q[3];
rz(0.019278968949728765) q[0];
rz(-0.18591372451480925) q[1];
rz(-0.03772593952560477) q[2];
rz(0.4385315910405081) q[3];
rx(-0.31616203986074465) q[0];
rx(-0.3444229935602071) q[1];
rx(-0.2880813484106383) q[2];
rx(-0.9504175438910313) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.019204769484873062) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.005899717691077655) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.11478584211152676) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-1.484811169464684) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(1.6945743462392324) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.15072932011526266) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.4329912735121068) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.6666232343111372) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.23631464760807225) q[3];
cx q[2],q[3];
rz(-0.15622428383642523) q[0];
rz(-0.16774785419534202) q[1];
rz(-0.4571526702143065) q[2];
rz(0.3426245083354987) q[3];
rx(-0.3948719655301569) q[0];
rx(-0.9539705101018342) q[1];
rx(-0.17481500075882708) q[2];
rx(-0.898831939601407) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.3294297618925385) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.07158646969123107) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.26692849937318397) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-1.6934423827075846) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(1.4334781626713102) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.1474795792861715) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.7104276575569624) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.6574219490705324) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.3900638761207939) q[3];
cx q[2],q[3];
rz(-0.3163648491299029) q[0];
rz(0.20999655113063287) q[1];
rz(4.369503050440905e-05) q[2];
rz(0.13928746967952269) q[3];
rx(-0.2635186629458837) q[0];
rx(-1.1658655985806563) q[1];
rx(0.2875098322762101) q[2];
rx(-1.0161803964900926) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.9273078932291944) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.8611689657718833) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.17278247196613797) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-1.662391198272424) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(1.359504572313273) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.4271909173639887) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.13056637321610148) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.8242229427186069) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.1956667322783151) q[3];
cx q[2],q[3];
rz(-0.03421600124820244) q[0];
rz(0.050112662677200805) q[1];
rz(0.11924333506432044) q[2];
rz(-0.06056634796687996) q[3];
rx(-0.6876858561174234) q[0];
rx(-0.8825905289525319) q[1];
rx(0.4228634433657942) q[2];
rx(-1.1452266856441011) q[3];