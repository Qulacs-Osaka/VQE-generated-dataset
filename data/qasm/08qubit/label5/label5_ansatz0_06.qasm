OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.5592220824460057) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
h q[0];
h q[4];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
rz(0.12161159794652832) q[4];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[4];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.6860551516641753) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
sdg q[0];
h q[0];
sdg q[4];
h q[4];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.19092749551076046) q[4];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[4];
s q[4];
cx q[0],q[1];
rz(0.08656041037397239) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.6774524961115982) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
h q[1];
h q[5];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.16136269114589524) q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[5];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.5965215609469575) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
sdg q[1];
h q[1];
sdg q[5];
h q[5];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
rz(0.29913964835673107) q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[5];
s q[5];
h q[2];
h q[6];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.2529862801007823) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[6];
sdg q[2];
h q[2];
sdg q[6];
h q[6];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(0.3274646486952646) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[6];
s q[6];
cx q[2],q[3];
rz(0.3966865557393608) q[3];
cx q[2],q[3];
h q[3];
h q[7];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.5000741406577585) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[7];
sdg q[3];
h q[3];
sdg q[7];
h q[7];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(0.5762506778659408) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[7];
s q[7];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.22545537908352253) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(0.0154526373543105) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(-0.07289407021482669) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.18725515649829605) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.5369012325915534) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.518155854912959) q[7];
cx q[6],q[7];
rx(-5.560297510549575e-05) q[0];
rz(0.12235959878851134) q[0];
rx(6.727549293316565e-05) q[1];
rz(-0.24198997039513487) q[1];
rx(9.836752397718883e-06) q[2];
rz(-0.06680327058725989) q[2];
rx(6.263982023811943e-05) q[3];
rz(0.07454016031817938) q[3];
rx(2.3375682540643534e-05) q[4];
rz(0.2707438050673027) q[4];
rx(-8.153176214370617e-05) q[5];
rz(0.4026669762376535) q[5];
rx(2.2782703467386573e-05) q[6];
rz(-0.2498016430360727) q[6];
rx(-0.000675617404472605) q[7];
rz(0.2304768545360633) q[7];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.027510257041888878) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
h q[0];
h q[4];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.20072218329346658) q[4];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[4];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.0008573470774162397) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
sdg q[0];
h q[0];
sdg q[4];
h q[4];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
rz(0.23858032485875058) q[4];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[4];
s q[4];
cx q[0],q[1];
rz(0.8829445170620474) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.04437641039788089) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
h q[1];
h q[5];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.045402413954833805) q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[5];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.1557810759932505) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
sdg q[1];
h q[1];
sdg q[5];
h q[5];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.005180283766250917) q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[5];
s q[5];
h q[2];
h q[6];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.0980564427883545) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[6];
sdg q[2];
h q[2];
sdg q[6];
h q[6];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(0.06836763212036596) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[6];
s q[6];
cx q[2],q[3];
rz(-0.7427171221511052) q[3];
cx q[2],q[3];
h q[3];
h q[7];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.17803229026565706) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[7];
sdg q[3];
h q[3];
sdg q[7];
h q[7];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(0.20257098471275742) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[7];
s q[7];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.8587727529126471) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(0.035428949820185726) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(0.10547664781452484) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.4915501922983134) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(0.2125200616558726) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[6],q[7];
rz(-0.1265899111397303) q[7];
cx q[6],q[7];
rx(-0.0004928042140726423) q[0];
rz(-0.2900652164179372) q[0];
rx(-0.000640694604366069) q[1];
rz(-0.71916894344924) q[1];
rx(3.122355554505317e-05) q[2];
rz(0.22425957977940536) q[2];
rx(-0.0005270159814550737) q[3];
rz(0.1991557252962238) q[3];
rx(-0.00011164177875407934) q[4];
rz(-0.203662288686169) q[4];
rx(1.957372814213157e-05) q[5];
rz(0.26749900930078435) q[5];
rx(-1.0621129788744258e-06) q[6];
rz(-0.15653815274030958) q[6];
rx(-0.7047408714266158) q[7];
rz(0.00033111907140669365) q[7];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.0037672695690223173) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
h q[0];
h q[4];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
rz(0.03695238260484773) q[4];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[4];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.1038820214168714) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
sdg q[0];
h q[0];
sdg q[4];
h q[4];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
rz(0.025638113534828925) q[4];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[4];
s q[4];
cx q[0],q[1];
rz(-0.3529384993328938) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.04636893420099009) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
h q[1];
h q[5];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
rz(0.041965399879729755) q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[5];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.049835787849883864) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
sdg q[1];
h q[1];
sdg q[5];
h q[5];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.1919431619283526) q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[5];
s q[5];
h q[2];
h q[6];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(0.0405241811487145) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[6];
sdg q[2];
h q[2];
sdg q[6];
h q[6];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.002103647604735155) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[6];
s q[6];
cx q[2],q[3];
rz(-0.9246277378076319) q[3];
cx q[2],q[3];
h q[3];
h q[7];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(0.13086438458492786) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[7];
sdg q[3];
h q[3];
sdg q[7];
h q[7];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.000767450903805996) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[7];
s q[7];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.9025870806139973) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.22126683930521626) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(-0.15489309596489897) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.5990796713438559) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.14872116748432862) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.0009154853280969888) q[7];
cx q[6],q[7];
rx(-0.6648572538335037) q[0];
rz(-0.16376813182405633) q[0];
rx(-0.38234965905928586) q[1];
rz(-0.12975425313868402) q[1];
rx(-0.00027167561195542007) q[2];
rz(0.20509556156543376) q[2];
rx(-0.8506011428398782) q[3];
rz(-0.05208596951549458) q[3];
rx(-0.8545351382267443) q[4];
rz(-0.002189601944196265) q[4];
rx(-0.00330584040552654) q[5];
rz(-0.019995328761811204) q[5];
rx(-0.00011135848564191278) q[6];
rz(0.005307281127920737) q[6];
rx(-0.8140779873225908) q[7];
rz(0.0007766026199471134) q[7];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.0047077212533920285) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
h q[0];
h q[4];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.00013508965545236108) q[4];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[4];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.00015481246396671059) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
sdg q[0];
h q[0];
sdg q[4];
h q[4];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
rz(-6.554383715548849e-06) q[4];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[4];
s q[4];
cx q[0],q[1];
rz(0.00047530791068782605) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.0005636083390204201) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
h q[1];
h q[5];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
rz(-1.8660436400796753e-05) q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[5];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(2.4271765947557413e-05) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
sdg q[1];
h q[1];
sdg q[5];
h q[5];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
rz(5.457343094996115e-06) q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[5];
s q[5];
h q[2];
h q[6];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(-2.3860789737517588e-05) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[6];
sdg q[2];
h q[2];
sdg q[6];
h q[6];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(2.590957958949122e-06) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[6];
s q[6];
cx q[2],q[3];
rz(-0.007206029969956075) q[3];
cx q[2],q[3];
h q[3];
h q[7];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(3.366982474306098e-06) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[7];
sdg q[3];
h q[3];
sdg q[7];
h q[7];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(-3.2488479026322957e-06) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[7];
s q[7];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.209370361044327) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.006864705216794784) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(-1.1078655284622914e-06) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.5076805408528261) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(0.11558417841908089) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[6],q[7];
rz(-0.0005934241240491648) q[7];
cx q[6],q[7];
rx(-0.7188881195392464) q[0];
rz(-0.025705720582457903) q[0];
rx(-0.1875051514402719) q[1];
rz(0.09065533508692038) q[1];
rx(0.00040696786260728963) q[2];
rz(0.11992650282083839) q[2];
rx(-1.2230344735653118) q[3];
rz(0.04463071429969753) q[3];
rx(-1.35579748043302) q[4];
rz(0.0020309966105268766) q[4];
rx(0.0054327820872768) q[5];
rz(-0.051205118894904415) q[5];
rx(0.00014960130183092567) q[6];
rz(0.07554803341039336) q[6];
rx(-0.6772038718750469) q[7];
rz(-0.0022468433355323906) q[7];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.0032475398116011424) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
h q[0];
h q[4];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
rz(0.0001055762825517406) q[4];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[4];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-2.3426503331699408e-05) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
sdg q[0];
h q[0];
sdg q[4];
h q[4];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
rz(-3.942908281142417e-05) q[4];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[4];
s q[4];
cx q[0],q[1];
rz(-0.00024250331368199396) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.010215154905179791) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
h q[1];
h q[5];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
rz(2.806629283712837e-05) q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[5];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-6.543826077531149e-06) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
sdg q[1];
h q[1];
sdg q[5];
h q[5];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
rz(-5.992094441435749e-06) q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[5];
s q[5];
h q[2];
h q[6];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(1.0536138480390194e-05) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[6];
sdg q[2];
h q[2];
sdg q[6];
h q[6];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(2.7379798900805257e-06) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[6];
s q[6];
cx q[2],q[3];
rz(0.006644558606969712) q[3];
cx q[2],q[3];
h q[3];
h q[7];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(-8.036305541330706e-06) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[7];
sdg q[3];
h q[3];
sdg q[7];
h q[7];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(-1.2114676333404776e-05) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[7];
s q[7];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.2967728510013885) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(0.006508824405648725) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(1.1985085905153231e-05) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.7413916084636895) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.02355328455445195) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[6],q[7];
rz(-2.3760273955059433e-05) q[7];
cx q[6],q[7];
rx(-0.8703429755994045) q[0];
rz(0.16118471679926782) q[0];
rx(0.5690645734339388) q[1];
rz(-0.47112232295535234) q[1];
rx(-0.0001810235557691095) q[2];
rz(0.01729336103342822) q[2];
rx(-1.0687765249858843) q[3];
rz(0.07970840351124905) q[3];
rx(-0.9312086925692484) q[4];
rz(0.07497462313351323) q[4];
rx(-0.0023479659821947314) q[5];
rz(-0.37305505405974143) q[5];
rx(-7.09761301778027e-05) q[6];
rz(-0.01138566638920601) q[6];
rx(-0.950963539599696) q[7];
rz(0.17419551084992757) q[7];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.20607124576230418) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
h q[0];
h q[4];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.0966230244676784) q[4];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[4];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(6.66758390831992e-05) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
sdg q[0];
h q[0];
sdg q[4];
h q[4];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
rz(-5.542743449616032e-05) q[4];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[4];
s q[4];
cx q[0],q[1];
rz(0.00010472338079033696) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.028687901242815055) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
h q[1];
h q[5];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
rz(0.08801154111084132) q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[5];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.22329164644065172) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
sdg q[1];
h q[1];
sdg q[5];
h q[5];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.1786381110001042) q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[5];
s q[5];
h q[2];
h q[6];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.0009445541550490118) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[6];
sdg q[2];
h q[2];
sdg q[6];
h q[6];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.0040679702295491164) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[6];
s q[6];
cx q[2],q[3];
rz(0.455244677878102) q[3];
cx q[2],q[3];
h q[3];
h q[7];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.11685965310327273) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[7];
sdg q[3];
h q[3];
sdg q[7];
h q[7];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.22252268859186752) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[7];
s q[7];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.21182028030698616) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(0.007401934473232296) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(-0.12115100153774402) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.0152945295082747) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(0.19956111954106864) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[6],q[7];
rz(-0.29617793278140286) q[7];
cx q[6],q[7];
rx(-0.9002440182174297) q[0];
rz(0.27034306797024815) q[0];
rx(0.0001605958583759075) q[1];
rz(-0.4419689204037326) q[1];
rx(-2.4509425146889595e-05) q[2];
rz(-0.000596822292964488) q[2];
rx(4.82799325089286e-05) q[3];
rz(-0.2865102411870918) q[3];
rx(2.180794882555272e-05) q[4];
rz(-0.31048107338928066) q[4];
rx(-5.659539718050788e-05) q[5];
rz(-0.3771480873569044) q[5];
rx(2.0655700320238437e-05) q[6];
rz(-0.519687147958913) q[6];
rx(0.0013172891099464273) q[7];
rz(0.3308792921952944) q[7];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.2817668066956825) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
h q[0];
h q[4];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
rz(0.342169816712131) q[4];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[4];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.06644251170086662) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
sdg q[0];
h q[0];
sdg q[4];
h q[4];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
rz(0.15296936970040245) q[4];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[4];
s q[4];
cx q[0],q[1];
rz(0.12823887782896098) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.3147616161442461) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
h q[1];
h q[5];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.18842873984665645) q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[5];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.4382934833616398) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
sdg q[1];
h q[1];
sdg q[5];
h q[5];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
rz(0.1312140717727466) q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[5];
s q[5];
h q[2];
h q[6];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(0.04663596049915892) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[6];
sdg q[2];
h q[2];
sdg q[6];
h q[6];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(0.031135051189501415) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[6];
s q[6];
cx q[2],q[3];
rz(0.49983210891969343) q[3];
cx q[2],q[3];
h q[3];
h q[7];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(0.2834256143715443) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[7];
sdg q[3];
h q[3];
sdg q[7];
h q[7];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(0.3788859102204638) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[7];
s q[7];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.17974486549916047) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.04221167980256134) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(-0.14215218941209656) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.17488028246150578) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.24676337355477643) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.3779633382150297) q[7];
cx q[6],q[7];
rx(-0.00027537730914188504) q[0];
rz(0.6910645070917927) q[0];
rx(-2.836778653885753e-05) q[1];
rz(-0.7334099206428013) q[1];
rx(5.665247187243216e-05) q[2];
rz(-0.18693302952857577) q[2];
rx(-8.300168906164828e-05) q[3];
rz(0.25810889228184536) q[3];
rx(-1.3764590451662379e-05) q[4];
rz(-0.6501690866076322) q[4];
rx(0.0001668904514713466) q[5];
rz(0.152042228084187) q[5];
rx(-2.3638564207718702e-05) q[6];
rz(-0.41526014713500997) q[6];
rx(-0.00048811210950641865) q[7];
rz(-0.26077852130489676) q[7];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.3499853671616043) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
h q[0];
h q[4];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.5526069891755571) q[4];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[4];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.31637977074208024) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
sdg q[0];
h q[0];
sdg q[4];
h q[4];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.5037630630213941) q[4];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[4];
s q[4];
cx q[0],q[1];
rz(1.0233015091116993) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.0750291585985645) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
h q[1];
h q[5];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.3180884784163886) q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[5];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.03704957774973518) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
sdg q[1];
h q[1];
sdg q[5];
h q[5];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.2897285025160057) q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[5];
s q[5];
h q[2];
h q[6];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.4022785913759039) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[6];
sdg q[2];
h q[2];
sdg q[6];
h q[6];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.42565381746778846) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[6];
s q[6];
cx q[2],q[3];
rz(1.4627902235881511) q[3];
cx q[2],q[3];
h q[3];
h q[7];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.9349275050230493) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[7];
sdg q[3];
h q[3];
sdg q[7];
h q[7];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.9819727215052396) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[7];
s q[7];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.6861497382094496) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.6883357822019563) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(0.09476714448521262) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.4773719414099171) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.402758881747124) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.24210082165129743) q[7];
cx q[6],q[7];
rx(5.11200046997098e-05) q[0];
rz(1.192882309754396) q[0];
rx(0.00010448333161811193) q[1];
rz(-0.01568566319759909) q[1];
rx(-3.70815307285231e-06) q[2];
rz(-1.2189609923961369) q[2];
rx(1.9619407591604575e-05) q[3];
rz(-1.5152651248698545) q[3];
rx(-3.3578875117880377e-06) q[4];
rz(-0.6216631501140902) q[4];
rx(2.9623173649288804e-05) q[5];
rz(0.7390837783682068) q[5];
rx(2.148069730490443e-05) q[6];
rz(0.2093469541400212) q[6];
rx(1.5975598221348237e-05) q[7];
rz(0.3011808538012915) q[7];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.6799277171383706) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
h q[0];
h q[4];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.8905848682406445) q[4];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[4];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.6742124731387606) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
sdg q[0];
h q[0];
sdg q[4];
h q[4];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.9052397818671101) q[4];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[4];
s q[4];
cx q[0],q[1];
rz(0.8227293130667392) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.012673112604426747) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
h q[1];
h q[5];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.6928627338567116) q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[5];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.028426925963149077) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
sdg q[1];
h q[1];
sdg q[5];
h q[5];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.7029593151750438) q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[5];
s q[5];
h q[2];
h q[6];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.32549067038354873) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[6];
sdg q[2];
h q[2];
sdg q[6];
h q[6];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.29951029174982585) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[6];
s q[6];
cx q[2],q[3];
rz(-0.05494094512813388) q[3];
cx q[2],q[3];
h q[3];
h q[7];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.012055108054787081) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[7];
sdg q[3];
h q[3];
sdg q[7];
h q[7];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.020049350919672235) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[7];
s q[7];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.5474587175966208) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.5164946868115537) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(0.06571473631958842) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.5787851698356749) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.6566305482183297) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.16370879074740335) q[7];
cx q[6],q[7];
rx(-4.494102344641438e-05) q[0];
rz(0.09903041401762035) q[0];
rx(8.64699645531567e-05) q[1];
rz(-0.7884994300052902) q[1];
rx(2.6335627967898436e-06) q[2];
rz(-0.34742433666876826) q[2];
rx(-2.0488866706026126e-05) q[3];
rz(1.4999145546571957) q[3];
rx(2.095717847185342e-06) q[4];
rz(-0.029675905760205217) q[4];
rx(-1.1244391993618713e-05) q[5];
rz(-0.38620688229573247) q[5];
rx(-4.160645263761583e-06) q[6];
rz(0.09011295971828268) q[6];
rx(-2.6515393186800392e-05) q[7];
rz(-0.577842470972282) q[7];