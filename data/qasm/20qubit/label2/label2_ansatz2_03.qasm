OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(0.049973991447161914) q[0];
rz(3.141574433977783) q[0];
ry(3.0619325948786915e-08) q[1];
rz(-0.04897167324210549) q[1];
ry(-1.5707962256997547) q[2];
rz(3.1415926232061704) q[2];
ry(-1.5207602936312883) q[3];
rz(-0.7444218247900078) q[3];
ry(3.141588566820798) q[4];
rz(2.7546593821305176) q[4];
ry(3.0418620006540937) q[5];
rz(-0.9172295215248478) q[5];
ry(1.5707963389009623) q[6];
rz(-3.6581115418866762e-06) q[6];
ry(1.5707963755675194) q[7];
rz(3.080256598974789) q[7];
ry(-1.5707962853375574) q[8];
rz(-1.5706660660067076) q[8];
ry(-1.5707962994531774) q[9];
rz(1.3800750704135378) q[9];
ry(1.5707962752928855) q[10];
rz(-3.069913892881767) q[10];
ry(-1.570796351841624) q[11];
rz(0.022376927753823713) q[11];
ry(1.570796334639204) q[12];
rz(0.0038480238041751487) q[12];
ry(1.5707963030901804) q[13];
rz(-1.9932932695363172) q[13];
ry(1.570796351985253) q[14];
rz(6.801656482480212e-08) q[14];
ry(1.5707963260990079) q[15];
rz(0.34004263254354766) q[15];
ry(-1.5707963223720864) q[16];
rz(1.437287690798527e-08) q[16];
ry(-1.5707962841933578) q[17];
rz(-1.9300714271253128e-07) q[17];
ry(4.514778018105403e-07) q[18];
rz(-1.675272826248929) q[18];
ry(-3.041923968368047) q[19];
rz(4.935855773956025e-06) q[19];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
cz q[0],q[16];
cz q[0],q[17];
cz q[0],q[18];
cz q[0],q[19];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[1],q[16];
cz q[1],q[17];
cz q[1],q[18];
cz q[1],q[19];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[2],q[16];
cz q[2],q[17];
cz q[2],q[18];
cz q[2],q[19];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[3],q[16];
cz q[3],q[17];
cz q[3],q[18];
cz q[3],q[19];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[4],q[16];
cz q[4],q[17];
cz q[4],q[18];
cz q[4],q[19];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[5],q[16];
cz q[5],q[17];
cz q[5],q[18];
cz q[5],q[19];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[6],q[16];
cz q[6],q[17];
cz q[6],q[18];
cz q[6],q[19];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[7],q[16];
cz q[7],q[17];
cz q[7],q[18];
cz q[7],q[19];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[8],q[16];
cz q[8],q[17];
cz q[8],q[18];
cz q[8],q[19];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[9],q[16];
cz q[9],q[17];
cz q[9],q[18];
cz q[9],q[19];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[10],q[16];
cz q[10],q[17];
cz q[10],q[18];
cz q[10],q[19];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[11],q[16];
cz q[11],q[17];
cz q[11],q[18];
cz q[11],q[19];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[12],q[16];
cz q[12],q[17];
cz q[12],q[18];
cz q[12],q[19];
cz q[13],q[14];
cz q[13],q[15];
cz q[13],q[16];
cz q[13],q[17];
cz q[13],q[18];
cz q[13],q[19];
cz q[14],q[15];
cz q[14],q[16];
cz q[14],q[17];
cz q[14],q[18];
cz q[14],q[19];
cz q[15],q[16];
cz q[15],q[17];
cz q[15],q[18];
cz q[15],q[19];
cz q[16],q[17];
cz q[16],q[18];
cz q[16],q[19];
cz q[17],q[18];
cz q[17],q[19];
cz q[18],q[19];
ry(1.5707963619161627) q[0];
rz(-3.1415925851416038) q[0];
ry(1.5707963353863887) q[1];
rz(-3.14159172772651) q[1];
ry(-1.2618042127225766) q[2];
rz(-6.45669801095049e-08) q[2];
ry(1.0543087558545494e-07) q[3];
rz(0.7444217257706347) q[3];
ry(-3.1415926102413474) q[4];
rz(1.359133539223575) q[4];
ry(3.141592645028894) q[5];
rz(-2.4880247572024885) q[5];
ry(-3.0251189779286385) q[6];
rz(3.034780291317602) q[6];
ry(-1.092331135176802) q[7];
rz(-1.0190689289443515) q[7];
ry(-1.570795577065099) q[8];
rz(0.05162957197409757) q[8];
ry(1.5698374553636105) q[9];
rz(-1.9083692099529364) q[9];
ry(-1.630188915485974) q[10];
rz(-0.254950776326253) q[10];
ry(1.5842036122286203) q[11];
rz(2.719568085222674) q[11];
ry(0.3705190308044932) q[12];
rz(1.9534514660642914) q[12];
ry(3.141588872898295) q[13];
rz(-0.4224969865925296) q[13];
ry(1.3274819622356018) q[14];
rz(1.5707963410803567) q[14];
ry(-3.141591958462718) q[15];
rz(1.728710916142144) q[15];
ry(1.0067840832512622) q[16];
rz(1.570796266181097) q[16];
ry(-2.459039234510952) q[17];
rz(-1.5707963291026266) q[17];
ry(1.570796310830115) q[18];
rz(-1.570796337913117) q[18];
ry(1.5707962952045404) q[19];
rz(1.5707963363835022) q[19];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
cz q[0],q[16];
cz q[0],q[17];
cz q[0],q[18];
cz q[0],q[19];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[1],q[16];
cz q[1],q[17];
cz q[1],q[18];
cz q[1],q[19];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[2],q[16];
cz q[2],q[17];
cz q[2],q[18];
cz q[2],q[19];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[3],q[16];
cz q[3],q[17];
cz q[3],q[18];
cz q[3],q[19];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[4],q[16];
cz q[4],q[17];
cz q[4],q[18];
cz q[4],q[19];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[5],q[16];
cz q[5],q[17];
cz q[5],q[18];
cz q[5],q[19];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[6],q[16];
cz q[6],q[17];
cz q[6],q[18];
cz q[6],q[19];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[7],q[16];
cz q[7],q[17];
cz q[7],q[18];
cz q[7],q[19];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[8],q[16];
cz q[8],q[17];
cz q[8],q[18];
cz q[8],q[19];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[9],q[16];
cz q[9],q[17];
cz q[9],q[18];
cz q[9],q[19];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[10],q[16];
cz q[10],q[17];
cz q[10],q[18];
cz q[10],q[19];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[11],q[16];
cz q[11],q[17];
cz q[11],q[18];
cz q[11],q[19];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[12],q[16];
cz q[12],q[17];
cz q[12],q[18];
cz q[12],q[19];
cz q[13],q[14];
cz q[13],q[15];
cz q[13],q[16];
cz q[13],q[17];
cz q[13],q[18];
cz q[13],q[19];
cz q[14],q[15];
cz q[14],q[16];
cz q[14],q[17];
cz q[14],q[18];
cz q[14],q[19];
cz q[15],q[16];
cz q[15],q[17];
cz q[15],q[18];
cz q[15],q[19];
cz q[16],q[17];
cz q[16],q[18];
cz q[16],q[19];
cz q[17],q[18];
cz q[17],q[19];
cz q[18],q[19];
ry(-1.570789557529305) q[0];
rz(-1.570796299687027) q[0];
ry(0.302282615940439) q[1];
rz(1.5707953247978184) q[1];
ry(1.5707964012757478) q[2];
rz(-1.3485782037386205) q[2];
ry(1.5707975836337824) q[3];
rz(3.1415611090595386) q[3];
ry(-1.5707963106838303) q[4];
rz(-1.5317489392625214) q[4];
ry(1.5707962820406012) q[5];
rz(1.8174969294691543) q[5];
ry(-1.4164469424161315e-08) q[6];
rz(-2.3145233915780965) q[6];
ry(-3.1415926453469423) q[7];
rz(0.5755326383368803) q[7];
ry(-1.8364882947707883e-09) q[8];
rz(-1.616102569233969) q[8];
ry(-3.1415926386709137) q[9];
rz(-3.083127883884803) q[9];
ry(-2.34371180277816e-08) q[10];
rz(2.122367206700501) q[10];
ry(4.662114260725299e-08) q[11];
rz(1.3807377349453294) q[11];
ry(7.772452459420266e-08) q[12];
rz(2.0021255058395355) q[12];
ry(-1.5001887018360942) q[13];
rz(-1.5707963146966764) q[13];
ry(0.38732438791444207) q[14];
rz(1.644158332022166) q[14];
ry(3.1415924758015272) q[15];
rz(-1.7529260064181733) q[15];
ry(-2.2055723553040947) q[16];
rz(2.8290102877433796) q[16];
ry(2.2711345464251287) q[17];
rz(1.57079663486689) q[17];
ry(-1.5707963057839025) q[18];
rz(-2.159629088638695) q[18];
ry(-1.5707963119264168) q[19];
rz(1.6462043096981291) q[19];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
cz q[0],q[16];
cz q[0],q[17];
cz q[0],q[18];
cz q[0],q[19];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[1],q[16];
cz q[1],q[17];
cz q[1],q[18];
cz q[1],q[19];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[2],q[16];
cz q[2],q[17];
cz q[2],q[18];
cz q[2],q[19];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[3],q[16];
cz q[3],q[17];
cz q[3],q[18];
cz q[3],q[19];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[4],q[16];
cz q[4],q[17];
cz q[4],q[18];
cz q[4],q[19];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[5],q[16];
cz q[5],q[17];
cz q[5],q[18];
cz q[5],q[19];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[6],q[16];
cz q[6],q[17];
cz q[6],q[18];
cz q[6],q[19];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[7],q[16];
cz q[7],q[17];
cz q[7],q[18];
cz q[7],q[19];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[8],q[16];
cz q[8],q[17];
cz q[8],q[18];
cz q[8],q[19];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[9],q[16];
cz q[9],q[17];
cz q[9],q[18];
cz q[9],q[19];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[10],q[16];
cz q[10],q[17];
cz q[10],q[18];
cz q[10],q[19];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[11],q[16];
cz q[11],q[17];
cz q[11],q[18];
cz q[11],q[19];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[12],q[16];
cz q[12],q[17];
cz q[12],q[18];
cz q[12],q[19];
cz q[13],q[14];
cz q[13],q[15];
cz q[13],q[16];
cz q[13],q[17];
cz q[13],q[18];
cz q[13],q[19];
cz q[14],q[15];
cz q[14],q[16];
cz q[14],q[17];
cz q[14],q[18];
cz q[14],q[19];
cz q[15],q[16];
cz q[15],q[17];
cz q[15],q[18];
cz q[15],q[19];
cz q[16],q[17];
cz q[16],q[18];
cz q[16],q[19];
cz q[17],q[18];
cz q[17],q[19];
cz q[18],q[19];
ry(1.5708046911382565) q[0];
rz(0.32509296689824696) q[0];
ry(1.5708046303906409) q[1];
rz(-1.5707963744562141) q[1];
ry(-1.5707963114426313) q[2];
rz(2.9527561974206554e-05) q[2];
ry(-1.5707963392205142) q[3];
rz(1.532879440484261) q[3];
ry(-1.2284369077519841e-09) q[4];
rz(2.4335787569014355) q[4];
ry(-4.265967668004578e-08) q[5];
rz(0.3344556617982973) q[5];
ry(-3.141592536021762) q[6];
rz(2.291056867207466) q[6];
ry(3.6173870121558593e-08) q[7];
rz(-2.5840814547125213) q[7];
ry(2.459394160979315e-08) q[8];
rz(-1.5005735646942586) q[8];
ry(4.063475570745961e-08) q[9];
rz(-1.880703253764004) q[9];
ry(-3.1415925724723572) q[10];
rz(-2.2719934816226486) q[10];
ry(1.2346474065338953e-08) q[11];
rz(-2.6222305859731763) q[11];
ry(-7.281915070933564e-08) q[12];
rz(0.7534407223243251) q[12];
ry(-1.8122471080566795) q[13];
rz(0.767093356176752) q[13];
ry(9.076675129949416e-07) q[14];
rz(1.4974343997823758) q[14];
ry(3.0634422465750046) q[15];
rz(-3.0211373777109456) q[15];
ry(3.1415926398793705) q[16];
rz(2.8290303267548125) q[16];
ry(-0.31960170596357873) q[17];
rz(2.0937400466601472) q[17];
ry(-5.456489969661937e-09) q[18];
rz(0.8911215092793894) q[18];
ry(2.182966962749333e-08) q[19];
rz(-1.6462036562760103) q[19];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
cz q[0],q[16];
cz q[0],q[17];
cz q[0],q[18];
cz q[0],q[19];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[1],q[16];
cz q[1],q[17];
cz q[1],q[18];
cz q[1],q[19];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[2],q[16];
cz q[2],q[17];
cz q[2],q[18];
cz q[2],q[19];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[3],q[16];
cz q[3],q[17];
cz q[3],q[18];
cz q[3],q[19];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[4],q[16];
cz q[4],q[17];
cz q[4],q[18];
cz q[4],q[19];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[5],q[16];
cz q[5],q[17];
cz q[5],q[18];
cz q[5],q[19];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[6],q[16];
cz q[6],q[17];
cz q[6],q[18];
cz q[6],q[19];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[7],q[16];
cz q[7],q[17];
cz q[7],q[18];
cz q[7],q[19];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[8],q[16];
cz q[8],q[17];
cz q[8],q[18];
cz q[8],q[19];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[9],q[16];
cz q[9],q[17];
cz q[9],q[18];
cz q[9],q[19];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[10],q[16];
cz q[10],q[17];
cz q[10],q[18];
cz q[10],q[19];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[11],q[16];
cz q[11],q[17];
cz q[11],q[18];
cz q[11],q[19];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[12],q[16];
cz q[12],q[17];
cz q[12],q[18];
cz q[12],q[19];
cz q[13],q[14];
cz q[13],q[15];
cz q[13],q[16];
cz q[13],q[17];
cz q[13],q[18];
cz q[13],q[19];
cz q[14],q[15];
cz q[14],q[16];
cz q[14],q[17];
cz q[14],q[18];
cz q[14],q[19];
cz q[15],q[16];
cz q[15],q[17];
cz q[15],q[18];
cz q[15],q[19];
cz q[16],q[17];
cz q[16],q[18];
cz q[16],q[19];
cz q[17],q[18];
cz q[17],q[19];
cz q[18],q[19];
ry(3.141592638769292) q[0];
rz(1.7229231423118305) q[0];
ry(1.5707962622035132) q[1];
rz(1.5204880935041214) q[1];
ry(1.5707952553111104) q[2];
rz(-2.4184545353688307) q[2];
ry(-2.9502536829022017e-05) q[3];
rz(-2.6024017994528705) q[3];
ry(3.141592589674131) q[4];
rz(-2.2397579082281984) q[4];
ry(-8.795399430994166e-08) q[5];
rz(0.8699032702682806) q[5];
ry(-1.147281371867214) q[6];
rz(0.11277343708377473) q[6];
ry(7.562213077960678e-09) q[7];
rz(-2.180380876109722) q[7];
ry(3.771212003834953e-08) q[8];
rz(3.071133244643426) q[8];
ry(7.377316535439604e-09) q[9];
rz(-0.08116429799863933) q[9];
ry(3.1415924286740347) q[10];
rz(-0.9935553004100638) q[10];
ry(3.1415903195366375) q[11];
rz(1.4777751698461803) q[11];
ry(-3.1393890167756946) q[12];
rz(1.5710119482646911) q[12];
ry(-1.7855688483336962e-08) q[13];
rz(-0.7670933835353281) q[13];
ry(-2.681575184049219) q[14];
rz(1.5707964895458277) q[14];
ry(3.1415926166745214) q[15];
rz(0.12045688359835706) q[15];
ry(0.03870346925480917) q[16];
rz(1.5707763764067284) q[16];
ry(-1.067303108115425e-07) q[17];
rz(1.0478492430315747) q[17];
ry(-1.5707962524267423) q[18];
rz(2.9871673266943297) q[18];
ry(1.5707963068038806) q[19];
rz(1.5697690347315012) q[19];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
cz q[0],q[16];
cz q[0],q[17];
cz q[0],q[18];
cz q[0],q[19];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[1],q[16];
cz q[1],q[17];
cz q[1],q[18];
cz q[1],q[19];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[2],q[16];
cz q[2],q[17];
cz q[2],q[18];
cz q[2],q[19];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[3],q[16];
cz q[3],q[17];
cz q[3],q[18];
cz q[3],q[19];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[4],q[16];
cz q[4],q[17];
cz q[4],q[18];
cz q[4],q[19];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[5],q[16];
cz q[5],q[17];
cz q[5],q[18];
cz q[5],q[19];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[6],q[16];
cz q[6],q[17];
cz q[6],q[18];
cz q[6],q[19];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[7],q[16];
cz q[7],q[17];
cz q[7],q[18];
cz q[7],q[19];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[8],q[16];
cz q[8],q[17];
cz q[8],q[18];
cz q[8],q[19];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[9],q[16];
cz q[9],q[17];
cz q[9],q[18];
cz q[9],q[19];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[10],q[16];
cz q[10],q[17];
cz q[10],q[18];
cz q[10],q[19];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[11],q[16];
cz q[11],q[17];
cz q[11],q[18];
cz q[11],q[19];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[12],q[16];
cz q[12],q[17];
cz q[12],q[18];
cz q[12],q[19];
cz q[13],q[14];
cz q[13],q[15];
cz q[13],q[16];
cz q[13],q[17];
cz q[13],q[18];
cz q[13],q[19];
cz q[14],q[15];
cz q[14],q[16];
cz q[14],q[17];
cz q[14],q[18];
cz q[14],q[19];
cz q[15],q[16];
cz q[15],q[17];
cz q[15],q[18];
cz q[15],q[19];
cz q[16],q[17];
cz q[16],q[18];
cz q[16],q[19];
cz q[17],q[18];
cz q[17],q[19];
cz q[18],q[19];
ry(3.141592644364723) q[0];
rz(-0.24786769634527148) q[0];
ry(-2.6701946464908445e-08) q[1];
rz(1.6064149212623602) q[1];
ry(1.2909875035660437e-07) q[2];
rz(-2.306469003077292) q[2];
ry(-3.141592450551507) q[3];
rz(-2.652832123315339) q[3];
ry(-1.5708382451046816) q[4];
rz(2.406297977043225) q[4];
ry(4.207708336423365e-05) q[5];
rz(0.927590291273126) q[5];
ry(-1.581596986710565e-08) q[6];
rz(3.0285525712815398) q[6];
ry(-1.5708031045946271) q[7];
rz(-0.9849543985055956) q[7];
ry(-1.570796378425058) q[8];
rz(5.699238701417641e-05) q[8];
ry(-1.5707961916951942) q[9];
rz(-3.072909818835487) q[9];
ry(1.570802236994508) q[10];
rz(2.4895288920362915) q[10];
ry(1.5705805683609402) q[11];
rz(-2.440007109606452) q[11];
ry(-1.5408791059858664) q[12];
rz(1.8310043752627374) q[12];
ry(-2.3019755350442583) q[13];
rz(2.9190550913308178) q[13];
ry(1.3455373171768492) q[14];
rz(1.9153805575332372) q[14];
ry(1.6080274096865423) q[15];
rz(-1.9184552615487274) q[15];
ry(2.0437829025618433) q[16];
rz(-1.57060115652844) q[16];
ry(2.933287743674601) q[17];
rz(-0.6549504566384199) q[17];
ry(-1.5706387682830583) q[18];
rz(0.7630519126421039) q[18];
ry(2.987167454518901) q[19];
rz(-1.9768023456105017) q[19];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
cz q[0],q[16];
cz q[0],q[17];
cz q[0],q[18];
cz q[0],q[19];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[1],q[16];
cz q[1],q[17];
cz q[1],q[18];
cz q[1],q[19];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[2],q[16];
cz q[2],q[17];
cz q[2],q[18];
cz q[2],q[19];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[3],q[16];
cz q[3],q[17];
cz q[3],q[18];
cz q[3],q[19];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[4],q[16];
cz q[4],q[17];
cz q[4],q[18];
cz q[4],q[19];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[5],q[16];
cz q[5],q[17];
cz q[5],q[18];
cz q[5],q[19];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[6],q[16];
cz q[6],q[17];
cz q[6],q[18];
cz q[6],q[19];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[7],q[16];
cz q[7],q[17];
cz q[7],q[18];
cz q[7],q[19];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[8],q[16];
cz q[8],q[17];
cz q[8],q[18];
cz q[8],q[19];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[9],q[16];
cz q[9],q[17];
cz q[9],q[18];
cz q[9],q[19];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[10],q[16];
cz q[10],q[17];
cz q[10],q[18];
cz q[10],q[19];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[11],q[16];
cz q[11],q[17];
cz q[11],q[18];
cz q[11],q[19];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[12],q[16];
cz q[12],q[17];
cz q[12],q[18];
cz q[12],q[19];
cz q[13],q[14];
cz q[13],q[15];
cz q[13],q[16];
cz q[13],q[17];
cz q[13],q[18];
cz q[13],q[19];
cz q[14],q[15];
cz q[14],q[16];
cz q[14],q[17];
cz q[14],q[18];
cz q[14],q[19];
cz q[15],q[16];
cz q[15],q[17];
cz q[15],q[18];
cz q[15],q[19];
cz q[16],q[17];
cz q[16],q[18];
cz q[16],q[19];
cz q[17],q[18];
cz q[17],q[19];
cz q[18],q[19];
ry(2.947741019159635) q[0];
rz(-1.233716690449163) q[0];
ry(1.3774950962440027) q[1];
rz(-1.1573909758276235) q[1];
ry(-2.948277587294817) q[2];
rz(1.971360605539792) q[2];
ry(-2.9482776319290998) q[3];
rz(1.9713812627448224) q[3];
ry(-3.140785445092764) q[4];
rz(-0.3223071634859567) q[4];
ry(-0.0008024731052295649) q[5];
rz(2.7467272594793752) q[5];
ry(-0.9682196773669909) q[6];
rz(0.41318464722371673) q[6];
ry(3.141210441013693) q[7];
rz(0.9988759674015347) q[7];
ry(-1.5699281631332687) q[8];
rz(-1.1577519735247899) q[8];
ry(3.1407363079334027) q[9];
rz(2.0525237662480484) q[9];
ry(0.00017530981991784373) q[10];
rz(-0.5056854217294903) q[10];
ry(3.1414173495990507) q[11];
rz(2.6854292496826524) q[11];
ry(3.1411269858809403) q[12];
rz(0.6732566585115665) q[12];
ry(-0.00011972204418375674) q[13];
rz(2.206382834259858) q[13];
ry(-0.00030417151967743195) q[14];
rz(-3.0731272630320925) q[14];
ry(3.141299558896421) q[15];
rz(-3.07620172392067) q[15];
ry(0.48013046616478916) q[16];
rz(0.4128770985773179) q[16];
ry(3.141480748205012) q[17];
rz(1.3288990081285927) q[17];
ry(3.141587561019488) q[18];
rz(1.1760988462231978) q[18];
ry(4.77571825047606e-06) q[19];
rz(-2.3235548333702396) q[19];