OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-1.5707959463915533) q[0];
rz(-1.0111616788321955e-07) q[0];
ry(1.5208623651608608) q[1];
rz(-3.1411278984887394) q[1];
ry(-1.0799760881087035e-06) q[2];
rz(2.6957875485562792) q[2];
ry(0.09776649874986365) q[3];
rz(1.6405168834863622) q[3];
ry(-1.3993352680982454) q[4];
rz(3.1415311630971017) q[4];
ry(-1.4051138471266933) q[5];
rz(3.141592397156625) q[5];
ry(1.1330616569438234e-06) q[6];
rz(-0.5816342747797721) q[6];
ry(-2.9521093485542567) q[7];
rz(-3.14159133556762) q[7];
ry(1.5064638433188824) q[8];
rz(3.141592269941534) q[8];
ry(1.5707365982960237) q[9];
rz(-0.04658808985318213) q[9];
ry(1.5707987087609596) q[10];
rz(-1.1378972463838012e-05) q[10];
ry(-1.570796410129869) q[11];
rz(-1.9590962164844186) q[11];
ry(1.570796352768461) q[12];
rz(-3.1415926492824093) q[12];
ry(3.141592639164674) q[13];
rz(-2.9979328601384094) q[13];
ry(-1.5707964158082834) q[14];
rz(2.830104642095523) q[14];
ry(1.5210643472017515) q[15];
rz(-0.39567711873549444) q[15];
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
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(1.2159050986931084) q[0];
rz(1.648790990834047) q[0];
ry(-5.671182072387941e-05) q[1];
rz(-1.5526780004003253) q[1];
ry(-3.1415925157849136) q[2];
rz(1.153454385888149) q[2];
ry(3.141568345547822) q[3];
rz(1.7068931062840962) q[3];
ry(-3.044614175941877) q[4];
rz(-1.5117404055222305) q[4];
ry(2.7778526727231054) q[5];
rz(0.8938819862849456) q[5];
ry(1.5707964433061659) q[6];
rz(-3.141588291466367) q[6];
ry(-1.5852491942527491) q[7];
rz(2.544182820802255) q[7];
ry(1.61312533896171) q[8];
rz(2.6391252952109694) q[8];
ry(7.176088961435763e-05) q[9];
rz(-2.603845887760671) q[9];
ry(-3.076878407976735) q[10];
rz(-1.5708077127077489) q[10];
ry(-3.1415926440295006) q[11];
rz(-0.5346801240012551) q[11];
ry(-1.4570817312022373) q[12];
rz(-0.007272484253083949) q[12];
ry(-3.1415925982866195) q[13];
rz(-2.1514287381667367) q[13];
ry(-3.1415918701973764) q[14];
rz(1.259309008210237) q[14];
ry(5.61527324954613e-07) q[15];
rz(-1.9700443536703327) q[15];
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
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(-2.449994199022868) q[0];
rz(1.6465506309412605) q[0];
ry(3.141522589733098) q[1];
rz(1.5937121124921643) q[1];
ry(-3.1415926119436053) q[2];
rz(-2.1361810420544822) q[2];
ry(-3.1415926328566712) q[3];
rz(-1.7869584619664978) q[3];
ry(-3.1415926464745914) q[4];
rz(0.31591643501263444) q[4];
ry(3.141592448366845) q[5];
rz(3.12688829931767) q[5];
ry(1.570789760335629) q[6];
rz(-3.0806452519459424) q[6];
ry(7.779642501759554e-06) q[7];
rz(0.6375671460608698) q[7];
ry(-3.1415923618174815) q[8];
rz(2.0024854997734023) q[8];
ry(8.572382981242299e-08) q[9];
rz(-1.3655519661297797) q[9];
ry(-2.2504784342603816) q[10];
rz(-0.36304032908436357) q[10];
ry(0.46914970067507694) q[11];
rz(0.4060085413076848) q[11];
ry(-1.4717703938708997) q[12];
rz(1.9709336798416448) q[12];
ry(-3.1415900441583884) q[13];
rz(-2.4826539814416306) q[13];
ry(0.5483914019072413) q[14];
rz(-1.5695871054144592) q[14];
ry(3.141592619276974) q[15];
rz(0.7758709368934964) q[15];
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
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(-3.1415453764768135) q[0];
rz(0.015638136700833465) q[0];
ry(3.1415504220782533) q[1];
rz(-3.137259461313807) q[1];
ry(3.1415925584904594) q[2];
rz(0.8891377615398436) q[2];
ry(-1.8918991617766776e-07) q[3];
rz(0.28252047521855506) q[3];
ry(-2.0414684698266683e-07) q[4];
rz(-0.25680062785400964) q[4];
ry(-1.6128768645172824e-08) q[5];
rz(-0.6621469563174257) q[5];
ry(3.1415795919815914) q[6];
rz(0.4757647851065552) q[6];
ry(-1.2923930204244982e-05) q[7];
rz(1.7670882624105113) q[7];
ry(3.1415915957330363) q[8];
rz(1.9808812063429535) q[8];
ry(-1.0266429064234553e-06) q[9];
rz(2.4451899015236602) q[9];
ry(-1.3982559973711883e-07) q[10];
rz(-3.122047305967716) q[10];
ry(-2.849592561915415e-09) q[11];
rz(-0.2752539331139943) q[11];
ry(3.141592619074667) q[12];
rz(0.32670664053376175) q[12];
ry(1.5707958271042823) q[13];
rz(2.8504246907424338) q[13];
ry(3.141418944907325) q[14];
rz(0.0012098413521595206) q[14];
ry(-1.5707118038167502) q[15];
rz(-0.0044604325771935285) q[15];
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
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(1.5707964759830428) q[0];
rz(3.643889063179678e-05) q[0];
ry(1.5707973407656877) q[1];
rz(-0.8832729866143573) q[1];
ry(1.5707964978847526) q[2];
rz(-1.571055407995953) q[2];
ry(-1.5861959275167266) q[3];
rz(-0.7309292797102697) q[3];
ry(-0.43564049602648014) q[4];
rz(0.8933395449276824) q[4];
ry(0.1291636666810972) q[5];
rz(0.04734773846251272) q[5];
ry(3.141398122362822) q[6];
rz(-1.4327479678313138) q[6];
ry(-0.0001176388522688826) q[7];
rz(2.692503838480114) q[7];
ry(-6.570090389525066e-06) q[8];
rz(2.096233897126399) q[8];
ry(-0.9979976559828281) q[9];
rz(1.5707934343477827) q[9];
ry(1.1567325560144557e-05) q[10];
rz(-1.2273013775557544) q[10];
ry(-1.5707995631774876) q[11];
rz(1.5565834830780516) q[11];
ry(-1.570791579153492) q[12];
rz(-1.6667126315060388) q[12];
ry(5.0744532913427065e-06) q[13];
rz(-1.1692214621598707) q[13];
ry(-1.5707961926308736) q[14];
rz(2.1985524331561077) q[14];
ry(-1.5707963797838307) q[15];
rz(0.3650391499171768) q[15];
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
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(1.5707871130539108) q[0];
rz(-3.141592527934244) q[0];
ry(-3.574405829098737e-07) q[1];
rz(2.0669709751003036) q[1];
ry(0.04971561286840309) q[2];
rz(2.0187404503234596) q[2];
ry(-3.1415773843413106) q[3];
rz(-0.5751728090007192) q[3];
ry(6.571895112004014e-07) q[4];
rz(2.29236615721435) q[4];
ry(-3.141592551434472) q[5];
rz(-2.9072960062597266) q[5];
ry(-3.141590121779245) q[6];
rz(2.997433316671395) q[6];
ry(3.1415901062873406) q[7];
rz(3.02767955646245) q[7];
ry(-3.1415923321346066) q[8];
rz(-0.0067159086781573745) q[8];
ry(2.9833048316091717) q[9];
rz(1.5707930890915256) q[9];
ry(1.7290492028959523) q[10];
rz(1.570796199394604) q[10];
ry(3.551776155674702e-05) q[11];
rz(2.9203488633831487) q[11];
ry(-1.1055882138677475e-06) q[12];
rz(1.4692133272638461) q[12];
ry(-1.157287291775333e-06) q[13];
rz(1.6517955744048693) q[13];
ry(3.141592621036329) q[14];
rz(-2.784680020766435) q[14];
ry(5.206848907590711e-08) q[15];
rz(-0.36503741860436373) q[15];
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
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(1.5708054681779027) q[0];
rz(-3.141457980011013) q[0];
ry(4.01292612653111e-08) q[1];
rz(1.9579096526948996) q[1];
ry(1.265652826987207e-08) q[2];
rz(3.1198805918217443) q[2];
ry(3.141592579565173) q[3];
rz(-0.9393957967529792) q[3];
ry(-3.1415911112957247) q[4];
rz(1.1104754950531586) q[4];
ry(1.569320215999426e-06) q[5];
rz(1.133019211525445) q[5];
ry(1.3337114110001382e-05) q[6];
rz(0.9337086610322975) q[6];
ry(3.1415792416225967) q[7];
rz(-1.0996833018447187) q[7];
ry(-1.9226577182594618e-05) q[8];
rz(1.5920282358280013) q[8];
ry(1.0249292128441434) q[9];
rz(-1.5707958957199721) q[9];
ry(-1.024916802250685) q[10];
rz(1.570796264991903) q[10];
ry(4.231001504706455e-08) q[11];
rz(2.829899800291366) q[11];
ry(4.771815881768447e-06) q[12];
rz(-1.3323892743540529) q[12];
ry(-3.1415878954591188) q[13];
rz(0.7489799479385228) q[13];
ry(1.57075371445474) q[14];
rz(3.141591563385396) q[14];
ry(1.570796359625017) q[15];
rz(2.73740064168188) q[15];
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
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(-1.5707964369656342) q[0];
rz(3.1413892143740854) q[0];
ry(-1.5707963640416875) q[1];
rz(1.6039827567813718) q[1];
ry(2.7668374196210266e-06) q[2];
rz(1.1448041556032678) q[2];
ry(-1.8609389044854652e-05) q[3];
rz(0.4750094960094804) q[3];
ry(3.141465555026604) q[4];
rz(1.163486417406445) q[4];
ry(3.1414815962037728) q[5];
rz(1.6841737123563936) q[5];
ry(-3.141581229169331) q[6];
rz(-3.137314827196906) q[6];
ry(1.0540135948922114e-05) q[7];
rz(-1.620708899923519) q[7];
ry(-1.7955279858483664e-05) q[8];
rz(-1.6204983761400704) q[8];
ry(1.1939376370629835) q[9];
rz(-1.2480447131430683) q[9];
ry(-1.946924690515049) q[10];
rz(-1.5707879739311013) q[10];
ry(1.4653433905567727e-07) q[11];
rz(1.82047687860827) q[11];
ry(3.141592636896906) q[12];
rz(-2.645096538744402) q[12];
ry(-3.1415925414552577) q[13];
rz(-2.9262511780993217) q[13];
ry(1.5707995860478041) q[14];
rz(-1.9133415981604331) q[14];
ry(3.38557490822498e-06) q[15];
rz(-1.362781511929247) q[15];
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
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(-1.570789474279748) q[0];
rz(-0.0340534065390203) q[0];
ry(-3.1413892581406393) q[1];
rz(-1.5716632416336833) q[1];
ry(-1.5708096369857554) q[2];
rz(1.536762333425637) q[2];
ry(3.1415688111177804) q[3];
rz(2.4874161270278266) q[3];
ry(-1.84538579430793e-05) q[4];
rz(3.010455758409355) q[4];
ry(-3.1415893752650383) q[5];
rz(-2.8113383492094624) q[5];
ry(1.205991949987375e-07) q[6];
rz(-2.113567840936658) q[6];
ry(-1.6905103086482318e-07) q[7];
rz(2.785102984196942) q[7];
ry(5.2617549964692345e-05) q[8];
rz(0.0025338323288161342) q[8];
ry(-1.0844198403248129e-05) q[9];
rz(-0.35677016099550984) q[9];
ry(1.7134192332022158) q[10];
rz(3.107574779777838) q[10];
ry(3.047992328584836e-05) q[11];
rz(0.2634497240489724) q[11];
ry(2.4507247883231607e-08) q[12];
rz(-2.060405425379679) q[12];
ry(-3.1415926075571257) q[13];
rz(-0.3762533033663322) q[13];
ry(6.877988057333548e-06) q[14];
rz(1.8793096266978608) q[14];
ry(6.847481833105461e-06) q[15];
rz(1.7329415695377302) q[15];