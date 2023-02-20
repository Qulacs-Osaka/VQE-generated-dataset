OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-1.5018959290903406) q[0];
rz(-1.8380456613377905) q[0];
ry(-2.795778501068832) q[1];
rz(0.3656192988313146) q[1];
ry(0.8538941198261744) q[2];
rz(-1.8429537197438473) q[2];
ry(-0.254041313310065) q[3];
rz(0.16776619981434937) q[3];
ry(1.5155258410407964) q[4];
rz(3.0743843438035534) q[4];
ry(-1.3569690058637318) q[5];
rz(-1.5510396174323198) q[5];
ry(-1.6119197954028703) q[6];
rz(-1.5746918117795792) q[6];
ry(-3.131010423920855) q[7];
rz(-2.589707707412073) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(1.4371707059253105) q[0];
rz(2.412356129600332) q[0];
ry(3.141092132380631) q[1];
rz(1.6754515482215142) q[1];
ry(3.1180121130189336) q[2];
rz(3.007382338063652) q[2];
ry(0.00622193665981019) q[3];
rz(-1.3857194643429704) q[3];
ry(-3.141292923877895) q[4];
rz(0.26685841160116386) q[4];
ry(-1.5912907060315906) q[5];
rz(-1.5529427605334618) q[5];
ry(1.2599312483015703) q[6];
rz(-1.3770210599568502) q[6];
ry(1.4266699424517217) q[7];
rz(-1.726353296565704) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-0.2698195562752678) q[0];
rz(-0.8313507667072074) q[0];
ry(-0.004147280922475246) q[1];
rz(0.3169986668358042) q[1];
ry(0.7840377380924314) q[2];
rz(2.9926109288298486) q[2];
ry(-1.5312682158460529) q[3];
rz(2.7616681165065615) q[3];
ry(1.6269374040210356) q[4];
rz(-3.098029928989159) q[4];
ry(-0.23251279095960078) q[5];
rz(0.21309184970307063) q[5];
ry(-0.047477939411104224) q[6];
rz(-0.8474921563416116) q[6];
ry(-2.8845933642430768) q[7];
rz(-1.7282498913202424) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-1.5485288031950226) q[0];
rz(1.4572994301050217) q[0];
ry(0.02534047908341286) q[1];
rz(-1.7145237930025594) q[1];
ry(-2.930988949111361) q[2];
rz(-1.62799470138047) q[2];
ry(-3.129556505992612) q[3];
rz(-1.9506989610708159) q[3];
ry(0.0022954048315154303) q[4];
rz(1.1407093278037213) q[4];
ry(0.02614478578379753) q[5];
rz(1.41683283163459) q[5];
ry(0.0010705591477462306) q[6];
rz(2.247254214840266) q[6];
ry(1.705992469699174) q[7];
rz(1.0774020731880256) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(1.5707177199133717) q[0];
rz(3.1389239344777) q[0];
ry(1.5709159844929415) q[1];
rz(-3.102296483630625) q[1];
ry(-1.5779639341007916) q[2];
rz(-0.19170488756280202) q[2];
ry(1.5274941894446588) q[3];
rz(-1.5424393544620387) q[3];
ry(-1.5727851151153782) q[4];
rz(-0.002802067635908483) q[4];
ry(2.6242023638251606) q[5];
rz(-1.4967018938498544) q[5];
ry(1.0496169594881204) q[6];
rz(1.5484878674058455) q[6];
ry(-1.5167353973680262) q[7];
rz(3.109204947575465) q[7];