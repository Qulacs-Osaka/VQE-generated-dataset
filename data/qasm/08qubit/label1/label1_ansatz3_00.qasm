OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(2.997973406338878) q[0];
rz(1.5709943498893493) q[0];
ry(-0.7781622275734392) q[1];
rz(1.571225136017171) q[1];
ry(-1.5703778777971844) q[2];
rz(-2.7754419521340545) q[2];
ry(-1.570556738810842) q[3];
rz(-1.090189939878294) q[3];
ry(2.760078657307295) q[4];
rz(1.5713167312223233) q[4];
ry(-0.00021660829496106257) q[5];
rz(0.00027461353542346245) q[5];
ry(0.3834489986362266) q[6];
rz(-3.0108290054335116) q[6];
ry(-1.5907449920143852) q[7];
rz(-1.5692738783955118) q[7];
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
ry(0.15821817986709139) q[0];
rz(-1.571034560251924) q[0];
ry(1.0574629925974832) q[1];
rz(1.5706997029561711) q[1];
ry(-1.3705129229665336) q[2];
rz(-1.545787420885205) q[2];
ry(-1.3456074831928797) q[3];
rz(0.9286837419326402) q[3];
ry(-0.14271967975861877) q[4];
rz(1.571827321359451) q[4];
ry(-1.0371388674904303) q[5];
rz(0.7412545568996698) q[5];
ry(-1.778122152777234) q[6];
rz(-1.5636916310255962) q[6];
ry(-1.7004439994591904) q[7];
rz(0.33695479392161903) q[7];
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
ry(0.5291027664887995) q[0];
rz(0.00048812253917904686) q[0];
ry(-1.2490796280988752) q[1];
rz(-1.57054190785884) q[1];
ry(-1.5707519490356558) q[2];
rz(-3.1410348617150894) q[2];
ry(-1.315198916714265e-05) q[3];
rz(-0.25676341573979927) q[3];
ry(-0.04198238636817741) q[4];
rz(-0.0014565828382248208) q[4];
ry(0.0003076254784641391) q[5];
rz(0.8294198260559273) q[5];
ry(3.042893394182034) q[6];
rz(0.009364793950139496) q[6];
ry(-1.5708863754828486) q[7];
rz(1.5713570536818613) q[7];
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
ry(-0.41070591781444943) q[0];
rz(2.1000679380152025) q[0];
ry(-2.680451762896725) q[1];
rz(2.100785834192278) q[1];
ry(-0.3916311637815028) q[2];
rz(2.100516044093846) q[2];
ry(-1.570980281138145) q[3];
rz(-2.6120213577359617) q[3];
ry(-1.0312484274888707) q[4];
rz(2.1001915993470432) q[4];
ry(1.7914480410045552) q[5];
rz(-1.0415250200828448) q[5];
ry(-0.05003778969320103) q[6];
rz(-1.042852818999413) q[6];
ry(1.749590106191871) q[7];
rz(2.1032926658897964) q[7];