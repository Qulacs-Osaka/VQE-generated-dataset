OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(3.1415693266769793) q[0];
rz(-1.7638704670649314) q[0];
ry(-1.5708568024020002) q[1];
rz(-1.5708568300108252) q[1];
ry(-1.6360289212258294e-07) q[2];
rz(-2.4440884469173594) q[2];
ry(-1.5707873026199985) q[3];
rz(0.2066541573676145) q[3];
ry(-1.5708009256351163) q[4];
rz(-1.524662324978572) q[4];
ry(-1.566620549893551) q[5];
rz(1.5707974392273103) q[5];
ry(1.2766404203348738) q[6];
rz(-1.2244434214991884e-05) q[6];
ry(-0.00010688776122158572) q[7];
rz(1.333034522410257) q[7];
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
ry(1.5707974301104777) q[0];
rz(-1.4031771387878988) q[0];
ry(-0.26290233451100425) q[1];
rz(1.5708550735540745) q[1];
ry(-1.57079302525025) q[2];
rz(1.4821745361328251) q[2];
ry(-3.14159197807131) q[3];
rz(-1.6669666273504482) q[3];
ry(5.727970880101493e-06) q[4];
rz(1.0273908665397613) q[4];
ry(1.570795541561882) q[5];
rz(2.3278372424895175) q[5];
ry(-1.570793375451337) q[6];
rz(-2.6451141319582323) q[6];
ry(1.5707991811397601) q[7];
rz(0.11718969658738818) q[7];
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
ry(1.6193294367630577) q[0];
rz(0.00042093647916749264) q[0];
ry(-1.570790016647499) q[1];
rz(0.18694399204009635) q[1];
ry(-3.141592587837902) q[2];
rz(-0.0686538024021866) q[2];
ry(3.141592178369611) q[3];
rz(-2.1365225405065607) q[3];
ry(3.1415915672799954) q[4];
rz(-2.9449167048201943) q[4];
ry(-2.971818306647705) q[5];
rz(-0.8064658136552999) q[5];
ry(-5.401452076725377e-07) q[6];
rz(-0.4966826621819084) q[6];
ry(1.5707971859500818) q[7];
rz(3.1415870317043777) q[7];
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
ry(-1.8749040618864166) q[0];
rz(-1.4167417585149877) q[0];
ry(-1.4656107190091916e-06) q[1];
rz(1.326160326557961) q[1];
ry(-9.636469554801005e-07) q[2];
rz(0.13401886307162056) q[2];
ry(1.5709477916022943) q[3];
rz(2.9162782258622526) q[3];
ry(-3.14158681287955) q[4];
rz(-0.7242310753491151) q[4];
ry(1.5707960996336696) q[5];
rz(1.3454853917540295) q[5];
ry(-3.1294143787818487) q[6];
rz(1.7232098468746084) q[6];
ry(-1.5707989742785686) q[7];
rz(1.3454861430499472) q[7];