OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-1.1405717289487418e-07) q[0];
rz(-0.6852178656801229) q[0];
ry(2.98290614672134) q[1];
rz(-3.141553137771769) q[1];
ry(1.5206530462229433) q[2];
rz(2.8789214279121126e-07) q[2];
ry(-0.12236182470669771) q[3];
rz(-3.141582545456583) q[3];
ry(2.420009112864107) q[4];
rz(-3.5144043568635656e-06) q[4];
ry(3.141589120092395) q[5];
rz(0.3961500245859702) q[5];
ry(1.5241544236551376) q[6];
rz(-1.5707959532079665) q[6];
ry(-1.5707965661168257) q[7];
rz(2.244299431588739) q[7];
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
ry(-1.9640432924461493e-07) q[0];
rz(-2.2489316178595096) q[0];
ry(-0.6658300757172447) q[1];
rz(-3.1415776313476624) q[1];
ry(-1.6435221511907272) q[2];
rz(6.491637341454749e-06) q[2];
ry(-1.5940411694574206) q[3];
rz(2.6301369704384575e-07) q[3];
ry(-2.957198324833609) q[4];
rz(-0.33281211159467455) q[4];
ry(1.2190038930311493) q[5];
rz(-1.5707964643917853) q[5];
ry(-1.5707967316376017) q[6];
rz(1.5707965783389266) q[6];
ry(-1.0710681586090232e-07) q[7];
rz(0.8972932073966651) q[7];
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
ry(-1.9236346238016138e-08) q[0];
rz(1.806042995421559) q[0];
ry(0.14084236358447397) q[1];
rz(3.1415924681051846) q[1];
ry(-0.11896841981051297) q[2];
rz(-3.141560415541332) q[2];
ry(-2.5600562803994333) q[3];
rz(-1.5707960732581359) q[3];
ry(-1.0670355443664903e-07) q[4];
rz(-2.8087938507611647) q[4];
ry(-1.5707965354749756) q[5];
rz(-5.424324132278985e-08) q[5];
ry(-1.570796396833349) q[6];
rz(4.67973559724951e-05) q[6];
ry(-1.5707962014040817) q[7];
rz(1.5707935381556355) q[7];
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
ry(2.8170522118159733e-07) q[0];
rz(1.2880135093337834) q[0];
ry(0.8740544013374799) q[1];
rz(-1.5707877604272982) q[1];
ry(-3.126656113127172) q[2];
rz(-1.5707635284237889) q[2];
ry(-1.5707961895900313) q[3];
rz(3.141586822786155) q[3];
ry(-3.077807524934279) q[4];
rz(3.1415767307103284) q[4];
ry(1.5593326116724358) q[5];
rz(2.6188661650568137) q[5];
ry(-3.136878852173155) q[6];
rz(4.5106432859931544e-05) q[6];
ry(1.4310835862573406) q[7];
rz(0.8978479479179684) q[7];
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
ry(1.5707894806205278) q[0];
rz(-1.6040668158430618) q[0];
ry(-1.5707901330523908) q[1];
rz(1.537523075804296) q[1];
ry(1.5708032721194556) q[2];
rz(1.53752600920511) q[2];
ry(1.5707878061669822) q[3];
rz(-0.03327347407694692) q[3];
ry(-1.5708032005717412) q[4];
rz(-1.6040669210864686) q[4];
ry(3.1415828056011534) q[5];
rz(-0.5559998917117592) q[5];
ry(1.5708030595486262) q[6];
rz(1.5375258061764057) q[6];
ry(3.1415817353761795) q[7];
rz(-0.7062214951020443) q[7];