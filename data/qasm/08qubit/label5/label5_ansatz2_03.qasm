OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-3.1415906235171334) q[0];
rz(-2.4951511914214186) q[0];
ry(3.1415925838465326) q[1];
rz(0.4201240781615505) q[1];
ry(-3.1415904816970244) q[2];
rz(-1.289663413013117) q[2];
ry(-7.950825780803972e-06) q[3];
rz(2.554964231699352) q[3];
ry(-1.57079557013508) q[4];
rz(-0.2845925925576598) q[4];
ry(-3.1415832323665054) q[5];
rz(-1.6053071655306357) q[5];
ry(0.03109255376255504) q[6];
rz(-0.6202815631871399) q[6];
ry(-1.5707986377938783) q[7];
rz(-0.22578030467097843) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-3.083622849564841) q[0];
rz(-0.00047219932253739216) q[0];
ry(3.141592546223314) q[1];
rz(-2.544166098032156) q[1];
ry(3.1408814094319184) q[2];
rz(-3.094872195298563) q[2];
ry(3.141590949279767) q[3];
rz(-0.47819979535398005) q[3];
ry(-3.1298899928895776) q[4];
rz(2.287338553128186) q[4];
ry(-3.1538173379530576e-06) q[5];
rz(-0.035197618344561345) q[5];
ry(-1.570784485572878) q[6];
rz(-0.00013611603204566336) q[6];
ry(3.1300671004170866) q[7];
rz(1.3489697112678478) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.5693672616616243) q[0];
rz(-2.899585823799848) q[0];
ry(-3.1415925958686746) q[1];
rz(0.6926096521867436) q[1];
ry(-1.5708112990253007) q[2];
rz(-3.1390841046957743) q[2];
ry(-3.1414069070880526) q[3];
rz(0.13992772794798092) q[3];
ry(-0.1207420593737373) q[4];
rz(-1.7650422689122067) q[4];
ry(-1.7960782269454967e-05) q[5];
rz(2.856840416441757) q[5];
ry(1.5454490630370572) q[6];
rz(2.453475714118915) q[6];
ry(-3.0766425648206965) q[7];
rz(0.29646367274308555) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(3.1310079923604555) q[0];
rz(-0.36030954015389455) q[0];
ry(-4.097241035339039e-08) q[1];
rz(1.6462649684690271) q[1];
ry(1.5604535553619894) q[2];
rz(-0.8680993475986504) q[2];
ry(3.0230569575842745) q[3];
rz(-1.217163213389859) q[3];
ry(2.639205380461362e-06) q[4];
rz(-2.376929499318587) q[4];
ry(-1.5707906722329457) q[5];
rz(0.801061041083396) q[5];
ry(6.768643886369612e-05) q[6];
rz(-0.8826593893749255) q[6];
ry(-1.5707789764279294) q[7];
rz(-1.5707389287144864) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.5706616034463634) q[0];
rz(-0.013793102907412749) q[0];
ry(-3.141592550509358) q[1];
rz(-1.9671434165627932) q[1];
ry(-1.5709333221183641) q[2];
rz(3.1415792816328345) q[2];
ry(-6.572519034353519e-05) q[3];
rz(-0.3535781426104512) q[3];
ry(-0.7557159339462638) q[4];
rz(-2.4947905914785506) q[4];
ry(-1.570796210231297) q[5];
rz(-1.2131572504307773) q[5];
ry(-2.449553974007434) q[6];
rz(-2.136841496464978) q[6];
ry(2.333847748940164) q[7];
rz(-0.6830225247388286) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(3.1415089822120943) q[0];
rz(-0.013772269001082282) q[0];
ry(-1.570797156847105) q[1];
rz(1.5707955811451868) q[1];
ry(-1.5708195105761058) q[2];
rz(-2.173965632955131e-06) q[2];
ry(1.5707309917517502) q[3];
rz(-1.5707778079622674) q[3];
ry(-3.353201996633802e-07) q[4];
rz(0.9239983666712774) q[4];
ry(4.2699722229144754e-06) q[5];
rz(-1.928435788027465) q[5];
ry(-3.141572163213614) q[6];
rz(2.5755204134229968) q[6];
ry(7.868232409553774e-06) q[7];
rz(2.253832751163958) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.5707966909754054) q[0];
rz(0.5146058840191053) q[0];
ry(-1.570814498309895) q[1];
rz(2.885859976132797) q[1];
ry(1.5707960420987988) q[2];
rz(-1.0561901752482923) q[2];
ry(-1.5707973584728778) q[3];
rz(-0.25573284277115693) q[3];
ry(1.5707953662054124) q[4];
rz(2.085412228378416) q[4];
ry(1.5707963058391357) q[5];
rz(2.885847689734815) q[5];
ry(1.570796138218027) q[6];
rz(-2.626976629309358) q[6];
ry(1.5707971824724072) q[7];
rz(1.3150505658689964) q[7];