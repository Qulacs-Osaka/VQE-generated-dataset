OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-2.0377242903126938) q[0];
rz(-0.25193918586568476) q[0];
ry(-1.4561567682265935e-07) q[1];
rz(-2.5755536882920134) q[1];
ry(1.5707958616586726) q[2];
rz(-2.7687323027563937) q[2];
ry(-1.5707964429189278) q[3];
rz(-1.8213281182342227) q[3];
ry(3.1415921714147874) q[4];
rz(-1.1317613784872351) q[4];
ry(-1.5707961304716738) q[5];
rz(2.029891034153308) q[5];
ry(1.5707970637135766) q[6];
rz(-2.243169729157893) q[6];
ry(1.5708004648502683) q[7];
rz(-1.1004736789247023) q[7];
ry(2.8890366287305196) q[8];
rz(1.4911466263722817) q[8];
ry(1.5707968333908344) q[9];
rz(0.6841785272262839) q[9];
ry(-1.570802151651821) q[10];
rz(0.7900027411279121) q[10];
ry(-1.5714131212989688) q[11];
rz(-0.9441874630835535) q[11];
ry(-0.25044722797942764) q[12];
rz(-2.4180873848443523) q[12];
ry(1.5708139440696307) q[13];
rz(2.538196148504571) q[13];
ry(-1.4576780615954572) q[14];
rz(-3.1415915373464967) q[14];
ry(4.5818822513865593e-07) q[15];
rz(-0.7026341744816743) q[15];
ry(2.681865821721771) q[16];
rz(-1.570801430106343) q[16];
ry(-3.1415913371178554) q[17];
rz(-1.877650339643699) q[17];
ry(-2.6437900977970936) q[18];
rz(-3.1415760624310085) q[18];
ry(-3.1415868648914214) q[19];
rz(1.6749917337919729) q[19];
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
ry(-3.141587801959795) q[0];
rz(-1.2413277677415606) q[0];
ry(3.141592613560109) q[1];
rz(-3.0908476813733223) q[1];
ry(1.4933705832289988e-06) q[2];
rz(-0.8762195029477907) q[2];
ry(-2.2055287020566572e-08) q[3];
rz(1.4433703298468805) q[3];
ry(4.599873459909531e-08) q[4];
rz(-1.0345799231854786) q[4];
ry(4.878201576019592e-08) q[5];
rz(-0.4720737944458353) q[5];
ry(-3.1415922658576596) q[6];
rz(2.4692193703555314) q[6];
ry(-5.710955015558739e-07) q[7];
rz(-2.453274356803218) q[7];
ry(1.7012247973624205e-07) q[8];
rz(0.009986004333202736) q[8];
ry(3.1415920329721447) q[9];
rz(2.299192572249708) q[9];
ry(-3.1415923971885737) q[10];
rz(2.5335698936941617) q[10];
ry(-7.359962603814552e-07) q[11];
rz(2.5149837058874835) q[11];
ry(1.8806556235148816e-07) q[12];
rz(-2.291876305418828) q[12];
ry(-1.8160046355575332e-07) q[13];
rz(2.7669387633027114) q[13];
ry(-1.570785500642469) q[14];
rz(-0.0005225208501968837) q[14];
ry(3.141592417594879) q[15];
rz(-3.0105344893478176) q[15];
ry(-1.5707966011696979) q[16];
rz(3.215278931824796e-06) q[16];
ry(-1.5707962852606168) q[17];
rz(-1.76564219802825) q[17];
ry(-1.57080307326531) q[18];
rz(-1.5561026503732215e-05) q[18];
ry(2.9900522643889103) q[19];
rz(1.0706540582106072e-05) q[19];
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
ry(1.301759061433927e-06) q[0];
rz(0.43623834154754615) q[0];
ry(3.1415925663889634) q[1];
rz(0.780397844033761) q[1];
ry(3.1415926073680565) q[2];
rz(-2.8364200087469964) q[2];
ry(3.1415924517287253) q[3];
rz(-1.948754365730331) q[3];
ry(3.1415925762664103) q[4];
rz(2.4447584066604726) q[4];
ry(3.1346235827765456) q[5];
rz(-0.012979594903942626) q[5];
ry(2.1344509067625066) q[6];
rz(2.2621672384453735) q[6];
ry(2.1963771968788635e-07) q[7];
rz(-2.989516865941142) q[7];
ry(-3.1415926386242266) q[8];
rz(1.394714711119305) q[8];
ry(-3.1405739169798608) q[9];
rz(-1.526552993734997) q[9];
ry(3.141569682906204) q[10];
rz(3.0121119893837496) q[10];
ry(-2.161938883147191) q[11];
rz(-1.5707973761121572) q[11];
ry(-3.141591325116625) q[12];
rz(-1.9898795147763622) q[12];
ry(0.04286988361316535) q[13];
rz(-2.1624072787893907) q[13];
ry(3.017684458314916) q[14];
rz(2.9709252394444707) q[14];
ry(1.5707963470424626) q[15];
rz(-3.141592615244115) q[15];
ry(-1.551314025983804) q[16];
rz(-1.6235533993260045) q[16];
ry(2.326292240607546) q[17];
rz(0.24576879638070184) q[17];
ry(3.062526904220554) q[18];
rz(1.7187711973838176) q[18];
ry(0.2712819258611825) q[19];
rz(2.2194850360471547) q[19];
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
ry(3.14159061660895) q[0];
rz(2.8392069241489257) q[0];
ry(1.5658886454161802e-07) q[1];
rz(2.6119762822563932) q[1];
ry(-2.9645424117312182) q[2];
rz(1.473364190418453) q[2];
ry(-1.5707961057756386) q[3];
rz(-1.2862198363085955) q[3];
ry(1.5707964458359986) q[4];
rz(5.7795289814268e-07) q[4];
ry(-1.570796624202356) q[5];
rz(0.356186961120053) q[5];
ry(2.0944214039982277e-07) q[6];
rz(0.41232246920564286) q[6];
ry(1.4396399921068337e-05) q[7];
rz(-2.881513307863192) q[7];
ry(5.886157010692727e-07) q[8];
rz(-3.0351580955317856) q[8];
ry(2.135224395779588) q[9];
rz(3.1415567384814715) q[9];
ry(-3.03696515047136e-07) q[10];
rz(0.049088840908853675) q[10];
ry(-2.2256387244854006) q[11];
rz(-6.824553303808046e-06) q[11];
ry(3.1415925417345334) q[12];
rz(-0.42148660566148727) q[12];
ry(-3.1409432489511326) q[13];
rz(0.000670553814471572) q[13];
ry(6.374563930044985e-06) q[14];
rz(-1.3199668088694896) q[14];
ry(-1.5707961848353496) q[15];
rz(-2.4817339540743704) q[15];
ry(-3.1415893535896022) q[16];
rz(-1.7688178436122997) q[16];
ry(3.1415917933851873) q[17];
rz(0.08454077065724792) q[17];
ry(-3.141588139258776) q[18];
rz(1.9017324628231438) q[18];
ry(-3.1415920287455315) q[19];
rz(-0.38192121171158716) q[19];
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
ry(4.348544170529209e-07) q[0];
rz(2.9994471417739144) q[0];
ry(-3.1415910369788507) q[1];
rz(0.7488913565444921) q[1];
ry(3.1415922164090433) q[2];
rz(-0.2492310854623002) q[2];
ry(6.764830483923611e-07) q[3];
rz(2.9199892723810574) q[3];
ry(1.5707959676745087) q[4];
rz(3.009653125161096) q[4];
ry(-3.68427394885012e-07) q[5];
rz(0.5728863802830909) q[5];
ry(-0.5784977138216219) q[6];
rz(-0.8429230459502586) q[6];
ry(-1.5707978608952562) q[7];
rz(3.0543309686985407) q[7];
ry(-1.5707970068084876) q[8];
rz(1.142240020980268) q[8];
ry(1.5707965801162205) q[9];
rz(-2.8155564284401047) q[9];
ry(1.636360533508726) q[10];
rz(0.923775744205078) q[10];
ry(1.570804360966978) q[11];
rz(2.140748489189466) q[11];
ry(1.5707956981489852) q[12];
rz(-2.161114546837223) q[12];
ry(1.5707535319980732) q[13];
rz(-0.09959119836309253) q[13];
ry(3.141587187232245) q[14];
rz(1.6514950025965858) q[14];
ry(-3.1415922387939568) q[15];
rz(-2.48173761071941) q[15];
ry(-1.0125092937048172e-05) q[16];
rz(0.14526296503119962) q[16];
ry(-3.931154934162168e-07) q[17];
rz(-2.850126242755271) q[17];
ry(-3.14158153047602) q[18];
rz(-2.958646258451485) q[18];
ry(-3.1415912817526457) q[19];
rz(-2.601417355042264) q[19];
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
ry(-1.3178613176734189e-05) q[0];
rz(-1.613793642961345) q[0];
ry(3.141591099904414) q[1];
rz(-0.6988971257097649) q[1];
ry(3.1415918048470948) q[2];
rz(0.714482078120239) q[2];
ry(-3.141592486997542) q[3];
rz(-0.948081754855878) q[3];
ry(-3.1415925487352117) q[4];
rz(-0.06631844240727869) q[4];
ry(-1.4735971401868675e-07) q[5];
rz(1.6765515614616664) q[5];
ry(-3.1415747949388475) q[6];
rz(-1.1768547241259393) q[6];
ry(3.1415917022990536) q[7];
rz(-2.4332788861483055) q[7];
ry(3.141592235777587) q[8];
rz(2.7786563175841983) q[8];
ry(7.266957421033854e-07) q[9];
rz(0.18755175053956388) q[9];
ry(-2.891289752693952e-06) q[10];
rz(2.2664888897609816) q[10];
ry(2.6792786578712927e-07) q[11];
rz(3.08677100563576) q[11];
ry(3.1415919053976764) q[12];
rz(2.6168955160665894) q[12];
ry(-6.831165615395207e-07) q[13];
rz(2.953684687381832) q[13];
ry(1.57087902427933) q[14];
rz(-1.505174688245253) q[14];
ry(1.5707960246901873) q[15];
rz(-0.2886684028320117) q[15];
ry(-1.5707972975775304) q[16];
rz(-3.075968791325919) q[16];
ry(1.5707961065886815) q[17];
rz(-1.8594664238992555) q[17];
ry(-1.5707961351258088) q[18];
rz(-1.5051720133744455) q[18];
ry(1.5707975055913495) q[19];
rz(2.8529194435249825) q[19];