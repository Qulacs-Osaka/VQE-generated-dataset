OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-2.464765381676452) q[0];
rz(-3.141570147782193) q[0];
ry(-1.0998639097201623) q[1];
rz(7.818419867078319e-08) q[1];
ry(1.5707959152694222) q[2];
rz(1.570797440858368) q[2];
ry(-3.1415921954952655) q[3];
rz(3.0569219301389383) q[3];
ry(1.5708175196112069) q[4];
rz(-1.0172950781729133) q[4];
ry(1.3701521685500313) q[5];
rz(-1.5707925638837974) q[5];
ry(-0.2168029355799277) q[6];
rz(-1.5707825642332804) q[6];
ry(3.141592547168966) q[7];
rz(1.7234356759393217) q[7];
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
ry(1.5707962240699196) q[0];
rz(-1.5707959912320018) q[0];
ry(-1.9521187047504434) q[1];
rz(-2.2879632239302827) q[1];
ry(3.0117873450978765) q[2];
rz(-3.141592456694099) q[2];
ry(1.6603150623816414) q[3];
rz(-3.141592394182489) q[3];
ry(-1.5707962304928327) q[4];
rz(1.5707959670689264) q[4];
ry(-0.8560152111847934) q[5];
rz(1.5707874939874238) q[5];
ry(2.994184243642145) q[6];
rz(-1.5707687211874797) q[6];
ry(1.5707970792941577) q[7];
rz(1.3897761529477157) q[7];
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
ry(1.570795499276187) q[0];
rz(-1.8800835830058) q[0];
ry(-3.1415925514521135) q[1];
rz(2.4244441377427317) q[1];
ry(0.12721195211869205) q[2];
rz(-1.570796326782725) q[2];
ry(1.570796528971015) q[3];
rz(2.4831170846643323e-07) q[3];
ry(-1.570794625623794) q[4];
rz(2.3425638103091728e-05) q[4];
ry(-3.01787508154978) q[5];
rz(-1.570797806929917) q[5];
ry(2.6011636876691164) q[6];
rz(-2.6153155058754463) q[6];
ry(1.2030635928795874e-06) q[7];
rz(-1.3897762389469328) q[7];
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
ry(1.5708021264182266) q[0];
rz(-1.579743937298163) q[0];
ry(0.27808513263260154) q[1];
rz(1.6767294705448992) q[1];
ry(-1.7829916702169744) q[2];
rz(-1.57079660972357) q[2];
ry(1.7402280103524956) q[3];
rz(1.5707962319319666) q[3];
ry(-1.5274608305791944) q[4];
rz(-7.3580997170807905e-06) q[4];
ry(-1.6830474082928477) q[5];
rz(3.4935828668223954e-07) q[5];
ry(-6.905094318554461e-07) q[6];
rz(0.5510514747648472) q[6];
ry(1.570795696602546) q[7];
rz(1.3386898226189758) q[7];
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
ry(3.1415914686121833) q[0];
rz(1.5618534917377007) q[0];
ry(3.141592603561038) q[1];
rz(-3.035665127338828) q[1];
ry(1.5707965472728382) q[2];
rz(-2.7026649966899186e-06) q[2];
ry(-1.5707945751548715) q[3];
rz(-5.76653691908291e-07) q[3];
ry(-1.570793682523007) q[4];
rz(1.570796532504314) q[4];
ry(-1.5707970005110323) q[5];
rz(1.5708149784856904) q[5];
ry(3.141591973880129) q[6];
rz(-0.49346961837360115) q[6];
ry(8.583770929604384e-07) q[7];
rz(0.2320952524624128) q[7];
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
ry(-0.7196023588684008) q[0];
rz(-1.1149583309330353) q[0];
ry(0.35820969209932907) q[1];
rz(1.1149678654249373) q[1];
ry(2.953806116895468) q[2];
rz(-0.0034229937041789778) q[2];
ry(1.9424299962661342) q[3];
rz(-3.1381602438509706) q[3];
ry(2.777684038883677) q[4];
rz(-0.006975715217594544) q[4];
ry(3.104485171144923) q[5];
rz(-0.006981940438983259) q[5];
ry(1.0322587689990725) q[6];
rz(-0.4558404209591751) q[6];
ry(0.3713423948518111) q[7];
rz(-0.45582594512310864) q[7];
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
ry(-1.567724235994853) q[0];
rz(2.5408070967484955) q[0];
ry(-1.5738791056161971) q[1];
rz(-0.6007867425080597) q[1];
ry(-1.1149599272733655) q[2];
rz(-2.1700763326707717) q[2];
ry(2.026624996211412) q[3];
rz(-2.1700716318483058) q[3];
ry(2.6857414922041123) q[4];
rz(0.9637468858932183) q[4];
ry(-2.685753783586595) q[5];
rz(-2.1778685302907927) q[5];
ry(1.573867039740012) q[6];
rz(2.5408066852786577) q[6];
ry(1.573876819560342) q[7];
rz(2.5408064137890336) q[7];