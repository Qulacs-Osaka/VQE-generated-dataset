OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-1.0126349281407458) q[0];
rz(-2.812264612392844) q[0];
ry(-0.2251458074890591) q[1];
rz(-0.30654587186712146) q[1];
ry(-2.0533267233602617) q[2];
rz(1.737427603758961) q[2];
ry(2.754730963725496) q[3];
rz(0.1546961310534096) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.8031368142714156) q[0];
rz(1.550461263318859) q[0];
ry(-2.146665613765676) q[1];
rz(-0.7434469826959295) q[1];
ry(0.05554236116725608) q[2];
rz(-0.6076920321530181) q[2];
ry(2.579168836175022) q[3];
rz(-2.1774923341477015) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.018313810053942635) q[0];
rz(-1.810442226966881) q[0];
ry(1.1115041720599157) q[1];
rz(-1.2017743127825442) q[1];
ry(2.729285639429971) q[2];
rz(1.6298123742497144) q[2];
ry(-2.9010935158775735) q[3];
rz(-1.620476453472081) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.3253762349662441) q[0];
rz(-1.6291016550880641) q[0];
ry(-1.255058627214492) q[1];
rz(-2.8200289232417872) q[1];
ry(1.4972200723150237) q[2];
rz(2.601529465350226) q[2];
ry(1.8086067362755687) q[3];
rz(0.07868385258318432) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.1878295211241525) q[0];
rz(-0.8434343458949926) q[0];
ry(-2.463947219035553) q[1];
rz(2.2427802829980568) q[1];
ry(-0.1504546332350618) q[2];
rz(-2.0698027892594264) q[2];
ry(0.4110687759229224) q[3];
rz(-2.657038685399329) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.592821506731894) q[0];
rz(2.6470468993338616) q[0];
ry(-1.5948859489385996) q[1];
rz(1.1972639123163349) q[1];
ry(0.2985212086299729) q[2];
rz(-0.7240548246316568) q[2];
ry(2.926869407031193) q[3];
rz(-0.5679831451076703) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.8699090716239724) q[0];
rz(2.0969095696162516) q[0];
ry(-3.007931328413549) q[1];
rz(1.34000096496402) q[1];
ry(-2.4334081442974593) q[2];
rz(1.3715036612650129) q[2];
ry(-2.7249721702489778) q[3];
rz(1.434820988892742) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.2893321031187914) q[0];
rz(2.1661601342632606) q[0];
ry(2.823021184522277) q[1];
rz(0.9068643637524998) q[1];
ry(0.022775142903336414) q[2];
rz(-0.5262222491689938) q[2];
ry(1.084355383135037) q[3];
rz(3.0663417136999334) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.0604366052856076) q[0];
rz(-1.6276815808506273) q[0];
ry(-0.14616237027030543) q[1];
rz(-0.8624274563043345) q[1];
ry(2.6926409503099085) q[2];
rz(0.48524467119110043) q[2];
ry(1.4004635066542352) q[3];
rz(-2.151481898137058) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.2166017109005198) q[0];
rz(0.7803618065610163) q[0];
ry(-0.8875714994133869) q[1];
rz(-1.6335982061288226) q[1];
ry(1.9815613546524806) q[2];
rz(-1.4111774127766985) q[2];
ry(0.7000546738115663) q[3];
rz(-0.8049045906623924) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.29827204253787354) q[0];
rz(2.4861983085298753) q[0];
ry(2.823489684982706) q[1];
rz(0.07595577766148395) q[1];
ry(1.3125447089040898) q[2];
rz(-2.8426667813830875) q[2];
ry(-1.9696402455876916) q[3];
rz(-1.2290779428616962) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.8845353895587165) q[0];
rz(-2.435801663460611) q[0];
ry(2.2511504830326072) q[1];
rz(1.3318912634165025) q[1];
ry(2.725972930347473) q[2];
rz(1.5927150826388008) q[2];
ry(1.1214387222701951) q[3];
rz(-2.4793483274199803) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.569743973878575) q[0];
rz(0.5289186450470564) q[0];
ry(-0.6992163963232835) q[1];
rz(-1.0147037953637503) q[1];
ry(2.726828542338999) q[2];
rz(2.1732324369367246) q[2];
ry(-0.8397476262052281) q[3];
rz(-1.3965874988326807) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.1474899194419865) q[0];
rz(-1.1529282541641042) q[0];
ry(2.0761903784186586) q[1];
rz(3.002489160333338) q[1];
ry(-2.330251350233624) q[2];
rz(2.798802597579566) q[2];
ry(-1.8794178002762918) q[3];
rz(2.4681581300315827) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.8609233071778282) q[0];
rz(1.2831099798554282) q[0];
ry(2.7411308684991114) q[1];
rz(-1.905995201631256) q[1];
ry(1.015040694233841) q[2];
rz(1.2894907783913263) q[2];
ry(1.4701977874767433) q[3];
rz(-0.351293448976066) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.7000078724080545) q[0];
rz(-0.3490994331874155) q[0];
ry(0.21418203931660848) q[1];
rz(1.292876776814782) q[1];
ry(0.19837059553848976) q[2];
rz(1.1985345976872133) q[2];
ry(1.4034681211425164) q[3];
rz(2.0980354553047356) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.8807904738382631) q[0];
rz(-0.4701561521949227) q[0];
ry(-1.9741266988241932) q[1];
rz(0.6482953697152953) q[1];
ry(-0.6874559445736468) q[2];
rz(2.2897910392328185) q[2];
ry(-0.4619036857817358) q[3];
rz(1.352045388959028) q[3];