OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-1.5707955567436427) q[0];
rz(1.5707957743286196) q[0];
ry(1.5707926198089268) q[1];
rz(-2.326246913841255) q[1];
ry(1.5707990091067114) q[2];
rz(1.108965141325401) q[2];
ry(1.5708327235582284) q[3];
rz(-1.5707965401980992) q[3];
ry(-2.271862106632914e-06) q[4];
rz(1.5135291281068122) q[4];
ry(0.11396966404521257) q[5];
rz(1.57077692658999) q[5];
ry(-1.5707971136099455) q[6];
rz(2.0480017658222325) q[6];
ry(1.5707960333178745) q[7];
rz(-1.6289693196401913) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.594485034796247) q[0];
rz(1.8109964321882928e-05) q[0];
ry(-3.977294975843386e-06) q[1];
rz(2.7313731500139373) q[1];
ry(1.735521712174262e-08) q[2];
rz(2.0326271702063154) q[2];
ry(1.57079432862162) q[3];
rz(3.141583858513341) q[3];
ry(1.22004396340708e-07) q[4];
rz(1.8520105643189095) q[4];
ry(-1.5707960834839683) q[5];
rz(3.1415809970416655) q[5];
ry(-1.4366871186802719) q[6];
rz(0.2530922132040585) q[6];
ry(-1.7661723588689107) q[7];
rz(-0.2914641449150772) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.4651716525472773) q[0];
rz(-3.616857786070681e-05) q[0];
ry(-0.060539000018932844) q[1];
rz(-0.8103949977718933) q[1];
ry(1.5708053911865492) q[2];
rz(-1.2261074021509812) q[2];
ry(1.5707947497372574) q[3];
rz(-0.2941294142865214) q[3];
ry(1.5707960440703974) q[4];
rz(3.1415575894484653) q[4];
ry(1.5707964613212655) q[5];
rz(0.5433374256017024) q[5];
ry(1.5707861256304125) q[6];
rz(2.710251987048665) q[6];
ry(-2.8148677859680964) q[7];
rz(-1.5707849600029196) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.5707924862632705) q[0];
rz(1.9233544841770032) q[0];
ry(3.081868806123698) q[1];
rz(-1.6751406152967183) q[1];
ry(-3.1415918297158547) q[2];
rz(1.7804329508109777) q[2];
ry(-3.14159255082687) q[3];
rz(-0.29412964072426906) q[3];
ry(3.0767469729880346) q[4];
rz(-3.52703889161532e-05) q[4];
ry(3.124411840360608e-08) q[5];
rz(-0.45622998703102063) q[5];
ry(-1.570804678704929) q[6];
rz(-3.1415767924713367) q[6];
ry(-1.5707966269996358) q[7];
rz(0.6167123791049898) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.997789007670586) q[0];
rz(-2.8851294796402542) q[0];
ry(-2.5161866732403837e-06) q[1];
rz(1.3869945999895406) q[1];
ry(3.1415489102061707) q[2];
rz(-1.7058485310364047) q[2];
ry(-1.5707952349615921) q[3];
rz(2.368150813930104) q[3];
ry(-1.5707951836819474) q[4];
rz(1.570795791155839) q[4];
ry(-1.5385284985790602) q[5];
rz(1.5696403628557842) q[5];
ry(-1.5707896535411856) q[6];
rz(0.6668296531049682) q[6];
ry(-3.1415925825393534) q[7];
rz(0.6167285559267636) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-6.679069745274545e-06) q[0];
rz(-1.7070027507168164) q[0];
ry(3.1415924670828255) q[1];
rz(-2.858983845950588) q[1];
ry(1.5707965059936164) q[2];
rz(2.927151018210755) q[2];
ry(7.767514372247978e-08) q[3];
rz(2.0307395571679323) q[3];
ry(-1.5707952745914961) q[4];
rz(-2.982674028154922) q[4];
ry(3.3903920776909504e-07) q[5];
rz(1.2634628530600258) q[5];
ry(4.029137604444344e-05) q[6];
rz(-0.4373752004473603) q[6];
ry(-1.570789654947378) q[7];
rz(1.9050410308125239) q[7];