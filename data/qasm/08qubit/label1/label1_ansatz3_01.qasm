OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(3.050878561563858) q[0];
rz(-3.1415918697136203) q[0];
ry(1.6688037820956898) q[1];
rz(6.430350578789297e-09) q[1];
ry(-1.5707966044587522) q[2];
rz(1.1311655671469916) q[2];
ry(2.0713977083505526) q[3];
rz(1.5708422877581516) q[3];
ry(1.5707961761679323) q[4];
rz(2.122178617963308) q[4];
ry(-2.279699484725918) q[5];
rz(-1.570792855970363) q[5];
ry(1.5707949789470765) q[6];
rz(0.8123070195573909) q[6];
ry(-0.2173243459656888) q[7];
rz(3.1415892607533342) q[7];
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
ry(1.5707964196549495) q[0];
rz(-1.7703581277744238) q[0];
ry(2.484035105398289) q[1];
rz(1.5707954051042137) q[1];
ry(0.9950248520739616) q[2];
rz(-1.5057745414631496) q[2];
ry(0.005495761146736) q[3];
rz(-1.5708425513037376) q[3];
ry(-2.9476116836235633e-07) q[4];
rz(-0.27515976492346716) q[4];
ry(3.120379182549771) q[5];
rz(3.766273233279851e-06) q[5];
ry(2.5627273642663972) q[6];
rz(-2.2858002958268453) q[6];
ry(1.570796053202442) q[7];
rz(1.1630305477024265) q[7];
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
ry(-2.092853642468537) q[0];
rz(-1.0874356806741903) q[0];
ry(0.0176241265237147) q[1];
rz(-1.5707951895310441) q[1];
ry(2.565425841810841) q[2];
rz(2.828236303318865) q[2];
ry(2.2065353263520295) q[3];
rz(-1.5708067273162518) q[3];
ry(4.7114651025026433e-07) q[4];
rz(1.1468216923029706) q[4];
ry(-2.5530463833051393) q[5];
rz(-4.1142140538852345e-07) q[5];
ry(2.732875263527026) q[6];
rz(-1.068614745667535) q[6];
ry(-1.7421842796828433) q[7];
rz(1.9698746702369814) q[7];
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
ry(1.570796329573385) q[0];
rz(5.324851967190391e-08) q[0];
ry(-2.581756923772872) q[1];
rz(3.1415912294341775) q[1];
ry(-2.5775930019537516) q[2];
rz(-1.8669412202834224) q[2];
ry(3.1367045392447) q[3];
rz(-1.5708123476058278) q[3];
ry(-5.749884834926888e-07) q[4];
rz(-0.712729360802065) q[4];
ry(-3.0133841420740684) q[5];
rz(-1.5707946845387186) q[5];
ry(-0.4089926159779079) q[6];
rz(-1.767186697098197) q[6];
ry(-1.5707954173557275) q[7];
rz(-3.1415906964351246) q[7];
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
ry(-2.6584665714107474) q[0];
rz(-2.708849232742987) q[0];
ry(0.11127063839542028) q[1];
rz(-2.708847930855834) q[1];
ry(1.5707963311940327) q[2];
rz(-1.1380528934010272) q[2];
ry(-2.904432369502195) q[3];
rz(-2.708853858288364) q[3];
ry(1.5707944078122957) q[4];
rz(-1.1380512069613662) q[4];
ry(-2.0878509159079233) q[5];
rz(0.43274655343121904) q[5];
ry(-1.5707947661677515) q[6];
rz(2.0035427986162704) q[6];
ry(0.37795247867757265) q[7];
rz(0.4327447541702725) q[7];