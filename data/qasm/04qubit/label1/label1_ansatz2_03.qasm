OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(0.5595791687888703) q[0];
rz(2.4615996394574102) q[0];
ry(-1.72407828738167) q[1];
rz(-2.947197866266942) q[1];
ry(1.6909103276108937) q[2];
rz(-2.532472252693621) q[2];
ry(-0.5074909487425989) q[3];
rz(-2.538302779921968) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.8640529455724102) q[0];
rz(2.229029811796255) q[0];
ry(-1.9735227035830478) q[1];
rz(1.5860231513891168) q[1];
ry(2.748105053950743) q[2];
rz(-1.2094223442971552) q[2];
ry(1.3879807612261263) q[3];
rz(0.8160994413770005) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.7671075157922544) q[0];
rz(-1.5175366289162069) q[0];
ry(2.967744220417021) q[1];
rz(-0.8689273062316779) q[1];
ry(0.03981814470042219) q[2];
rz(-1.620770215371195) q[2];
ry(1.4802988908673589) q[3];
rz(-2.0475371656864665) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.7878940118772486) q[0];
rz(1.2244453545056433) q[0];
ry(-3.099406224750543) q[1];
rz(-1.6921687190360046) q[1];
ry(2.9628779878548586) q[2];
rz(1.3933695707636238) q[2];
ry(1.4801380030746083) q[3];
rz(2.393492360426915) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.770899579516314) q[0];
rz(1.365636169558855) q[0];
ry(-2.244367037579256) q[1];
rz(-3.018692067008313) q[1];
ry(2.4963929886010114) q[2];
rz(0.6547879851912928) q[2];
ry(-1.1198070402613842) q[3];
rz(-1.4225643067768323) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(3.0824098334024637) q[0];
rz(0.6478202369755444) q[0];
ry(-2.707622657334151) q[1];
rz(-2.3257454758695872) q[1];
ry(1.8063522566632697) q[2];
rz(2.251077521879064) q[2];
ry(-0.21995029442657543) q[3];
rz(0.4261904289592904) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.1998736336861911) q[0];
rz(2.49526660987048) q[0];
ry(1.8658806572135067) q[1];
rz(-1.0178631852272575) q[1];
ry(0.6221456235695725) q[2];
rz(-1.7967807641337037) q[2];
ry(-0.4418590228324928) q[3];
rz(-0.6802561454915343) q[3];