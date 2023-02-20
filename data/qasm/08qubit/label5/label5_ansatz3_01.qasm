OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(0.0013322287194545979) q[0];
rz(1.31227887006841) q[0];
ry(2.7599399520412775) q[1];
rz(2.1755837850253315) q[1];
ry(1.572202327331341) q[2];
rz(0.7383911575579081) q[2];
ry(-1.2650444907588243) q[3];
rz(-0.6962200964383555) q[3];
ry(-1.816998130424195) q[4];
rz(1.1025467909759215) q[4];
ry(-1.5689258135454176) q[5];
rz(3.128421397007014) q[5];
ry(1.572275310409771) q[6];
rz(2.8300357332362807) q[6];
ry(0.0003021999425705246) q[7];
rz(-0.19792111901176349) q[7];
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
ry(1.3588543586325486) q[0];
rz(1.4437364663768335) q[0];
ry(-2.20742298264596) q[1];
rz(2.41348791773099) q[1];
ry(3.1388115184362233) q[2];
rz(-2.402775469121366) q[2];
ry(0.00022520034505529206) q[3];
rz(-0.8858524995785328) q[3];
ry(-0.00016681174666206494) q[4];
rz(0.2563252160394507) q[4];
ry(2.280952209967319) q[5];
rz(3.113885498953644) q[5];
ry(0.0029064941676706373) q[6];
rz(0.24033348974263102) q[6];
ry(0.9792849895924921) q[7];
rz(0.5058715225137914) q[7];
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
ry(-3.1378296133757697) q[0];
rz(0.9359370276273052) q[0];
ry(-1.5689306832399152) q[1];
rz(-3.0518405545956564) q[1];
ry(1.5695221021520454) q[2];
rz(0.8697917258459027) q[2];
ry(2.1737686055507957) q[3];
rz(-1.586408144223325) q[3];
ry(-3.02183030340875) q[4];
rz(2.827411245836819) q[4];
ry(-1.5655387453073746) q[5];
rz(-1.0474430903622836) q[5];
ry(-3.140267127974572) q[6];
rz(3.071608382467428) q[6];
ry(0.0002819372191273928) q[7];
rz(1.2238228502745092) q[7];
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
ry(-0.0014372233311350515) q[0];
rz(-1.431953973586209) q[0];
ry(-0.1091605527641466) q[1];
rz(3.0535410964027783) q[1];
ry(0.0016799029667717846) q[2];
rz(0.7775668280130345) q[2];
ry(-0.4683045858398822) q[3];
rz(-3.135291978570571) q[3];
ry(-3.1414116790678106) q[4];
rz(0.7084021615688153) q[4];
ry(2.1597879215457496) q[5];
rz(-2.0058136230129824) q[5];
ry(1.5663478281880183) q[6];
rz(-2.007388216733106) q[6];
ry(-0.04307726482625467) q[7];
rz(-0.10649746603480627) q[7];
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
ry(-0.0013916194875117603) q[0];
rz(0.11194486791280502) q[0];
ry(1.6479339982712287) q[1];
rz(0.13890204712838372) q[1];
ry(3.141338332824672) q[2];
rz(2.960913128681751) q[2];
ry(-1.5291612156210876) q[3];
rz(-2.940214068409054) q[3];
ry(3.140561712055994) q[4];
rz(-2.9253071364354266) q[4];
ry(-0.008430139020349614) q[5];
rz(2.9723770491826516) q[5];
ry(-3.140121454440861) q[6];
rz(-1.0182958846842354) q[6];
ry(-0.5244256024096918) q[7];
rz(0.11248805869807477) q[7];