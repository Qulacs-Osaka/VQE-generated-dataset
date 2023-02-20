OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-2.9875517700841483) q[0];
rz(1.5678134000253714) q[0];
ry(0.9621870359009453) q[1];
rz(2.242540744030288) q[1];
ry(0.5207741928734888) q[2];
rz(-2.737270170258898) q[2];
ry(0.7382167377391753) q[3];
rz(-2.0346058169764527) q[3];
ry(1.5752566693367482) q[4];
rz(-0.3500284905351645) q[4];
ry(-0.5901562060140901) q[5];
rz(-3.1181920671512597) q[5];
ry(1.689047913872287) q[6];
rz(-0.4794235011506342) q[6];
ry(2.5907554780417046) q[7];
rz(-0.18561158592351992) q[7];
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
ry(-0.3212313617719295) q[0];
rz(2.891635103440222) q[0];
ry(1.0328440806831798) q[1];
rz(-1.0917812923691912) q[1];
ry(0.01941171213579019) q[2];
rz(-0.406293347653725) q[2];
ry(-3.0963375993582964) q[3];
rz(1.384952060279958) q[3];
ry(-0.21652316465899155) q[4];
rz(3.108074408179194) q[4];
ry(-0.16615159097584978) q[5];
rz(3.0641853793346643) q[5];
ry(2.5636907562559466) q[6];
rz(-1.767203878568191) q[6];
ry(-0.13459466391750055) q[7];
rz(1.7396853176901361) q[7];
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
ry(2.7419941000194785) q[0];
rz(-1.1123050726744061) q[0];
ry(-0.9685786234402868) q[1];
rz(-1.6984718055464256) q[1];
ry(0.9256887791874303) q[2];
rz(0.5148028330027664) q[2];
ry(0.05019130731387654) q[3];
rz(1.4127987539086861) q[3];
ry(-1.4633050965031824) q[4];
rz(-0.6241528655573875) q[4];
ry(2.472865992033217) q[5];
rz(1.655631247209425) q[5];
ry(1.3785272517896563) q[6];
rz(0.08062581014943147) q[6];
ry(-1.5972606256933677) q[7];
rz(-1.424592120045189) q[7];
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
ry(-0.07014276571906919) q[0];
rz(-0.9646897319550352) q[0];
ry(-2.0107217660605996) q[1];
rz(-1.6081714822928443) q[1];
ry(-0.015771205192851134) q[2];
rz(-1.7128763175522206) q[2];
ry(2.971245681350596) q[3];
rz(0.10779091096948168) q[3];
ry(3.1334458473559623) q[4];
rz(2.3267600206021277) q[4];
ry(-3.126056785468697) q[5];
rz(0.12629084911409638) q[5];
ry(-0.2944146656255856) q[6];
rz(3.1324240020116125) q[6];
ry(-1.557297698410716) q[7];
rz(-1.578908127907788) q[7];
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
ry(-2.9386171581314824) q[0];
rz(-0.007669346917486597) q[0];
ry(-1.4402677306651073) q[1];
rz(-3.0369856672768685) q[1];
ry(2.1352182451230073) q[2];
rz(-2.8475667206919217) q[2];
ry(2.6046752166641762) q[3];
rz(-0.41568725837762305) q[3];
ry(-3.0510204076858343) q[4];
rz(-0.7438978298151295) q[4];
ry(2.377971225795372) q[5];
rz(-0.3734752268412649) q[5];
ry(-2.345884392189708) q[6];
rz(-0.3443292240686677) q[6];
ry(-1.8209908220003301) q[7];
rz(-0.35466159298192723) q[7];