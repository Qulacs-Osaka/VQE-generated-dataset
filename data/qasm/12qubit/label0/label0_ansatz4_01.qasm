OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-0.25127346309633886) q[0];
rz(1.568857611394234) q[0];
ry(-1.570625647127992) q[1];
rz(-3.1381712802894657) q[1];
ry(2.877094195638443) q[2];
rz(1.5718807669630337) q[2];
ry(-0.0029523825474073325) q[3];
rz(-2.702276218186643) q[3];
ry(-2.927583005326768) q[4];
rz(3.1414819947422497) q[4];
ry(1.6634082659727767e-05) q[5];
rz(-0.03298217058603825) q[5];
ry(1.5709521315177195) q[6];
rz(3.141421434456375) q[6];
ry(1.5739317005802507) q[7];
rz(0.000685607122798082) q[7];
ry(3.1415920758025253) q[8];
rz(-0.9565454980836258) q[8];
ry(-2.9219452138627013) q[9];
rz(-6.414973054180166e-06) q[9];
ry(0.0758500441274809) q[10];
rz(-1.5514854181952358) q[10];
ry(-3.039048686383701) q[11];
rz(3.1406359136355606) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.0602268260671056) q[0];
rz(1.555791513497839) q[0];
ry(1.5707620748828832) q[1];
rz(-2.352064889605487) q[1];
ry(2.903120290171424) q[2];
rz(1.5501490611242885) q[2];
ry(1.5678891797721404) q[3];
rz(-1.481510604284577) q[3];
ry(0.049805506895602036) q[4];
rz(-1.5687934197623599) q[4];
ry(-3.1415913476319823) q[5];
rz(2.957582635695459) q[5];
ry(2.07756818501465) q[6];
rz(-3.119065750734416) q[6];
ry(-0.826372593399842) q[7];
rz(0.06753097394559136) q[7];
ry(3.1415926015208733) q[8];
rz(-2.7915525063221978) q[8];
ry(0.05404620615134259) q[9];
rz(1.570703135125763) q[9];
ry(1.5711906201787134) q[10];
rz(-3.125915187146596) q[10];
ry(-2.7926378938004506) q[11];
rz(-1.580940854668447) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-0.00498777293232454) q[0];
rz(-3.1264423424237977) q[0];
ry(-3.1390500620087645) q[1];
rz(0.789742273022906) q[1];
ry(-3.111440131450723) q[2];
rz(2.618141041453565) q[2];
ry(-1.5699447428983275) q[3];
rz(0.0025760508751087297) q[3];
ry(-0.260878669517979) q[4];
rz(1.2918124552202632) q[4];
ry(-1.5707687391432923) q[5];
rz(0.8672732764860628) q[5];
ry(1.4843870351941817) q[6];
rz(-1.8300341567430927) q[6];
ry(-1.3170265045936744) q[7];
rz(-1.8377817765926334) q[7];
ry(1.570793868911191) q[8];
rz(5.300408631587638e-06) q[8];
ry(-0.2469732452597627) q[9];
rz(-3.014482464522429) q[9];
ry(-1.569680499440908) q[10];
rz(0.6415292798325885) q[10];
ry(-0.041640158499448354) q[11];
rz(2.0192276556588475) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-0.056023587635815275) q[0];
rz(0.0011791290582006692) q[0];
ry(3.0290968987923965) q[1];
rz(0.0008755749086314476) q[1];
ry(4.519459842811102e-06) q[2];
rz(0.5022694499020393) q[2];
ry(-1.570785462858393) q[3];
rz(-1.871313907581174) q[3];
ry(6.978019492496036e-06) q[4];
rz(1.8476815940682216) q[4];
ry(3.140990244013813) q[5];
rz(0.8672862726858419) q[5];
ry(-0.2196163703561922) q[6];
rz(1.5708310665300522) q[6];
ry(0.06961346265746116) q[7];
rz(-1.5707870996395004) q[7];
ry(-1.5707925715043767) q[8];
rz(7.421565825054734e-05) q[8];
ry(-3.141590138455166) q[9];
rz(-3.0148601225608243) q[9];
ry(3.1415900600060302) q[10];
rz(2.212060019115289) q[10];
ry(1.3032684593738961e-06) q[11];
rz(-2.009074235555294) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.5707959486928749) q[0];
rz(3.1415923350236614) q[0];
ry(1.5707944170167218) q[1];
rz(3.1415920554075507) q[1];
ry(-1.5707913822907078) q[2];
rz(3.141493788136449) q[2];
ry(3.1415732152640823) q[3];
rz(2.840684037916652) q[3];
ry(-1.5707927552594745) q[4];
rz(3.141519138714103) q[4];
ry(-1.5707608320180069) q[5];
rz(-1.570757642879225) q[5];
ry(1.5707966927817996) q[6];
rz(-3.1415815054629634) q[6];
ry(1.570797094631076) q[7];
rz(1.5878924289653933e-07) q[7];
ry(-1.5707968151980143) q[8];
rz(5.2270395109223245e-06) q[8];
ry(1.5707986032705157) q[9];
rz(-3.141591769212712) q[9];
ry(-1.570798149424154) q[10];
rz(3.141590591229203) q[10];
ry(-1.570797390192098) q[11];
rz(7.285275994830982e-07) q[11];