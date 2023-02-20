OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(1.5787360586222157) q[0];
rz(3.116390945570805) q[0];
ry(1.52665157388468) q[1];
rz(-1.559213831506029) q[1];
ry(-0.055129131547154486) q[2];
rz(-1.6519148679996745) q[2];
ry(-3.1298909281370078) q[3];
rz(1.8599865736470553) q[3];
ry(0.03085566524804495) q[4];
rz(1.556414161423059) q[4];
ry(1.571129974305505) q[5];
rz(-1.570833583058591) q[5];
ry(1.567648992938631) q[6];
rz(1.5707101966618824) q[6];
ry(1.6979285976672365) q[7];
rz(-3.0144186737399994) q[7];
ry(0.02207552179691857) q[8];
rz(-0.005532925323457503) q[8];
ry(-0.054043197261094626) q[9];
rz(1.4668740540744956) q[9];
ry(3.1266502828940568) q[10];
rz(0.8910632751822131) q[10];
ry(-3.1363624553596243) q[11];
rz(3.1214841653316276) q[11];
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
ry(-1.2869986505790845) q[0];
rz(-2.0624233382315795) q[0];
ry(-1.8787635097130417) q[1];
rz(3.0111768404955246) q[1];
ry(1.5261235560516926) q[2];
rz(2.446927851201348) q[2];
ry(-0.034675907835365116) q[3];
rz(1.1944465788800267) q[3];
ry(0.72507016432921) q[4];
rz(1.64245621473788) q[4];
ry(-2.5985279752483974) q[5];
rz(1.5706678253583826) q[5];
ry(-0.6481374230604198) q[6];
rz(1.5734587253892107) q[6];
ry(-3.11052431669434) q[7];
rz(0.7555822184317601) q[7];
ry(1.5747648285750375) q[8];
rz(-0.015801723771862508) q[8];
ry(-3.093155450268953) q[9];
rz(-1.5577832056231884) q[9];
ry(1.7110634099784174) q[10];
rz(-0.014161786034007423) q[10];
ry(-1.5755104768653565) q[11];
rz(-5.009201983759235e-05) q[11];
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
ry(-3.0653417688285725) q[0];
rz(-0.6017782714717805) q[0];
ry(0.13915415502802286) q[1];
rz(2.999239493358746) q[1];
ry(-3.140564040086509) q[2];
rz(2.406747519318548) q[2];
ry(0.06693561039568063) q[3];
rz(-1.7858089982764618) q[3];
ry(1.6898231893769378e-05) q[4];
rz(-1.497486072510554) q[4];
ry(-1.571141399608429) q[5];
rz(-1.07405283316742) q[5];
ry(3.11583057323926) q[6];
rz(0.0021547507965973267) q[6];
ry(1.6723181608874216) q[7];
rz(-0.9287207838595458) q[7];
ry(-0.0006022284940150305) q[8];
rz(-3.12586220462055) q[8];
ry(-3.0717262752933614) q[9];
rz(0.6634592978724844) q[9];
ry(-3.1198237836593106) q[10];
rz(2.3263893552766803) q[10];
ry(-1.5707126509290648) q[11];
rz(2.343947534749257) q[11];
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
ry(-1.9087344120823098) q[0];
rz(-1.8090899520168726) q[0];
ry(-3.1226403405789696) q[1];
rz(1.361310518810412) q[1];
ry(-1.615772916455435) q[2];
rz(-1.5172572610701636) q[2];
ry(-0.00039927063034448344) q[3];
rz(-2.873742692905996) q[3];
ry(-3.1415546126947924) q[4];
rz(2.939641342035) q[4];
ry(-3.1415529544955145) q[5];
rz(1.7205387344442373) q[5];
ry(-1.5737579132192094) q[6];
rz(1.228294268695949) q[6];
ry(-1.4191939080809846e-05) q[7];
rz(-2.47784152940836) q[7];
ry(1.5666983586693117) q[8];
rz(-1.8849260537748664) q[8];
ry(0.00023872292747206814) q[9];
rz(0.5936373724604616) q[9];
ry(3.141269886613763) q[10];
rz(-2.700917220227877) q[10];
ry(0.0003509154338932064) q[11];
rz(-1.0904778689810313) q[11];