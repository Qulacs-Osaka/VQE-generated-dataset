OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(0.14225147847046682) q[0];
rz(-2.8917345971698727) q[0];
ry(0.6318313028044841) q[1];
rz(0.3905331990794239) q[1];
ry(-0.09247771919706782) q[2];
rz(-2.5903299175061725) q[2];
ry(2.4355764617709927) q[3];
rz(-1.3779553820606436) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.49119109868320837) q[0];
rz(-2.7724227243944912) q[0];
ry(-2.8796211829437364) q[1];
rz(1.3906175129864498) q[1];
ry(3.001941990860445) q[2];
rz(0.9449201063896234) q[2];
ry(1.262316811972652) q[3];
rz(-0.36500684634586716) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.053362156160437) q[0];
rz(-0.5554606746021058) q[0];
ry(-2.724046674286655) q[1];
rz(1.3861267649115838) q[1];
ry(-1.8709159702435505) q[2];
rz(-2.7281676165311777) q[2];
ry(-0.41340537183826953) q[3];
rz(-0.6599182953647393) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.9791443825316954) q[0];
rz(0.5908481649972819) q[0];
ry(-2.377692100402774) q[1];
rz(2.916812182331894) q[1];
ry(1.0584121478853843) q[2];
rz(2.776240697340315) q[2];
ry(1.7731155188443681) q[3];
rz(-0.9960468601225116) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.1840645287311116) q[0];
rz(-1.8060049170268921) q[0];
ry(0.4883298988036508) q[1];
rz(-0.33841803728148356) q[1];
ry(-2.087143640280722) q[2];
rz(1.8124294871669775) q[2];
ry(2.98878941823925) q[3];
rz(1.301124353936034) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.8195490406777033) q[0];
rz(-2.4214971027891385) q[0];
ry(1.5813113310506155) q[1];
rz(2.552133058295106) q[1];
ry(-2.6737933284871773) q[2];
rz(0.7793476804543609) q[2];
ry(1.5762233119320546) q[3];
rz(-0.144473380798996) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.010845874564777) q[0];
rz(0.9780655392910969) q[0];
ry(3.129893470990796) q[1];
rz(0.9068114083629747) q[1];
ry(2.024729399752804) q[2];
rz(2.896485632622403) q[2];
ry(1.7745303096894656) q[3];
rz(2.983797863631184) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.3116662991321655) q[0];
rz(-0.5740676722784207) q[0];
ry(-2.193355357209553) q[1];
rz(0.04063598814608849) q[1];
ry(-1.9311201788816328) q[2];
rz(-3.0228646150576144) q[2];
ry(-3.056321845162376) q[3];
rz(0.24796996705607785) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.3346165733418075) q[0];
rz(0.6533583011680113) q[0];
ry(1.0304102838474725) q[1];
rz(2.8453116591600014) q[1];
ry(-1.9820015335543557) q[2];
rz(-1.1667948032378215) q[2];
ry(-2.7447860112035607) q[3];
rz(-1.5611711768925431) q[3];