OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
cx q[0],q[1];
rz(-0.006234301129923909) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.09296203763192118) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.0007131486779035876) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.06019768745824121) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.05825197077876796) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.0019983258857737595) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.051794945351119545) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.03675767251311493) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.013642773309388912) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.08182112323496164) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.08635275102378198) q[11];
cx q[10],q[11];
h q[0];
rz(2.6068450267656407) q[0];
h q[0];
h q[1];
rz(1.8357373054435095) q[1];
h q[1];
h q[2];
rz(1.5755498989974317) q[2];
h q[2];
h q[3];
rz(1.553206829873086) q[3];
h q[3];
h q[4];
rz(-1.577691669077388) q[4];
h q[4];
h q[5];
rz(-1.5687811220110583) q[5];
h q[5];
h q[6];
rz(1.5732573531320575) q[6];
h q[6];
h q[7];
rz(1.548316652970773) q[7];
h q[7];
h q[8];
rz(-1.551302402901066) q[8];
h q[8];
h q[9];
rz(1.573365554970458) q[9];
h q[9];
h q[10];
rz(1.5710515978000583) q[10];
h q[10];
h q[11];
rz(3.0664679838949533) q[11];
h q[11];
rz(-2.9104472171250393) q[0];
rz(-0.8851183689900504) q[1];
rz(-1.9457467178644636) q[2];
rz(-1.574723338590681) q[3];
rz(-1.5729976996277524) q[4];
rz(-1.5499496472157224) q[5];
rz(1.5573131142740382) q[6];
rz(-1.5677255085524022) q[7];
rz(1.5655658282231308) q[8];
rz(-1.5277233340947) q[9];
rz(-0.31573168872351964) q[10];
rz(0.3113314116908696) q[11];
cx q[0],q[1];
rz(0.1167604253702021) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.9761807781934304) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(-2.9794142072216765) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.7492349501355212) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.22992634953472355) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(0.45724150468852787) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(0.21357556921295542) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.8111194157950911) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.15322804759328504) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(1.7176394235700343) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(1.3591373682214491) q[11];
cx q[10],q[11];
h q[0];
rz(-0.9276901537480366) q[0];
h q[0];
h q[1];
rz(2.441395640214579) q[1];
h q[1];
h q[2];
rz(-0.24349380267596943) q[2];
h q[2];
h q[3];
rz(-0.8177340839953673) q[3];
h q[3];
h q[4];
rz(-2.609334919214247) q[4];
h q[4];
h q[5];
rz(-0.29960039449258163) q[5];
h q[5];
h q[6];
rz(-0.32832025480970406) q[6];
h q[6];
h q[7];
rz(-0.5404138240380543) q[7];
h q[7];
h q[8];
rz(0.8086485914896786) q[8];
h q[8];
h q[9];
rz(-0.22429281947065435) q[9];
h q[9];
h q[10];
rz(0.47521436569750747) q[10];
h q[10];
h q[11];
rz(-1.21255950045235) q[11];
h q[11];
rz(-1.3806035194664592) q[0];
rz(0.07330929829601428) q[1];
rz(-0.18884977972623038) q[2];
rz(0.0027058894966848607) q[3];
rz(-3.139339484868941) q[4];
rz(1.7901610338720295) q[5];
rz(0.0008836745926178146) q[6];
rz(0.0039794735362548555) q[7];
rz(0.0030092191975877687) q[8];
rz(-0.013475625811464933) q[9];
rz(-0.03678398885884683) q[10];
rz(-1.6152591918323633) q[11];
cx q[0],q[1];
rz(-0.7807638384220553) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.9252722630126683) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(0.12962012631212105) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.8467645327317623) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(2.687683354431527) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.37531845298218947) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(2.7035095988047555) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(0.9116966053059451) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.13264754846649449) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(-1.6429498592474239) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(2.8063153213032757) q[11];
cx q[10],q[11];
h q[0];
rz(-1.5923219319880413) q[0];
h q[0];
h q[1];
rz(0.08248495256342586) q[1];
h q[1];
h q[2];
rz(3.1198068375383143) q[2];
h q[2];
h q[3];
rz(1.9924608816634848) q[3];
h q[3];
h q[4];
rz(-1.4561643643016149) q[4];
h q[4];
h q[5];
rz(3.137723035873493) q[5];
h q[5];
h q[6];
rz(-3.1191366859511866) q[6];
h q[6];
h q[7];
rz(-1.4793888712028989) q[7];
h q[7];
h q[8];
rz(1.1272386663908753) q[8];
h q[8];
h q[9];
rz(-2.9930187255321514) q[9];
h q[9];
h q[10];
rz(2.9668116563333995) q[10];
h q[10];
h q[11];
rz(-1.4747982323520377) q[11];
h q[11];
rz(-1.2336527349058055) q[0];
rz(-0.08102119246213603) q[1];
rz(2.9397639606146257) q[2];
rz(-3.1283296963204523) q[3];
rz(3.1279292122722246) q[4];
rz(1.8054496672824718) q[5];
rz(-0.011492935213824461) q[6];
rz(3.127894882595301) q[7];
rz(-0.03258657883887932) q[8];
rz(-0.029377167553137427) q[9];
rz(3.1056164487785014) q[10];
rz(-2.7857983500162984) q[11];