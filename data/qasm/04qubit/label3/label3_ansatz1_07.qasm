OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-1.0442593932999147) q[0];
rz(1.0046380785671882) q[0];
ry(-2.543817585421408) q[1];
rz(2.88191833465177) q[1];
ry(-0.14197169079100913) q[2];
rz(-1.3262559278142678) q[2];
ry(2.215582730956707) q[3];
rz(-0.5078764984665206) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.998427477348825) q[0];
rz(0.11331781683614454) q[0];
ry(-2.801471580867454) q[1];
rz(1.4023933685082435) q[1];
ry(-0.8606555111147393) q[2];
rz(-0.8478283304364187) q[2];
ry(-1.5365690204047002) q[3];
rz(1.431516548744276) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.4101427281802028) q[0];
rz(2.9408818872633202) q[0];
ry(-1.7109465381111753) q[1];
rz(1.8185279846895435) q[1];
ry(-2.9674712307421625) q[2];
rz(0.32888259099908473) q[2];
ry(1.7714192063895897) q[3];
rz(0.3194122640001917) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.0709906026045397) q[0];
rz(-2.513146216365992) q[0];
ry(-1.2815170496498614) q[1];
rz(1.110974346663064) q[1];
ry(1.1952987629461171) q[2];
rz(1.3600510382950246) q[2];
ry(-2.3285251903751623) q[3];
rz(-2.449732070059366) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.6350325775962627) q[0];
rz(2.2241917651534666) q[0];
ry(-0.22082159133553514) q[1];
rz(-2.751820991993182) q[1];
ry(2.180802330166191) q[2];
rz(2.582310372339029) q[2];
ry(-2.241971465538076) q[3];
rz(-1.935079916267754) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.3002821778800195) q[0];
rz(0.6929938167628693) q[0];
ry(1.983910779953699) q[1];
rz(2.0221420856737) q[1];
ry(-2.4301095584980126) q[2];
rz(-3.0741899902675374) q[2];
ry(-2.464210009064626) q[3];
rz(-2.527218671674908) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.8414716794284491) q[0];
rz(-0.8801207541116192) q[0];
ry(1.5369434969087714) q[1];
rz(-1.4832896334429266) q[1];
ry(3.124810816535151) q[2];
rz(-0.026370120605317033) q[2];
ry(-2.244742015340136) q[3];
rz(0.12689834225071372) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.2642900688131267) q[0];
rz(-0.5273393060306542) q[0];
ry(0.0307404603858686) q[1];
rz(1.6371931946019234) q[1];
ry(-0.7476333912597575) q[2];
rz(-1.8053144754543853) q[2];
ry(2.892184696803589) q[3];
rz(-0.17245868858986188) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.5743036728633626) q[0];
rz(-0.17446176187369225) q[0];
ry(2.576304847923459) q[1];
rz(1.8603538344045099) q[1];
ry(1.7350743123833028) q[2];
rz(2.1482229880703514) q[2];
ry(0.051903799377715124) q[3];
rz(2.595382312962032) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.8067932918325895) q[0];
rz(-1.7751152377102848) q[0];
ry(-0.4498681328270493) q[1];
rz(-2.2543411491944205) q[1];
ry(-0.5849740867709644) q[2];
rz(-2.42760111170565) q[2];
ry(-2.3941981581888774) q[3];
rz(1.8718358969242694) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.095777997340035) q[0];
rz(1.9745493830753116) q[0];
ry(-2.5500869932715315) q[1];
rz(-2.86005726300065) q[1];
ry(-0.737890488151102) q[2];
rz(-2.5651999572262847) q[2];
ry(2.080716716421147) q[3];
rz(0.895209136303504) q[3];