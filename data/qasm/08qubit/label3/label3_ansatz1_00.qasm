OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(1.6074255323254842) q[0];
rz(1.570796383154315) q[0];
ry(-1.666193466807649) q[1];
rz(1.5708034012897247) q[1];
ry(-1.0130758711797614) q[2];
rz(3.1415915218000623) q[2];
ry(-0.7399831029851116) q[3];
rz(-3.1415919325844084) q[3];
ry(2.470685119729587) q[4];
rz(-3.1415920183063974) q[4];
ry(-1.5707968238997418) q[5];
rz(-0.06504775048497058) q[5];
ry(1.5707963538823604) q[6];
rz(0.34966550253101314) q[6];
ry(-1.7244922490867272) q[7];
rz(-1.5707956088803228) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(1.5707892719628747) q[0];
rz(-1.59772612039387) q[0];
ry(1.5707961408774889) q[1];
rz(0.9160520558058821) q[1];
ry(1.7173423733337483) q[2];
rz(-1.5707971227227504) q[2];
ry(-1.0197149057546682) q[3];
rz(1.5707980544884883) q[3];
ry(1.570795983024529) q[4];
rz(1.4171903283180967) q[4];
ry(1.4524247466464315) q[5];
rz(-1.4241510456036457) q[5];
ry(2.0701712540501505) q[6];
rz(0.6168028247598346) q[6];
ry(3.0085873369375955) q[7];
rz(1.5707967555007496) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.6091282148074244) q[0];
rz(-2.583692749717752) q[0];
ry(2.067640343260899) q[1];
rz(1.5099419516804202) q[1];
ry(-2.604933233035941) q[2];
rz(3.141590485451309) q[2];
ry(0.15384639588265436) q[3];
rz(-2.38800711382936e-06) q[3];
ry(2.904163787493649) q[4];
rz(2.120006961014547) q[4];
ry(-0.7858422982469211) q[5];
rz(-1.2661125989963553) q[5];
ry(-1.5707961919985352) q[6];
rz(1.5707962034093577) q[6];
ry(1.6254987809614299) q[7];
rz(-1.5707955638069535) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-3.1415909071496686) q[0];
rz(2.411847506337301) q[0];
ry(-9.773392680253435e-07) q[1];
rz(0.6278927288843108) q[1];
ry(-1.5707955612382103) q[2];
rz(-2.8679920222391178) q[2];
ry(1.5707955858560094) q[3];
rz(-0.09180862237005451) q[3];
ry(4.427057813671809e-07) q[4];
rz(-1.9639526800137599) q[4];
ry(-6.409680361782648e-07) q[5];
rz(-2.331005671274103) q[5];
ry(-1.5707965324164967) q[6];
rz(0.35314703042326534) q[6];
ry(1.5707968333749576) q[7];
rz(2.4886609682550387) q[7];