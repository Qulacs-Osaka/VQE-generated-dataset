OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-3.1321757245631914) q[0];
rz(1.5744836782892586) q[0];
ry(0.0541958058777281) q[1];
rz(-1.5742444784445149) q[1];
ry(-3.1415926335688793) q[2];
rz(1.957556688045766) q[2];
ry(-3.1363807785754485) q[3];
rz(-1.5455580627121697) q[3];
ry(0.23113155161579185) q[4];
rz(1.5843118995124517) q[4];
ry(0.6973476239060181) q[5];
rz(-1.5483633945242024) q[5];
ry(1.5705107932389932) q[6];
rz(0.1023830089105351) q[6];
ry(-1.6968413304321437) q[7];
rz(3.140204522791764) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.7539929798991573) q[0];
rz(1.052274907180796) q[0];
ry(-1.5771837722515454) q[1];
rz(0.0021107185429851687) q[1];
ry(2.516677068149527e-07) q[2];
rz(-1.899165369834436) q[2];
ry(-0.2263773707258561) q[3];
rz(-3.063126332500018) q[3];
ry(0.03461399105427354) q[4];
rz(3.1277465601998697) q[4];
ry(-0.05604490241959237) q[5];
rz(-1.5927991073535175) q[5];
ry(-3.1402306235205826) q[6];
rz(1.881907869022796) q[6];
ry(1.5706195729486292) q[7];
rz(-0.3935723669031717) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-3.1396022860374564) q[0];
rz(-0.535308573242477) q[0];
ry(-1.4678167925202734) q[1];
rz(-1.5758726927597353) q[1];
ry(-3.141592554829403) q[2];
rz(0.9659570307653428) q[2];
ry(3.1349264300408257) q[3];
rz(1.6455242120240408) q[3];
ry(-0.3528317816412879) q[4];
rz(-1.5708376553559509) q[4];
ry(1.5705109797532453) q[5];
rz(3.076928653211748) q[5];
ry(1.3103599606926313) q[6];
rz(3.0139541573509314) q[6];
ry(0.6962610762765082) q[7];
rz(-0.36189061794204436) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(0.04780881080054655) q[0];
rz(-3.104150057616881) q[0];
ry(-2.8258162498750283) q[1];
rz(0.003899410792698749) q[1];
ry(-3.291162276042314e-08) q[2];
rz(2.4930663778811395) q[2];
ry(-1.1966859516869883) q[3];
rz(-3.1383122741826246) q[3];
ry(-1.8061075278296286) q[4];
rz(-7.461375570905204e-05) q[4];
ry(-1.5709453927493922) q[5];
rz(3.141550920147209) q[5];
ry(0.3134720423961271) q[6];
rz(-1.0958202287042902) q[6];
ry(2.5150788047864987) q[7];
rz(2.341688962473531) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(0.005620905128220777) q[0];
rz(-1.5894879396205663) q[0];
ry(-3.1186189824657875) q[1];
rz(1.573455263518584) q[1];
ry(1.1267305932705084e-07) q[2];
rz(-2.7150103274809805) q[2];
ry(-0.0909753339720254) q[3];
rz(-1.572289364831327) q[3];
ry(-1.6391290214627343) q[4];
rz(-1.571616083114295) q[4];
ry(1.5698427444196006) q[5];
rz(-3.1339139852144196) q[5];
ry(1.5704169780014647) q[6];
rz(0.0005176351686673186) q[6];
ry(1.57039649779895) q[7];
rz(3.1411431137189227) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-2.8836975486246392) q[0];
rz(0.0011343578509726369) q[0];
ry(2.6876196830657104) q[1];
rz(3.1362728064632956) q[1];
ry(-3.141592059481754) q[2];
rz(-0.2178491204125148) q[2];
ry(-0.12235200751546838) q[3];
rz(-0.0029058125876275653) q[3];
ry(2.405332557985269) q[4];
rz(3.140182847133327) q[4];
ry(1.5700881985435853) q[5];
rz(-1.6007908550226204) q[5];
ry(-1.58780104647449) q[6];
rz(-0.03932964902728475) q[6];
ry(1.5118167475593614) q[7];
rz(-3.1188954354868033) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(0.4065484078505768) q[0];
rz(0.10893254820684016) q[0];
ry(-2.24117658798171) q[1];
rz(2.8112466949258987) q[1];
ry(1.5707967341725686) q[2];
rz(3.1415921949295766) q[2];
ry(1.8471990100528963) q[3];
rz(-2.451367870362391) q[3];
ry(1.6680038595085973) q[4];
rz(-0.8437975340325502) q[4];
ry(3.0995655735349397) q[5];
rz(-2.980916541733638) q[5];
ry(-0.03126053782770288) q[6];
rz(0.9003651161802043) q[6];
ry(-0.04124899964673823) q[7];
rz(-1.0469745906536345) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(5.764939339071566e-09) q[0];
rz(0.3032803157517386) q[0];
ry(-3.141592331671889) q[1];
rz(0.08027495251562804) q[1];
ry(1.5707959102867024) q[2];
rz(-2.732577471595668) q[2];
ry(-1.67084523106098e-07) q[3];
rz(2.8599596663051243) q[3];
ry(-3.1415925957888007) q[4];
rz(-0.4350810701673872) q[4];
ry(1.1108029545794929e-07) q[5];
rz(-2.9239633865382566) q[5];
ry(-3.14159218839977) q[6];
rz(-1.8720378873536079) q[6];
ry(-3.141592137695866) q[7];
rz(-0.6159223359146946) q[7];