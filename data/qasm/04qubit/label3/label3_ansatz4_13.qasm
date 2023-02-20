OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-0.7534963987197925) q[0];
rz(2.4487917218160145) q[0];
ry(-1.6751165055557509) q[1];
rz(1.312726944449006) q[1];
ry(3.045983975799317) q[2];
rz(-2.592791893880042) q[2];
ry(-2.231487267329576) q[3];
rz(-2.1262871147678126) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.5235426725380292) q[0];
rz(-0.16296025748680096) q[0];
ry(-2.4792880184186115) q[1];
rz(2.1167266304639876) q[1];
ry(0.8736899816123209) q[2];
rz(-0.9614872006527833) q[2];
ry(-1.9280886977782388) q[3];
rz(2.938925214280449) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.21612802483531188) q[0];
rz(1.8974831522112146) q[0];
ry(1.6008498843634842) q[1];
rz(-2.268193153422097) q[1];
ry(-1.4653343303834747) q[2];
rz(3.0183108463744075) q[2];
ry(2.1476564364928756) q[3];
rz(1.1792594766798468) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.1388140448977246) q[0];
rz(-2.3080739645261943) q[0];
ry(2.7284176206397786) q[1];
rz(2.5229327082049893) q[1];
ry(1.8450512157380343) q[2];
rz(-2.670427629434133) q[2];
ry(0.3406191313378761) q[3];
rz(1.627423455427103) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.0114087513028747) q[0];
rz(-2.379232648726771) q[0];
ry(1.9166626833394906) q[1];
rz(1.0838836985431612) q[1];
ry(0.9343821654736761) q[2];
rz(-2.0602787808914638) q[2];
ry(-0.5697168553771412) q[3];
rz(1.3988628594145798) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.7926000845197385) q[0];
rz(2.0862941076299677) q[0];
ry(-1.839970266645076) q[1];
rz(0.8719100656980929) q[1];
ry(-0.6708269503711568) q[2];
rz(-1.768761723515392) q[2];
ry(-2.4440019067416276) q[3];
rz(-1.2822416731448478) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.8009017606970334) q[0];
rz(-1.2930893349108095) q[0];
ry(0.6361811638556026) q[1];
rz(0.3741976708429702) q[1];
ry(-2.899817311297774) q[2];
rz(0.5097659451454755) q[2];
ry(-2.225640231168896) q[3];
rz(2.4264482285647775) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.7861124145991165) q[0];
rz(2.649595689148116) q[0];
ry(1.8411556082130107) q[1];
rz(0.7430075754510593) q[1];
ry(2.6464672728938776) q[2];
rz(2.287092032800876) q[2];
ry(-0.9603818089228905) q[3];
rz(0.8519638018650074) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.9012072288935209) q[0];
rz(-1.345800843386556) q[0];
ry(-2.768573424742134) q[1];
rz(-0.11424626249531399) q[1];
ry(-2.9755200870917933) q[2];
rz(2.2371030431703756) q[2];
ry(1.5150700131646244) q[3];
rz(1.1615206498101402) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.2734719065605915) q[0];
rz(2.830260419425426) q[0];
ry(-0.6928958557240188) q[1];
rz(2.6074207972515167) q[1];
ry(2.0700823911534734) q[2];
rz(0.4467582004819114) q[2];
ry(-0.9646290305052156) q[3];
rz(1.8489786545534717) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.06719730904427212) q[0];
rz(0.8731041810390249) q[0];
ry(0.5953285180615477) q[1];
rz(-1.089468296046804) q[1];
ry(-2.7930880462463765) q[2];
rz(1.4388590717826606) q[2];
ry(2.528241279601852) q[3];
rz(1.8921785245526768) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.729316085565987) q[0];
rz(-0.15182195045117247) q[0];
ry(-0.9779568600987787) q[1];
rz(0.35173246253039814) q[1];
ry(0.8349841763223002) q[2];
rz(1.2838850584338135) q[2];
ry(0.8020039755838457) q[3];
rz(-2.9193002086748066) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.1152952030988068) q[0];
rz(-0.9939347749679961) q[0];
ry(2.5500756359006402) q[1];
rz(-2.906986492869846) q[1];
ry(1.1010976064779614) q[2];
rz(2.606212976648892) q[2];
ry(2.3610780117259114) q[3];
rz(3.0068005531361974) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.0850002655760904) q[0];
rz(1.2594206836549604) q[0];
ry(-3.019333690583447) q[1];
rz(-2.152602658490361) q[1];
ry(-1.0572496490639838) q[2];
rz(0.7001023430917569) q[2];
ry(-0.17976978362007168) q[3];
rz(1.505116458253033) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.7632967324681331) q[0];
rz(-0.4104605366736589) q[0];
ry(-0.5127112828215763) q[1];
rz(0.03267984184384698) q[1];
ry(-2.754945873867272) q[2];
rz(1.333837592567762) q[2];
ry(0.6982287739062638) q[3];
rz(2.9528841016319234) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.1599359251631247) q[0];
rz(-1.3657708727065925) q[0];
ry(0.4945882342915557) q[1];
rz(-0.14849535292315963) q[1];
ry(2.9484073410428437) q[2];
rz(-1.516428546513537) q[2];
ry(-2.6309181868170533) q[3];
rz(2.9140814973937523) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.29749545995607) q[0];
rz(2.6502829165080577) q[0];
ry(-1.8422936359069437) q[1];
rz(-1.7797759750656557) q[1];
ry(-2.0265151669902117) q[2];
rz(-1.008822286534129) q[2];
ry(-0.26055337884650154) q[3];
rz(-1.6836166320382295) q[3];