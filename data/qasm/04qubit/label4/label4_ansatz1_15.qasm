OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.1579717499452566) q[0];
rz(-0.2450696016893135) q[0];
ry(-0.40878140085286463) q[1];
rz(-3.128707046059734) q[1];
ry(-2.067619227304405) q[2];
rz(0.6933519015892092) q[2];
ry(-2.7560670462183356) q[3];
rz(0.0406768667037003) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.3949499554845288) q[0];
rz(0.12112544259508783) q[0];
ry(0.005592083617996124) q[1];
rz(-2.4208092749921795) q[1];
ry(1.5391865128927282) q[2];
rz(0.7074770285831855) q[2];
ry(2.5893017669491503) q[3];
rz(-2.2280985353979634) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.17482154133602035) q[0];
rz(-1.0036102101952025) q[0];
ry(-0.7178994125573961) q[1];
rz(-1.2300627127085328) q[1];
ry(-1.4304123276696705) q[2];
rz(-1.5017326262041084) q[2];
ry(-1.9487178131319083) q[3];
rz(-1.6119096498472634) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.0521006350124036) q[0];
rz(1.7670223444724273) q[0];
ry(-2.4206051016144317) q[1];
rz(3.103947039206226) q[1];
ry(-1.2814527410057053) q[2];
rz(1.2206217986623367) q[2];
ry(-3.1337150019417384) q[3];
rz(-0.05469861155868827) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.8119030597630443) q[0];
rz(0.8003897884173954) q[0];
ry(0.7044451790864577) q[1];
rz(2.1377491761778233) q[1];
ry(1.4776877684196421) q[2];
rz(-1.1197212901384361) q[2];
ry(-2.6503759452421316) q[3];
rz(-2.779585574144104) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.7692075184893903) q[0];
rz(-1.7297778892144449) q[0];
ry(-0.45164670306444776) q[1];
rz(-2.5188157920215293) q[1];
ry(-1.1827727951528313) q[2];
rz(2.3335406676086623) q[2];
ry(1.3184194346809583) q[3];
rz(-0.45543643643534454) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.19198868798927848) q[0];
rz(0.3603047619692257) q[0];
ry(1.577116964024274) q[1];
rz(-2.5315599959608583) q[1];
ry(-0.9068091425591103) q[2];
rz(-2.736504620973538) q[2];
ry(-0.5133321878648759) q[3];
rz(-0.5225075243118732) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.6883770602748394) q[0];
rz(-1.162348999968037) q[0];
ry(1.8462934681070762) q[1];
rz(-0.27303101392080187) q[1];
ry(-0.3135387446561833) q[2];
rz(-1.2969563769166887) q[2];
ry(2.0449574355845943) q[3];
rz(-1.3261167851592999) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.0266802399356996) q[0];
rz(1.106565737520861) q[0];
ry(0.9490661299915816) q[1];
rz(-0.5017188271013115) q[1];
ry(0.9762660965826839) q[2];
rz(-1.537552891530355) q[2];
ry(-0.224687665179407) q[3];
rz(2.845828389584944) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.9273200496198557) q[0];
rz(1.2958097267877942) q[0];
ry(2.032080681378532) q[1];
rz(1.9426391008153518) q[1];
ry(-0.5726988080850938) q[2];
rz(0.9753757421154082) q[2];
ry(-2.9641091705666374) q[3];
rz(-1.142284272368394) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-3.1038609454066153) q[0];
rz(-0.22531706533730492) q[0];
ry(2.851649691433561) q[1];
rz(0.6936166290743) q[1];
ry(-2.4996954407566676) q[2];
rz(2.585278136104201) q[2];
ry(-0.33907030582641895) q[3];
rz(2.3162200481852833) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.3548217522924506) q[0];
rz(2.9860459574413136) q[0];
ry(2.8312442465981555) q[1];
rz(1.2087041657278315) q[1];
ry(-1.448435473776717) q[2];
rz(2.1923982306411625) q[2];
ry(1.907620421805861) q[3];
rz(2.417172950843478) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.02593025964070229) q[0];
rz(1.200789182364521) q[0];
ry(2.9343416642252858) q[1];
rz(0.5836361490226674) q[1];
ry(-0.12486515731019399) q[2];
rz(1.7279091080545133) q[2];
ry(0.28788690603113626) q[3];
rz(-1.6952038122495994) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.887463059219668) q[0];
rz(-2.027605796349124) q[0];
ry(-0.8877743165655393) q[1];
rz(-3.1412075352936686) q[1];
ry(2.5578759544429652) q[2];
rz(0.7743471149865735) q[2];
ry(2.0810066987163722) q[3];
rz(-1.0719925146088878) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.3821816254458588) q[0];
rz(-0.680152347066482) q[0];
ry(-0.565982747291936) q[1];
rz(0.7727778490501261) q[1];
ry(-2.0839018346070146) q[2];
rz(-1.0550495996617384) q[2];
ry(2.6962512756259955) q[3];
rz(-0.1943916541146452) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.5798272350943208) q[0];
rz(-0.3283877860858251) q[0];
ry(1.4456954107435278) q[1];
rz(-2.395927114967806) q[1];
ry(-1.979146957414299) q[2];
rz(-0.3751199977653458) q[2];
ry(-2.8686545905930685) q[3];
rz(-1.457858948636293) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.4628370510250317) q[0];
rz(0.698605245574815) q[0];
ry(1.0851756778275767) q[1];
rz(-1.9608492500263082) q[1];
ry(-2.5933874667338808) q[2];
rz(-2.5470681823950354) q[2];
ry(0.5130882827745173) q[3];
rz(3.057028707552346) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.5996400351385445) q[0];
rz(0.7205392270079658) q[0];
ry(-2.12607387632103) q[1];
rz(-1.1395539861849446) q[1];
ry(2.623646229247078) q[2];
rz(-2.4948197419457023) q[2];
ry(-2.9014260401836314) q[3];
rz(2.692260437814118) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.5595519000160509) q[0];
rz(-2.2443231791048657) q[0];
ry(2.194805822775547) q[1];
rz(-2.8284030845980275) q[1];
ry(1.7382571750053408) q[2];
rz(-0.8244545065206967) q[2];
ry(0.4862127839071136) q[3];
rz(2.5885690715722327) q[3];